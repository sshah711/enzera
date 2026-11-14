"""Retrain the theft model with confirmed inspection outcomes (closing the feedback loop).

Inspectors record outcomes in the app (case status theft_confirmed / normal). Those outcomes are new,
high-quality labels. This script:

  1. loads the labels: from the running backend (GET /labels, manager login) or from a CSV (customer_id,label)
  2. rebuilds the SAME train/validation/test split as ml/train.py from data/raw/data.csv
  3. adds the labelled customers to the training data (an inspection outcome overrides the SGCC label;
     customers in the TEST split are never added, so the test stays a fair, fixed yardstick)
  4. retrains XGBoost with the same settings, tunes the threshold on validation, evaluates on test
  5. compares with the current model and writes reports/retrain_report.json + models/candidate_model.joblib
  6. with --promote, replaces the deployed model ONLY if the candidate's test PR-AUC is not worse

Examples:
    python -m ml.retrain --api https://enzera-api.onrender.com --user manager --password manager123
    python -m ml.retrain --labels-csv my_labels.csv --promote
"""
import argparse
import json
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ml.features import compute_features, split_wide
from ml.scoring import fit_calibrator
from ml.train import ROOT, SEED, XGB_PARAMS, best_f1_threshold, evaluate, make_splits

MODEL = ROOT / "models/enzera_model.joblib"
CANDIDATE = ROOT / "models/candidate_model.joblib"


def load_labels(args) -> pd.DataFrame:
    if args.labels_csv:
        df = pd.read_csv(args.labels_csv)
    else:
        import requests
        base = args.api.rstrip("/")
        tok = requests.post(f"{base}/auth/login", json={"username": args.user, "password": args.password},
                            timeout=90).json()["token"]
        df = pd.DataFrame(requests.get(f"{base}/labels", headers={"Authorization": f"Bearer {tok}"}, timeout=90).json())
    if df.empty:
        raise SystemExit("No confirmed inspection outcomes yet - nothing to retrain with.")
    return df[["customer_id", "label"]].astype({"customer_id": str, "label": int})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels-csv")
    src.add_argument("--api", help="backend URL, e.g. http://localhost:8000")
    ap.add_argument("--user", default="manager")
    ap.add_argument("--password")
    ap.add_argument("--data", default=str(ROOT / "data/raw/data.csv"))
    ap.add_argument("--promote", action="store_true", help="deploy the candidate if it is not worse")
    args = ap.parse_args()

    new = load_labels(args)
    ids, labels, dates, values = split_wide(pd.read_csv(args.data))
    tr, tr_fit, val, te, demo = make_splits(labels)
    pos = {cid: i for i, cid in enumerate(ids)}
    test_set = set(te.tolist())

    y = labels.copy()
    added, skipped_test, unknown = [], 0, 0
    for cid, lab in zip(new.customer_id, new.label):
        i = pos.get(cid)
        if i is None:
            unknown += 1
        elif i in test_set:
            skipped_test += 1
        else:
            y[i] = lab
            added.append(i)
    train_idx = np.unique(np.r_[tr_fit, added])
    changed = int(sum(labels[i] != y[i] for i in added))
    print(f"{len(new)} inspection labels: {len(added)} used ({changed} differ from the SGCC label), "
          f"{skipped_test} skipped (test split), {unknown} unknown IDs")

    X = compute_features(values, dates).to_numpy()
    neg, npos = np.bincount(y[train_idx])
    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=neg / npos, eval_metric="aucpr",
                          n_jobs=-1, random_state=SEED).fit(X[train_idx], y[train_idx])
    p_val = model.predict_proba(X[val])[:, 1]
    thr = best_f1_threshold(y[val], p_val)
    cand = evaluate(y[te], model.predict_proba(X[te])[:, 1], thr)

    current = json.loads((ROOT / "reports/metrics.json").read_text())["results"]["xgboost"]
    keys = ["precision", "recall", "f1", "roc_auc", "pr_auc", "recall_at_top10"]
    print(f"\n{'metric':16s}{'current':>10s}{'candidate':>11s}")
    for k in keys:
        print(f"{k:16s}{current[k]:10.4f}{cand[k]:11.4f}")
    better = cand["pr_auc"] >= current["pr_auc"]

    bundle = joblib.load(MODEL)
    bundle.update({"model": model, "threshold": thr,
                   "calibrator": fit_calibrator(p_val, y[val]),
                   "version": datetime.now(timezone.utc).strftime("xgb-retrain-%Y%m%d-%H%M")})
    joblib.dump(bundle, CANDIDATE)
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "candidate_version": bundle["version"], "labels_received": int(len(new)),
              "labels_used": len(added), "labels_changed": changed, "skipped_test": skipped_test,
              "unknown_ids": unknown, "train_rows": int(len(train_idx)),
              "current": {k: current[k] for k in keys}, "candidate": {k: cand[k] for k in keys},
              "candidate_not_worse": bool(better), "promoted": False}

    if args.promote and better:
        shutil.copy(MODEL, ROOT / "models/enzera_model.prev.joblib")
        shutil.copy(CANDIDATE, MODEL)
        m = json.loads((ROOT / "reports/metrics.json").read_text())
        m["results"]["xgboost"] = cand
        m["model_version"] = bundle["version"]
        m["retraining"] = report
        (ROOT / "reports/metrics.json").write_text(json.dumps(m, indent=2))
        report["promoted"] = True
        print("\nPromoted: models/enzera_model.joblib replaced (previous kept as enzera_model.prev.joblib). "
              "Commit and push to redeploy.")
    elif args.promote:
        print("\nNot promoted: the candidate's test PR-AUC is lower than the current model's.")
    else:
        print(f"\nCandidate saved to {CANDIDATE.name} (not deployed). Re-run with --promote to deploy it if not worse.")
    (ROOT / "reports/retrain_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

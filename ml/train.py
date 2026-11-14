"""ENZERA training pipeline: one command from raw SGCC CSV to a deployable model.

    python -m ml.train --data data/raw/data.csv

Produces:
    models/enzera_model.joblib      model bundle loaded by the backend
    reports/metrics.json            all results (shown on the Model performance page)
    reports/*.png                   EDA + evaluation charts for the report
    data/demo/demo_readings.csv.gz  unseen customers loaded into the deployed app
    data/demo/demo_labels.csv       their true labels (kept hidden from the app UI)
    data/samples/sample_upload.csv  small file for testing the upload feature
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.features import FEATURE_INFO, FEATURE_NAMES, clean, compute_features, monthly_means, split_wide
from ml.scoring import shap_values

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def top_k_capture(y, scores, frac=0.10):
    """If inspectors visit only the top `frac` riskiest customers, what share of
    all thieves do they catch, and how many visits find a thief?"""
    k = max(1, int(len(scores) * frac))
    top = np.argsort(-scores)[:k]
    caught = y[top].sum()
    return {"recall_at_top10": float(caught / max(1, y.sum())),
            "precision_at_top10": float(caught / k)}


def evaluate(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "threshold": float(threshold),
    }
    out.update(top_k_capture(y, prob))
    return out


def best_f1_threshold(y, prob):
    p, r, t = precision_recall_curve(y, prob)
    f1 = 2 * p * r / (p + r + 1e-9)
    return float(t[np.argmax(f1[:-1])])


def save_eda(values, labels, dates, out):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = np.bincount(labels)
    ax.bar(["Normal", "Theft"], counts, color=["#4C78A8", "#E45756"])
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}\n({c / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Class balance (train split)")
    ax.set_ylabel("Customers")
    fig.tight_layout(); fig.savefig(out / "eda_class_balance.png", dpi=120); plt.close(fig)

    miss = np.isnan(values).mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([miss[labels == 0], miss[labels == 1]], bins=30, label=["Normal", "Theft"],
            color=["#4C78A8", "#E45756"], density=True)
    ax.set_title("Share of missing days per customer")
    ax.set_xlabel("Missing ratio"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "eda_missing.png", dpi=120); plt.close(fig)

    x = clean(values)
    m = monthly_means(x, dates)
    months = dates.to_period("M").unique().astype(str)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(months, m[labels == 0].mean(axis=0), label="Normal", color="#4C78A8")
    ax.plot(months, m[labels == 1].mean(axis=0), label="Theft", color="#E45756")
    ax.set_xticks(months[::3]); ax.tick_params(axis="x", rotation=45)
    ax.set_title("Average monthly consumption: theft vs normal")
    ax.set_ylabel("kWh per day"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "eda_avg_consumption.png", dpi=120); plt.close(fig)

    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
    for row, (lab, name, col) in enumerate([(0, "Normal", "#4C78A8"), (1, "Theft", "#E45756")]):
        idx = rng.choice(np.flatnonzero(labels == lab), 3, replace=False)
        for ax, i in zip(axes[row], idx):
            ax.plot(m[i], color=col); ax.set_title(f"{name} customer", fontsize=9)
    fig.suptitle("Sample monthly consumption curves")
    fig.tight_layout(); fig.savefig(out / "eda_sample_curves.png", dpi=120); plt.close(fig)


def save_eval_plots(y, probs, threshold, xgb_prob, mean_abs_shap, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    a1.set_title("ROC curve (test)"); a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate"); a1.legend()
    a2.set_title("Precision-recall curve (test)"); a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.legend()
    fig.tight_layout(); fig.savefig(out / "roc_pr.png", dpi=120); plt.close(fig)

    cm = confusion_matrix(y, (xgb_prob >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Normal", "Theft"]); ax.set_yticks([0, 1], ["Normal", "Theft"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("XGBoost confusion matrix (test)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=120); plt.close(fig)

    order = np.argsort(mean_abs_shap)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([FEATURE_INFO[FEATURE_NAMES[i]] for i in order], mean_abs_shap[order], color="#4C78A8")
    ax.set_title("Global feature importance (mean |SHAP|, test)")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout(); fig.savefig(out / "shap_importance.png", dpi=120); plt.close(fig)


def maybe_mlflow(results, params):
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed - skipping experiment logging")
        return
    mlflow.set_experiment("enzera")
    for name, metrics in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params.get(name, {}))
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
    print("Logged runs to MLflow (run `mlflow ui` to view)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/data.csv"))
    ap.add_argument("--demo-size", type=int, default=1000,
                    help="customers from the unseen demo split loaded into the app")
    args = ap.parse_args()

    (ROOT / "models").mkdir(exist_ok=True)
    reports = ROOT / "reports"; reports.mkdir(exist_ok=True)
    (ROOT / "data/demo").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/samples").mkdir(parents=True, exist_ok=True)

    print("Loading", args.data)
    ids, labels, dates, values = split_wide(pd.read_csv(args.data))
    print(f"{len(ids):,} customers x {len(dates):,} days, theft rate {labels.mean():.2%}")

    # 70 / 15 / 15 split: train, test (metrics), demo (never seen; shown in the app)
    idx = np.arange(len(ids))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    te, demo = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=SEED)
    tr_fit, val = train_test_split(tr, test_size=0.15, stratify=labels[tr], random_state=SEED)

    print("EDA charts (train split only)...")
    save_eda(values[tr], labels[tr], dates, reports)

    print("Computing features...")
    feats = compute_features(values, dates)
    X = feats.to_numpy(); y = labels

    neg, pos = np.bincount(y[tr_fit])
    spw = neg / pos
    params = {
        "logistic_regression": {"class_weight": "balanced", "max_iter": 2000},
        "random_forest": {"n_estimators": 300, "class_weight": "balanced_subsample", "min_samples_leaf": 2},
        "xgboost": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": round(spw, 3)},
    }
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(**params["logistic_regression"])),
        "random_forest": RandomForestClassifier(**params["random_forest"], n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(**params["xgboost"], eval_metric="aucpr", n_jobs=-1, random_state=SEED),
    }

    results, test_probs, thresholds = {}, {}, {}
    for name, model in models.items():
        print("Training", name)
        model.fit(X[tr_fit], y[tr_fit])
        thr = best_f1_threshold(y[val], model.predict_proba(X[val])[:, 1])  # tuned on validation, not test
        prob = model.predict_proba(X[te])[:, 1]
        results[name] = evaluate(y[te], prob, thr)
        test_probs[name], thresholds[name] = prob, thr
        print("  ", {k: round(v, 3) for k, v in results[name].items()})

    # Unsupervised abnormal-consumption detector, fitted on normal customers only
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1)
    iso.fit(X[tr_fit][y[tr_fit] == 0])
    raw_tr = -iso.score_samples(X[tr_fit])
    iso_range = (float(np.percentile(raw_tr, 1)), float(np.percentile(raw_tr, 99)))
    iso_te = -iso.score_samples(X[te])
    results["isolation_forest"] = {"roc_auc": float(roc_auc_score(y[te], iso_te)),
                                   "pr_auc": float(average_precision_score(y[te], iso_te)),
                                   **top_k_capture(y[te], iso_te)}

    bundle = {
        "model": models["xgboost"], "iforest": iso, "iso_range": iso_range,
        "threshold": thresholds["xgboost"], "feature_names": FEATURE_NAMES,
        "version": datetime.now(timezone.utc).strftime("xgb-%Y%m%d-%H%M"),
    }
    joblib.dump(bundle, ROOT / "models/enzera_model.joblib")

    contribs = shap_values(bundle, feats.iloc[te])
    mean_abs = np.abs(contribs).mean(axis=0)
    save_eval_plots(y[te], test_probs, thresholds["xgboost"], test_probs["xgboost"], mean_abs, reports)

    metrics = {
        "model_version": bundle["version"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"customers": int(len(ids)), "days": int(len(dates)),
                    "theft_rate": float(labels.mean()),
                    "split": {"train": int(len(tr)), "test": int(len(te)), "demo": int(len(demo))}},
        "main_model": "xgboost",
        "results": results,
        "feature_importance": {FEATURE_INFO[f]: float(v) for f, v in zip(FEATURE_NAMES, mean_abs)},
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    maybe_mlflow(results, params)

    # Demo customers for the deployed app: unseen by the model
    demo_ids, _ = train_test_split(demo, train_size=min(args.demo_size, len(demo) - 2),
                                   stratify=labels[demo], random_state=SEED)
    demo_ids = np.sort(demo_ids)
    wide = pd.DataFrame(np.round(values[demo_ids], 3), columns=dates.strftime("%Y-%m-%d"))
    wide.insert(0, "CONS_NO", ids[demo_ids])
    wide.to_csv(ROOT / "data/demo/demo_readings.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"customer_id": ids[demo_ids], "flag": labels[demo_ids]}).to_csv(
        ROOT / "data/demo/demo_labels.csv", index=False)

    # Small upload sample in long format: 50 customers, last 365 days
    sample = wide.iloc[:50, [0] + list(range(max(1, wide.shape[1] - 365), wide.shape[1]))]
    long = sample.melt(id_vars="CONS_NO", var_name="date", value_name="consumption")
    long.rename(columns={"CONS_NO": "customer_id"}).to_csv(
        ROOT / "data/samples/sample_upload.csv", index=False)

    x = results["xgboost"]
    print(f"\nDone. XGBoost test: recall {x['recall']:.3f}, precision {x['precision']:.3f}, "
          f"AUC {x['roc_auc']:.3f}, theft caught in top 10%: {x['recall_at_top10']:.1%}")


if __name__ == "__main__":
    main()

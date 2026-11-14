"""Scoring + explanations, shared by training and the backend.

Explanations use XGBoost's built-in TreeSHAP (pred_contribs=True), which gives
exact SHAP values for tree models without needing the `shap` package at runtime.
"""
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from ml.features import FEATURE_INFO, FEATURE_NAMES


def load_bundle(path):
    return joblib.load(path)


def _fmt(name, value):
    if name.endswith("_ratio") or name in ("max_monthly_drop",):
        return f"{value * 100:.0f}%"
    if name == "longest_zero_run":
        return f"{int(value)} days"
    return f"{value:.2f}"


def _contribs(bundle, feats: pd.DataFrame) -> np.ndarray:
    booster = bundle["model"].get_booster()
    dm = xgb.DMatrix(feats[FEATURE_NAMES].to_numpy(), feature_names=FEATURE_NAMES)
    return booster.predict(dm, pred_contribs=True)  # one column per feature + bias (last)


def shap_values(bundle, feats: pd.DataFrame) -> np.ndarray:
    return _contribs(bundle, feats)[:, :-1]


def explain_one(bundle, feats_row: pd.DataFrame) -> dict:
    """All SHAP contributions for one customer (for the waterfall chart).

    log-odds = base_value + sum(contributions);  risk = 1 / (1 + exp(-log-odds))
    """
    c = _contribs(bundle, feats_row)[0]
    rows = [{"feature": f, "label": FEATURE_INFO[f], "value": _fmt(f, float(feats_row[f].iloc[0])),
             "raw_value": float(feats_row[f].iloc[0]), "impact": round(float(c[j]), 4)}
            for j, f in enumerate(FEATURE_NAMES)]
    rows.sort(key=lambda r: -abs(r["impact"]))
    return {"base_value": round(float(c[-1]), 4), "log_odds": round(float(c.sum()), 4), "contributions": rows}


def band(score, threshold):
    if score >= threshold:
        return "High"
    if score >= threshold / 2:
        return "Medium"
    return "Low"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(val_scores, val_labels):
    """Platt scaling: a 1-feature logistic regression on the logit of the raw score, fitted on VALIDATION data.

    Class weights make raw XGBoost scores too high (a 0.5 score is far from a 50% theft chance); this maps them
    back to real probabilities. Chosen over isotonic regression, whose top step reached 100% on only
    4 validation customers; Platt is smooth and never claims 0% or 100%.
    """
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1e6).fit(_logit(val_scores), val_labels)


def calibrated(bundle, risk):
    """Raw risk score -> estimated probability of theft (calibrated on the validation split)."""
    cal = bundle.get("calibrator")
    return cal.predict_proba(_logit(risk))[:, 1] if cal is not None else np.asarray(risk)


def score_features(bundle, feats: pd.DataFrame, ids, top_k=3) -> pd.DataFrame:
    X = feats[FEATURE_NAMES]
    risk = bundle["model"].predict_proba(X.to_numpy())[:, 1]
    prob = calibrated(bundle, risk)

    iso = bundle["iforest"]
    raw = -iso.score_samples(X.to_numpy())
    lo, hi = bundle["iso_range"]
    anomaly = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)

    contribs = shap_values(bundle, feats)
    rows = []
    for i, cid in enumerate(ids):
        order = np.argsort(-contribs[i])[:top_k]
        reasons = [
            {
                "feature": FEATURE_NAMES[j],
                "label": FEATURE_INFO[FEATURE_NAMES[j]],
                "value": _fmt(FEATURE_NAMES[j], float(X.iloc[i, j])),
                "impact": round(float(contribs[i, j]), 4),
            }
            for j in order if contribs[i, j] > 0
        ]
        rows.append({
            "customer_id": str(cid),
            "risk_score": round(float(risk[i]), 4),
            "theft_probability": round(float(prob[i]), 4),
            "anomaly_score": round(float(anomaly[i]), 4),
            "risk_band": band(risk[i], bundle["threshold"]),
            "reasons": reasons,
        })
    return pd.DataFrame(rows)


def drift_report(bundle, feats: pd.DataFrame) -> list:
    """Population Stability Index of each feature vs the training data.

    PSI = sum((new% - train%) * ln(new% / train%)) over the training deciles.
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
    """
    ref = bundle.get("drift_ref")
    if not ref or len(feats) == 0:
        return []
    out = []
    for name in FEATURE_NAMES:
        edges, p_ref = np.array(ref[name]["edges"]), np.array(ref[name]["props"])
        counts = np.bincount(np.searchsorted(edges, feats[name].to_numpy(), side="right"), minlength=len(p_ref))
        p_new = counts / counts.sum()
        a, b = np.clip(p_new, 1e-4, None), np.clip(p_ref, 1e-4, None)
        psi = float(np.sum((a - b) * np.log(a / b)))
        status = "stable" if psi < 0.1 else "moderate shift" if psi < 0.25 else "major shift"
        out.append({"feature": name, "label": FEATURE_INFO[name], "psi": round(psi, 4), "status": status})
    return sorted(out, key=lambda r: -r["psi"])

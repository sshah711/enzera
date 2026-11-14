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


def shap_values(bundle, feats: pd.DataFrame) -> np.ndarray:
    booster = bundle["model"].get_booster()
    dm = xgb.DMatrix(feats[FEATURE_NAMES].to_numpy(), feature_names=FEATURE_NAMES)
    contribs = booster.predict(dm, pred_contribs=True)
    return contribs[:, :-1]  # last column is the bias term


def band(score, threshold):
    if score >= threshold:
        return "High"
    if score >= threshold / 2:
        return "Medium"
    return "Low"


def score_features(bundle, feats: pd.DataFrame, ids, top_k=3) -> pd.DataFrame:
    X = feats[FEATURE_NAMES]
    risk = bundle["model"].predict_proba(X.to_numpy())[:, 1]

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
            "anomaly_score": round(float(anomaly[i]), 4),
            "risk_band": band(risk[i], bundle["threshold"]),
            "reasons": reasons,
        })
    return pd.DataFrame(rows)

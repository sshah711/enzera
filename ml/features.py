"""Shared data loading, cleaning and feature engineering for ENZERA.

Used by BOTH training (ml/train.py) and serving (backend/main.py), so the
deployed app always processes data exactly the way the model was trained.
"""
import numpy as np
import pandas as pd

EPS = 1e-6

# Feature name -> plain-language label shown to inspectors as a "reason".
FEATURE_INFO = {
    "mean_daily": "Average daily consumption (kWh)",
    "std_daily": "Day-to-day variation (kWh)",
    "cv_daily": "Relative variability of consumption",
    "max_daily": "Highest daily consumption (kWh)",
    "missing_ratio": "Share of days with missing readings",
    "zero_ratio": "Share of days with zero consumption",
    "longest_zero_run": "Longest run of zero-consumption days",
    "low_day_ratio": "Share of days far below the usual level",
    "spike_ratio": "Share of days with unusual spikes",
    "trend_slope": "Long-term trend of monthly usage",
    "max_monthly_drop": "Largest month-to-month drop",
    "monthly_cv": "Variability between months",
    "recent_vs_past": "Recent usage compared with earlier usage",
    "weekend_ratio": "Weekend vs weekday usage",
}
FEATURE_NAMES = list(FEATURE_INFO)


# ---------------------------------------------------------------- loading
def split_wide(df: pd.DataFrame):
    """SGCC wide format: CONS_NO, FLAG (optional), then one column per day.

    Returns ids, labels (or None), sorted DatetimeIndex, float32 matrix.
    """
    id_col = "CONS_NO" if "CONS_NO" in df.columns else df.columns[0]
    other = [c for c in df.columns if c not in (id_col, "FLAG")]
    parsed = pd.to_datetime(pd.Index(other), errors="coerce", format="mixed")
    keep = [(d, c) for d, c in zip(parsed, other) if not pd.isna(d)]
    if not keep:
        raise ValueError("No date columns found")
    keep.sort(key=lambda t: t[0])  # SGCC date columns are NOT in date order
    dates = pd.DatetimeIndex([d for d, _ in keep])
    cols = [c for _, c in keep]
    values = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    ids = df[id_col].astype(str).to_numpy()
    labels = df["FLAG"].astype(int).to_numpy() if "FLAG" in df.columns else None
    return ids, labels, dates, values


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Long format (customer_id, date, consumption) -> SGCC-style wide frame."""
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["customer_id", "date", "consumption"]
    if not all(n in cols for n in need):
        raise ValueError("Long format needs columns: customer_id, date, consumption")
    d = df[[cols[n] for n in need]].copy()
    d.columns = need
    d["date"] = pd.to_datetime(d["date"], errors="coerce", format="mixed")
    d = d.dropna(subset=["date"])
    d["consumption"] = pd.to_numeric(d["consumption"], errors="coerce")
    wide = d.pivot_table(index="customer_id", columns="date",
                         values="consumption", aggfunc="mean",
                         dropna=False)  # keep customers whose readings are all missing
    wide.columns = [c.strftime("%Y-%m-%d") for c in wide.columns]
    return wide.reset_index().rename(columns={"customer_id": "CONS_NO"})


def read_any(df: pd.DataFrame):
    """Accept either long format or SGCC wide format."""
    lower = {c.lower().strip() for c in df.columns}
    if {"customer_id", "date", "consumption"} <= lower:
        df = long_to_wide(df)
    return split_wide(df)


# ---------------------------------------------------------------- cleaning
def clean(values: np.ndarray) -> np.ndarray:
    """Interpolate missing days, clip negatives, cap outliers (3-sigma rule)."""
    filled = (pd.DataFrame(values)
              .interpolate(axis=1, limit_direction="both")
              .fillna(0.0)
              .to_numpy(np.float32))
    filled = np.clip(filled, 0, None)
    mean = filled.mean(axis=1, keepdims=True)
    std = filled.std(axis=1, keepdims=True)
    return np.minimum(filled, mean + 3 * std)


def monthly_means(cleaned: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    months = dates.to_period("M").astype(str).to_numpy()
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    sums = np.add.reduceat(cleaned, starts, axis=1)
    counts = np.diff(np.r_[starts, len(months)])
    return sums / counts


def _longest_run(mask_row: np.ndarray) -> int:
    if not mask_row.any():
        return 0
    padded = np.r_[0, mask_row.astype(np.int8), 0]
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[::2]).max())


# ---------------------------------------------------------------- features
def compute_features(values_raw: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    missing_ratio = np.isnan(values_raw).mean(axis=1)
    x = clean(values_raw)
    n_days = x.shape[1]

    mean = x.mean(axis=1)
    std = x.std(axis=1)
    zero_mask = x <= EPS

    m = monthly_means(x, dates)
    n_months = m.shape[1]
    if n_months >= 2:
        t = np.arange(n_months) - (n_months - 1) / 2
        slope = (m * t).sum(axis=1) / (t ** 2).sum()
        trend_slope = slope / (mean + EPS)
        drops = (m[:, :-1] - m[:, 1:]) / (m[:, :-1] + EPS)
        max_drop = np.clip(drops.max(axis=1), 0, 1)
    else:
        trend_slope = np.zeros(len(x))
        max_drop = np.zeros(len(x))
    monthly_cv = m.std(axis=1) / (m.mean(axis=1) + EPS)

    recent = max(1, min(90, n_days // 3))
    past_mean = x[:, :-recent].mean(axis=1) if n_days > recent else mean
    recent_vs_past = x[:, -recent:].mean(axis=1) / (past_mean + EPS)

    weekend = dates.dayofweek.to_numpy() >= 5
    if weekend.any() and (~weekend).any():
        weekend_ratio = x[:, weekend].mean(axis=1) / (x[:, ~weekend].mean(axis=1) + EPS)
    else:
        weekend_ratio = np.ones(len(x))

    feats = pd.DataFrame({
        "mean_daily": mean,
        "std_daily": std,
        "cv_daily": std / (mean + EPS),
        "max_daily": x.max(axis=1),
        "missing_ratio": missing_ratio,
        "zero_ratio": zero_mask.mean(axis=1),
        "longest_zero_run": [_longest_run(r) for r in zero_mask],
        "low_day_ratio": (x < 0.2 * mean[:, None]).mean(axis=1),
        "spike_ratio": (x > (mean + 2 * std)[:, None]).mean(axis=1),
        "trend_slope": trend_slope,
        "max_monthly_drop": max_drop,
        "monthly_cv": monthly_cv,
        "recent_vs_past": np.clip(recent_vs_past, 0, 10),
        "weekend_ratio": np.clip(weekend_ratio, 0, 10),
    })
    return feats[FEATURE_NAMES].astype(np.float32)

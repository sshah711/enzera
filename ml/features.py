"""Shared data loading, cleaning and feature engineering for ENZERA.

Used by BOTH training (ml/train.py) and serving (backend/main.py), so the
deployed app always processes data exactly the way the model was trained.
"""
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

EPS = 1e-6


@contextmanager
def _quiet():
    """Customers with no readings make NaN-aware statistics warn ("mean of empty slice").
    That is expected: those values become 0 at the end of compute_features."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

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
    "active_span_ratio": "Share of the period the meter was reporting",
    "weekly_autocorr": "Weekly repetition of the usage pattern",
    "dow_strength": "Strength of the day-of-week cycle",
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


# ---------------------------------------------------------------- gap-aware cleaning
def clean_inside(values: np.ndarray) -> np.ndarray:
    """Like clean(), but only fills gaps BETWEEN readings.

    Days before a customer's first reading or after the last one stay NaN, so a meter that
    started in 2016 is not "filled" with a flat copy of its first value for 2014-2015
    (notebooks/03_feature_experiments.ipynb shows this improves PR-AUC from 0.332 to 0.376).
    """
    x = pd.DataFrame(values).interpolate(axis=1, limit_area="inside").to_numpy(np.float32)
    x = np.clip(x, 0, None)
    with _quiet():
        cap = np.nanmean(x, axis=1, keepdims=True) + 3 * np.nanstd(x, axis=1, keepdims=True)
    return np.where(np.isnan(x), np.nan, np.minimum(x, cap))


def nan_monthly_means(x: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Mean of the observed days in each calendar month (NaN if a month has no reading)."""
    months = dates.to_period("M").astype(str).to_numpy()
    cols = []
    with _quiet():
        for mth in pd.unique(months):
            cols.append(np.nanmean(x[:, months == mth], axis=1))
    return np.column_stack(cols)


# ---------------------------------------------------------------- features
def compute_features(values_raw: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """17 features per customer, computed only on the days each meter actually reported."""
    missing_ratio = np.isnan(values_raw).mean(axis=1)
    x = clean_inside(values_raw)
    obs = ~np.isnan(x)
    n_days = x.shape[1]
    n_obs = np.maximum(obs.sum(axis=1), 1)

    with _quiet():
        mean = np.nanmean(x, axis=1)
        std = np.nanstd(x, axis=1)

        # monthly level: trend (least-squares slope over observed months), drops, variability
        m = nan_monthly_means(x, dates)
        mv = ~np.isnan(m)
        t = np.arange(m.shape[1], dtype=float)
        tm = (mv * t).sum(axis=1) / np.maximum(mv.sum(axis=1), 1)
        mm = np.nanmean(m, axis=1)
        dt = np.where(mv, t - tm[:, None], 0.0)
        dm = np.where(mv, m - mm[:, None], 0.0)
        slope = (dt * dm).sum(axis=1) / np.maximum((dt ** 2).sum(axis=1), EPS)
        monthly_cv = np.nanstd(m, axis=1) / (mm + EPS)
        if m.shape[1] >= 2:
            drops = (m[:, :-1] - m[:, 1:]) / (m[:, :-1] + EPS)
            max_drop = np.clip(np.nan_to_num(np.nanmax(drops, axis=1), nan=0.0), 0, 1)
        else:
            max_drop = np.zeros(len(x))

        recent = max(1, min(90, n_days // 3))
        recent_vs_past = np.nanmean(x[:, -recent:], axis=1) / (np.nanmean(x[:, :-recent], axis=1) + EPS)

        dow = dates.dayofweek.to_numpy()
        weekend = dow >= 5
        weekend_ratio = np.nanmean(x[:, weekend], axis=1) / (np.nanmean(x[:, ~weekend], axis=1) + EPS)

        # weekly pattern: correlation with 7 days earlier, and share of variance explained by weekday
        d = x - mean[:, None]
        a, b = d[:, 7:], d[:, :-7]
        ok = ~np.isnan(a) & ~np.isnan(b)
        a0, b0 = np.where(ok, a, 0), np.where(ok, b, 0)
        weekly_autocorr = (a0 * b0).sum(axis=1) / np.sqrt((a0 ** 2).sum(axis=1) * (b0 ** 2).sum(axis=1))
        dow_means = np.column_stack([np.nanmean(x[:, dow == k], axis=1) for k in range(7)])
        dow_strength = np.nanvar(dow_means, axis=1) / np.nanvar(x, axis=1)

    zero = np.where(obs, x <= EPS, False)
    first = np.where(obs.any(axis=1), obs.argmax(axis=1), n_days)
    last = np.where(obs.any(axis=1), n_days - 1 - obs[:, ::-1].argmax(axis=1), -1)

    feats = pd.DataFrame({
        "mean_daily": mean,
        "std_daily": std,
        "cv_daily": std / (mean + EPS),
        "max_daily": np.max(np.where(obs, x, -np.inf), axis=1),
        "missing_ratio": missing_ratio,
        "zero_ratio": zero.sum(axis=1) / n_obs,
        "longest_zero_run": [_longest_run(r) for r in zero],
        "low_day_ratio": np.where(obs, x < 0.2 * mean[:, None], False).sum(axis=1) / n_obs,
        "spike_ratio": np.where(obs, x > (mean + 2 * std)[:, None], False).sum(axis=1) / n_obs,
        "trend_slope": slope / (mean + EPS),
        "max_monthly_drop": max_drop,
        "monthly_cv": monthly_cv,
        "recent_vs_past": np.clip(np.nan_to_num(recent_vs_past, nan=1.0), 0, 10),
        "weekend_ratio": np.clip(np.nan_to_num(weekend_ratio, nan=1.0), 0, 10),
        "active_span_ratio": np.clip(last - first + 1, 0, None) / n_days,
        "weekly_autocorr": np.nan_to_num(weekly_autocorr, nan=0.0),
        "dow_strength": np.clip(np.nan_to_num(dow_strength, nan=0.0), 0, 1),
    })
    # customers with no reading at all: every statistic is undefined -> 0
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats[FEATURE_NAMES].astype(np.float32)

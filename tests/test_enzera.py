"""Run with:  pytest -q   (API tests need a trained model in models/)"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_NAMES, clean, compute_features, long_to_wide, split_wide

ROOT = Path(__file__).resolve().parents[1]
HAS_MODEL = (ROOT / "models/enzera_model.joblib").exists() and (ROOT / "data/demo/demo_readings.csv.gz").exists()


def make_wide():
    dates = pd.date_range("2015-01-01", periods=120).strftime("%Y/%-m/%-d")
    rng = np.random.default_rng(0)
    vals = rng.gamma(2, 5, size=(3, 120))
    vals[1, 60:] = 0           # customer 2: usage drops to zero halfway
    vals[2, ::7] = np.nan      # customer 3: missing readings
    df = pd.DataFrame(vals, columns=list(dates))
    df = df[list(reversed(df.columns))]  # shuffled order, like SGCC
    df.insert(0, "FLAG", [0, 1, 0])
    df.insert(0, "CONS_NO", ["A", "B", "C"])
    return df


def test_split_wide_sorts_dates_and_reads_labels():
    ids, labels, dates, values = split_wide(make_wide())
    assert list(ids) == ["A", "B", "C"]
    assert list(labels) == [0, 1, 0]
    assert dates.is_monotonic_increasing
    assert values.shape == (3, 120)


def test_clean_removes_missing_and_negatives():
    x = np.array([[1, np.nan, 3, -5, 1000]], dtype=np.float32)
    c = clean(x)
    assert not np.isnan(c).any() and (c >= 0).all()


def test_features_shape_and_signals():
    _, _, dates, values = split_wide(make_wide())
    f = compute_features(values, dates)
    assert list(f.columns) == FEATURE_NAMES and len(f) == 3
    assert f.loc[1, "longest_zero_run"] >= 60
    assert f.loc[1, "recent_vs_past"] < f.loc[0, "recent_vs_past"]
    assert f.loc[2, "missing_ratio"] > 0


def test_long_format_matches_wide():
    long = pd.DataFrame({"customer_id": ["X"] * 3, "date": ["2015-01-01", "2015-01-02", "2015-01-03"],
                         "consumption": [1.0, 2.0, 3.0]})
    ids, labels, dates, values = split_wide(long_to_wide(long))
    assert list(ids) == ["X"] and labels is None and values.tolist() == [[1.0, 2.0, 3.0]]


def test_long_format_keeps_customer_with_all_missing_readings():
    long = pd.DataFrame({"customer_id": ["X", "X", "Y", "Y"],
                         "date": ["2015-01-01", "2015-01-02"] * 2,
                         "consumption": [1.0, 2.0, None, None]})
    ids, _, _, values = split_wide(long_to_wide(long))
    assert sorted(ids) == ["X", "Y"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    if not HAS_MODEL:
        pytest.skip("No trained model yet - run `python -m ml.train` first")
    import os
    os.environ["DB_PATH"] = str(tmp_path_factory.mktemp("db") / "test.db")
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["customers_scored"] > 0


def test_customer_list_sorted_with_reasons(client):
    rows = client.get("/customers", params={"limit": 20}).json()
    scores = [r["risk_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores)
    assert "reasons" in rows[0]


def test_predict_sample_upload(client):
    sample = ROOT / "data/samples/sample_upload.csv"
    with open(sample, "rb") as fh:
        r = client.post("/predict", files={"file": ("s.csv", fh, "text/csv")})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 50


def test_predict_rejects_bad_csv(client):
    r = client.post("/predict", files={"file": ("bad.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422


def test_case_lifecycle(client):
    cid = client.get("/customers", params={"limit": 1}).json()[0]["customer_id"]
    case = client.post("/cases", json={"customer_id": cid, "inspector": "Test"}).json()
    assert case["status"] == "open"
    upd = client.patch(f"/cases/{case['case_id']}", json={"status": "theft_confirmed"}).json()
    assert upd["status"] == "theft_confirmed"
    assert client.patch(f"/cases/{case['case_id']}", json={"status": "bogus"}).status_code == 422
    assert client.delete(f"/cases/{case['case_id']}").status_code == 204

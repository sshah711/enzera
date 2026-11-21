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


def test_gap_aware_features_ignore_days_before_first_reading():
    dates = pd.date_range("2015-01-01", periods=200)
    rng = np.random.default_rng(1)
    vals = rng.gamma(2, 5, size=(2, 200)).astype(np.float32)
    vals[1, :100] = np.nan                     # meter B only starts reporting on day 100
    f = compute_features(vals, dates)
    assert abs(f.loc[1, "active_span_ratio"] - 0.5) < 0.01 and f.loc[0, "active_span_ratio"] == 1.0
    assert f.loc[1, "longest_zero_run"] == 0   # the unreported period is not treated as zero usage
    seen = vals[1, 100:]
    expected = np.std(np.minimum(seen, seen.mean() + 3 * seen.std()))  # observed days only, same 3-sigma cap
    assert abs(f.loc[1, "std_daily"] - expected) < 1e-3


def test_weekly_pattern_features():
    dates = pd.date_range("2015-01-05", periods=140)               # starts on a Monday
    weekly = np.where(dates.dayofweek >= 5, 20.0, 5.0)             # strong weekend cycle
    noise = np.random.default_rng(2).gamma(2, 3, size=140)
    f = compute_features(np.vstack([weekly, noise]).astype(np.float32), dates)
    assert f.loc[0, "weekly_autocorr"] > 0.9 and f.loc[0, "dow_strength"] > 0.9
    assert f.loc[1, "weekly_autocorr"] < 0.5 and f.loc[1, "dow_strength"] < 0.5


def test_inspection_sheet_pdf():
    import sys
    sys.path.insert(0, str(ROOT / "frontend"))
    import sheet
    c = {"customer_id": "ABC", "risk_score": 0.9, "theft_probability": 0.5, "anomaly_score": 0.3, "risk_band": "High",
         "explanation": {"contributions": [{"label": "Day-to-day variation (kWh)", "value": "12.0", "impact": 0.8}]},
         "dates": ["2015-01-01", "2015-02-01", "2015-03-01"], "consumption": [1.0, None, 3.0], "cases": []}
    pdf = sheet.inspection_sheet(c, 0.65, "test")
    assert pdf[:4] == b"%PDF" and len(pdf) > 1000


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
        r = client.post("/predict", files={"file": ("s.csv", fh, "text/csv")}, headers=login(client, "manager", "manager123"))
    assert r.status_code == 200
    assert len(r.json()["results"]) == 50


def test_predict_rejects_bad_csv(client):
    r = client.post("/predict", files={"file": ("bad.csv", b"a,b\n1,2\n", "text/csv")},
                    headers=login(client, "manager", "manager123"))
    assert r.status_code == 422


def login(client, user, pw):
    r = client.post("/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_login_rejects_wrong_password(client):
    assert client.post("/auth/login", json={"username": "manager", "password": "nope"}).status_code == 401


def test_case_lifecycle(client):
    mgr = login(client, "manager", "manager123")
    cid = client.get("/customers", params={"limit": 1}).json()[0]["customer_id"]
    case = client.post("/cases", json={"customer_id": cid, "inspector": "inspector"}, headers=mgr).json()
    assert case["status"] == "open"
    upd = client.patch(f"/cases/{case['case_id']}", json={"status": "theft_confirmed"}, headers=mgr).json()
    assert upd["status"] == "theft_confirmed"
    assert client.patch(f"/cases/{case['case_id']}", json={"status": "bogus"}, headers=mgr).status_code == 422
    assert client.delete(f"/cases/{case['case_id']}", headers=mgr).status_code == 204
    history = client.get(f"/cases/{case['case_id']}/history", headers=mgr).json()
    assert [e["action"] for e in history] == ["created", "status_changed", "deleted"]
    assert history[1]["from_status"] == "open" and history[1]["to_status"] == "theft_confirmed"


def test_roles_are_enforced(client):
    cid = client.get("/customers", params={"limit": 1}).json()[0]["customer_id"]
    assert client.post("/cases", json={"customer_id": cid}).status_code == 401          # not logged in
    insp = login(client, "inspector", "inspector123")
    case = client.post("/cases", json={"customer_id": cid}, headers=insp).json()
    assert case["inspector"] == "inspector"
    assert client.patch(f"/cases/{case['case_id']}", json={"notes": "visited"}, headers=insp).status_code == 200
    assert client.patch(f"/cases/{case['case_id']}", json={"status": "normal"}, headers=insp).status_code == 403
    assert client.delete(f"/cases/{case['case_id']}", headers=insp).status_code == 403
    assert client.get("/labels", headers=insp).status_code == 403
    assert client.post("/cases", json={"customer_id": "NOT-A-CUSTOMER"}, headers=insp).status_code == 404


def test_labels_export_for_retraining(client):
    mgr = login(client, "manager", "manager123")
    cid = client.get("/customers", params={"limit": 1, "band": "Low"}).json()[0]["customer_id"]
    case = client.post("/cases", json={"customer_id": cid, "inspector": "inspector2"}, headers=mgr).json()
    client.patch(f"/cases/{case['case_id']}", json={"status": "normal"}, headers=mgr)
    labels = {x["customer_id"]: x["label"] for x in client.get("/labels", headers=mgr).json()}
    assert labels[cid] == 0


def test_explanation_adds_up_to_score(client):
    c = client.get("/customers", params={"limit": 1}).json()[0]
    d = client.get(f"/customers/{c['customer_id']}").json()
    e = d["explanation"]
    assert len(e["contributions"]) == len(FEATURE_NAMES)
    log_odds = e["base_value"] + sum(x["impact"] for x in e["contributions"])
    assert abs(1 / (1 + np.exp(-log_odds)) - d["risk_score"]) < 1e-3   # SHAP additivity
    assert 0 <= d["theft_probability"] <= 1


def test_what_if_changes_score_only_when_features_change(client):
    cid = client.get("/customers", params={"limit": 1}).json()[0]["customer_id"]
    same = client.post("/whatif", json={"customer_id": cid, "changes": {}}).json()
    assert same["original"] == same["what_if"]
    changed = client.post("/whatif", json={"customer_id": cid, "changes": {"std_daily": 0.0, "max_daily": 1.0}}).json()
    assert changed["what_if"]["risk_score"] != changed["original"]["risk_score"]
    assert client.post("/whatif", json={"customer_id": cid, "changes": {"bogus": 1}}).status_code == 422


def test_similar_customers(client):
    cid = client.get("/customers", params={"limit": 1}).json()[0]["customer_id"]
    r = client.get(f"/customers/{cid}/similar", params={"k": 3}).json()
    assert len(r["similar"]) == 3 and cid not in [s["customer_id"] for s in r["similar"]]
    d = [s["distance"] for s in r["similar"]]
    assert d == sorted(d)


def test_drift_report(client):
    mgr = login(client, "manager", "manager123")
    rows = client.get("/drift", headers=mgr).json()["features"]
    assert len(rows) == len(FEATURE_NAMES) and all(r["psi"] >= 0 for r in rows)
    sample = ROOT / "data/samples/sample_upload.csv"
    with open(sample, "rb") as fh:
        res = client.post("/predict", files={"file": ("s.csv", fh, "text/csv")}, headers=mgr).json()
    assert len(res["drift"]) == len(FEATURE_NAMES)


def test_inspector_sees_only_own_cases_and_no_manager_features(client):
    mgr, insp, insp2 = (login(client, u, p) for u, p in
                        (("manager", "manager123"), ("inspector", "inspector123"), ("inspector2", "inspector456")))
    cid = client.get("/customers", params={"limit": 3}).json()[2]["customer_id"]
    mine = client.post("/cases", json={"customer_id": cid, "inspector": "inspector"}, headers=mgr).json()
    other = client.post("/cases", json={"customer_id": cid, "inspector": "inspector2"}, headers=mgr).json()
    assert mine["inspector_name"] == "Rahul Patel"
    seen = {c["case_id"] for c in client.get("/cases", headers=insp).json()}
    assert mine["case_id"] in seen and other["case_id"] not in seen
    assert {mine["case_id"], other["case_id"]} <= {c["case_id"] for c in client.get("/cases", headers=mgr).json()}
    assert client.patch(f"/cases/{other['case_id']}", json={"notes": "x"}, headers=insp).status_code == 403
    assert client.get(f"/cases/{other['case_id']}/history", headers=insp).status_code == 403
    assert client.get("/cases").status_code == 401                                   # not signed in
    for path in ("/drift", "/metrics", "/staff"):                                    # manager-only
        assert client.get(path, headers=insp).status_code == 403
        assert client.get(path, headers=mgr).status_code == 200
    bad = client.post("/predict", files={"file": ("s.csv", b"a,b\n1,2\n", "text/csv")}, headers=insp)
    assert bad.status_code == 403
    assert client.post("/cases", json={"customer_id": cid, "inspector": "nobody"}, headers=mgr).status_code == 422
    detail = client.get(f"/customers/{cid}", headers=insp2).json()
    assert {c["inspector"] for c in detail["cases"]} == {"inspector2"}

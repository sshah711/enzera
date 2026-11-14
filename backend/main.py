"""ENZERA backend: REST API over the theft model, customers and inspection cases.

Run locally:  uvicorn backend.main:app --reload
API docs:     http://localhost:8000/docs
"""
import io
import json
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.features import compute_features, read_any, split_wide  # noqa: E402
from ml.scoring import load_bundle, score_features  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models/enzera_model.joblib"))
DEMO_PATH = Path(os.getenv("DEMO_PATH", ROOT / "data/demo/demo_readings.csv.gz"))
METRICS_PATH = ROOT / "reports/metrics.json"
DB_PATH = os.getenv("DB_PATH", str(ROOT / "backend/enzera.db"))
API_KEY = os.getenv("API_KEY")  # optional: set it to require an X-API-Key header
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_CUSTOMERS = 2000

STATE = {}


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS risk_scores (
            customer_id TEXT PRIMARY KEY,
            risk_score REAL, anomaly_score REAL, risk_band TEXT,
            reasons TEXT, model_version TEXT, scored_at TEXT);
        CREATE TABLE IF NOT EXISTS inspection_cases (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            inspector TEXT, notes TEXT,
            created_at TEXT, updated_at TEXT);
        """)


def seed_scores():
    """Score the unseen demo customers with the current model at startup."""
    ids, _, dates, values = STATE["demo"]
    feats = compute_features(values, dates)
    scored = score_features(STATE["bundle"], feats, ids)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db() as con:
        con.execute("DELETE FROM risk_scores")
        con.executemany(
            "INSERT INTO risk_scores VALUES (?,?,?,?,?,?,?)",
            [(r.customer_id, r.risk_score, r.anomaly_score, r.risk_band,
              json.dumps(r.reasons), STATE["bundle"]["version"], now)
             for r in scored.itertuples()])


@asynccontextmanager
async def lifespan(app):
    STATE["bundle"] = load_bundle(MODEL_PATH)
    STATE["demo"] = split_wide(pd.read_csv(DEMO_PATH))
    STATE["demo_index"] = {cid: i for i, cid in enumerate(STATE["demo"][0])}
    init_db()
    seed_scores()
    yield


app = FastAPI(title="ENZERA API", version="1.0",
              description="Explainable electricity theft and abnormal consumption detection",
              lifespan=lifespan)


def check_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")


def row_to_dict(r):
    d = dict(r)
    if "reasons" in d:
        d["reasons"] = json.loads(d["reasons"])
    return d


@app.get("/health")
def health():
    with db() as con:
        n = con.execute("SELECT COUNT(*) FROM risk_scores").fetchone()[0]
    return {"status": "ok", "model_version": STATE["bundle"]["version"], "customers_scored": n}


@app.get("/stats", dependencies=[Depends(check_key)])
def stats():
    with db() as con:
        bands = dict(con.execute("SELECT risk_band, COUNT(*) FROM risk_scores GROUP BY risk_band").fetchall())
        cases = dict(con.execute("SELECT status, COUNT(*) FROM inspection_cases GROUP BY status").fetchall())
    return {"bands": bands, "cases": cases, "threshold": STATE["bundle"]["threshold"]}


@app.get("/customers", dependencies=[Depends(check_key)])
def list_customers(band: Optional[Literal["High", "Medium", "Low"]] = None,
                   limit: int = 100, offset: int = 0):
    limit = max(1, min(limit, 1000))
    q, args = "SELECT * FROM risk_scores", []
    if band:
        q += " WHERE risk_band = ?"; args.append(band)
    q += " ORDER BY risk_score DESC LIMIT ? OFFSET ?"; args += [limit, offset]
    with db() as con:
        return [row_to_dict(r) for r in con.execute(q, args).fetchall()]


@app.get("/customers/{customer_id}", dependencies=[Depends(check_key)])
def get_customer(customer_id: str):
    with db() as con:
        r = con.execute("SELECT * FROM risk_scores WHERE customer_id = ?", (customer_id,)).fetchone()
        cases = con.execute("SELECT * FROM inspection_cases WHERE customer_id = ? ORDER BY case_id DESC",
                            (customer_id,)).fetchall()
    if r is None:
        raise HTTPException(404, "Customer not found")
    _, _, dates, values = STATE["demo"]
    series = values[STATE["demo_index"][customer_id]]
    out = row_to_dict(r)
    out["dates"] = dates.strftime("%Y-%m-%d").tolist()
    out["consumption"] = [None if np.isnan(v) else round(float(v), 3) for v in series]
    out["cases"] = [dict(c) for c in cases]
    return out


@app.post("/predict", dependencies=[Depends(check_key)])
async def predict(file: UploadFile = File(...)):
    """Score a CSV: long format (customer_id, date, consumption) or SGCC wide format."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")
    try:
        ids, _, dates, values = read_any(pd.read_csv(io.BytesIO(raw)))
    except Exception as e:
        raise HTTPException(422, f"Could not read CSV: {e}")
    if len(ids) > MAX_UPLOAD_CUSTOMERS:
        raise HTTPException(413, f"Too many customers (max {MAX_UPLOAD_CUSTOMERS})")
    if len(dates) < 30:
        raise HTTPException(422, "Need at least 30 days of readings per customer")
    scored = score_features(STATE["bundle"], compute_features(values, dates), ids)
    return {"model_version": STATE["bundle"]["version"], "days": len(dates),
            "results": scored.sort_values("risk_score", ascending=False).to_dict("records")}


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "metrics.json not found - run training first")
    return json.loads(METRICS_PATH.read_text())


# ------------------------------------------------------------ inspection cases
class CaseCreate(BaseModel):
    customer_id: str
    inspector: Optional[str] = None
    notes: Optional[str] = None


class CaseUpdate(BaseModel):
    status: Optional[Literal["open", "theft_confirmed", "normal"]] = None
    notes: Optional[str] = None


@app.get("/cases", dependencies=[Depends(check_key)])
def list_cases(status: Optional[str] = None):
    q, args = "SELECT * FROM inspection_cases", []
    if status:
        q += " WHERE status = ?"; args.append(status)
    with db() as con:
        return [dict(r) for r in con.execute(q + " ORDER BY case_id DESC", args).fetchall()]


@app.post("/cases", status_code=201, dependencies=[Depends(check_key)])
def create_case(body: CaseCreate):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db() as con:
        cur = con.execute(
            "INSERT INTO inspection_cases (customer_id, inspector, notes, created_at, updated_at) VALUES (?,?,?,?,?)",
            (body.customer_id, body.inspector, body.notes, now, now))
        return dict(con.execute("SELECT * FROM inspection_cases WHERE case_id = ?", (cur.lastrowid,)).fetchone())


@app.patch("/cases/{case_id}", dependencies=[Depends(check_key)])
def update_case(case_id: int, body: CaseUpdate):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db() as con:
        r = con.execute("SELECT * FROM inspection_cases WHERE case_id = ?", (case_id,)).fetchone()
        if r is None:
            raise HTTPException(404, "Case not found")
        con.execute("UPDATE inspection_cases SET status = ?, notes = ?, updated_at = ? WHERE case_id = ?",
                    (body.status or r["status"], body.notes if body.notes is not None else r["notes"], now, case_id))
        return dict(con.execute("SELECT * FROM inspection_cases WHERE case_id = ?", (case_id,)).fetchone())


@app.delete("/cases/{case_id}", status_code=204, dependencies=[Depends(check_key)])
def delete_case(case_id: int):
    with db() as con:
        if con.execute("DELETE FROM inspection_cases WHERE case_id = ?", (case_id,)).rowcount == 0:
            raise HTTPException(404, "Case not found")

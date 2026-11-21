"""ENZERA backend: REST API over the theft model, customers, inspection cases and users.

Run locally:  uvicorn backend.main:app --reload
API docs:     http://localhost:8000/docs

Database: SQLite by default (file backend/enzera.db, or DB_PATH). Set DATABASE_URL to a PostgreSQL URL
(e.g. a free Neon or Render Postgres database) to keep cases when the free Render instance restarts.
The same SQLAlchemy code runs on both.
"""
import base64
import hashlib
import hmac
import io
import json
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import (Column, Float, Integer, MetaData, String, Table, Text, create_engine, delete, func,
                        insert, select, update)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.features import FEATURE_INFO, FEATURE_NAMES, compute_features, nan_monthly_means, read_any, split_wide  # noqa: E402
from ml.scoring import band, calibrated, drift_report, explain_one, load_bundle, score_features  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models/enzera_model.joblib"))
DEMO_PATH = Path(os.getenv("DEMO_PATH", ROOT / "data/demo/demo_readings.csv.gz"))
METRICS_PATH = ROOT / "reports/metrics.json"
API_KEY = os.getenv("API_KEY")  # optional: set it to require an X-API-Key header on data endpoints
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_CUSTOMERS = 2000

STATE = {}


# ------------------------------------------------------------ database (SQLite or PostgreSQL)
def database_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        return f"sqlite:///{os.getenv('DB_PATH', str(ROOT / 'backend/enzera.db'))}"
    # Render/Neon give "postgres://..." or "postgresql://...". SQLAlchemy 2.1 would pick the psycopg (v3)
    # driver for "postgresql://", but we install psycopg2, so name the driver explicitly.
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            url = "postgresql+psycopg2://" + url[len(prefix):]
    return url


meta = MetaData()
risk_scores = Table(
    "risk_scores", meta,
    Column("customer_id", String(64), primary_key=True),
    Column("risk_score", Float), Column("theft_probability", Float), Column("anomaly_score", Float),
    Column("risk_band", String(10)), Column("reasons", Text),
    Column("model_version", String(40)), Column("scored_at", String(40)))
cases = Table(
    "inspection_cases", meta,
    Column("case_id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", String(64), nullable=False),
    Column("status", String(20), nullable=False, default="open"),
    Column("inspector", String(100)), Column("notes", Text),
    Column("created_at", String(40)), Column("updated_at", String(40)))
case_events = Table(  # audit trail: one row per action on a case
    "case_events", meta,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("case_id", Integer, nullable=False),
    Column("at", String(40)), Column("username", String(100)), Column("role", String(20)),
    Column("action", String(30)), Column("from_status", String(20)), Column("to_status", String(20)),
    Column("notes", Text))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db():
    engine = create_engine(database_url(), pool_pre_ping=True)
    risk_scores.drop(engine, checkfirst=True)  # scores are recomputed at every start
    meta.create_all(engine)
    STATE["engine"] = engine


def db():
    return STATE["engine"].begin()  # connection + transaction, committed on exit


def log_event(con, case_id, user, action, from_status=None, to_status=None, notes=None):
    con.execute(insert(case_events).values(
        case_id=case_id, at=now(), username=user["u"], role=user["r"], action=action,
        from_status=from_status, to_status=to_status, notes=notes))


def seed_scores():
    """Score the unseen demo customers with the current model at startup."""
    ids, _, dates, values = STATE["demo"]
    feats = compute_features(values, dates)
    feats.index = ids
    STATE["demo_feats"] = feats
    sim = np.sign(feats.to_numpy()) * np.log1p(np.abs(feats.to_numpy()))
    STATE["demo_z"] = (sim - STATE["bundle"]["sim_mean"]) / STATE["bundle"]["sim_std"]
    scored = score_features(STATE["bundle"], feats, ids)
    stamp = now()
    with db() as con:
        con.execute(insert(risk_scores), [
            {"customer_id": r.customer_id, "risk_score": r.risk_score, "theft_probability": r.theft_probability,
             "anomaly_score": r.anomaly_score, "risk_band": r.risk_band, "reasons": json.dumps(r.reasons),
             "model_version": STATE["bundle"]["version"], "scored_at": stamp}
            for r in scored.itertuples()])


# ------------------------------------------------------------ users and roles
# Demo accounts unless USERS is set as "name:password:role,name:password:role" (roles: inspector, manager).
DEFAULT_USERS = ("inspector:inspector123:inspector,inspector2:inspector456:inspector,"
                 "manager:manager123:manager")
# Dummy staff details shown after sign-in (demo data, not real people).
PROFILES = {
    "inspector": {"full_name": "Rahul Patel", "title": "Field Inspector", "employee_id": "INS-1021",
                  "zone": "Ahmedabad East", "phone": "+91 90000 11021", "email": "rahul.patel@enzera.demo"},
    "inspector2": {"full_name": "Priya Shah", "title": "Field Inspector", "employee_id": "INS-1034",
                   "zone": "Gandhinagar", "phone": "+91 90000 11034", "email": "priya.shah@enzera.demo"},
    "manager": {"full_name": "Anil Mehta", "title": "Revenue Protection Manager", "employee_id": "MGR-2007",
                "zone": "All zones", "phone": "+91 90000 22007", "email": "anil.mehta@enzera.demo"},
}
ROLE_LEVEL = {"inspector": 1, "manager": 2}
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)  # set it in production so logins survive restarts
TOKEN_HOURS = 12


def _hash(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users(spec):
    users = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, pw, role = item.split(":")
        if role not in ROLE_LEVEL:
            raise ValueError(f"Unknown role {role!r} for user {name!r}")
        users[name] = {"hash": _hash(pw), "role": role}
    return users


USERS = load_users(os.getenv("USERS", DEFAULT_USERS))


def _sign(payload: str) -> str:
    return hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_token(username, role):
    payload = base64.urlsafe_b64encode(json.dumps(
        {"u": username, "r": role, "exp": int(time.time()) + TOKEN_HOURS * 3600}).encode()).decode()
    return f"{payload}.{_sign(payload)}"


def read_token(token):
    try:
        payload, sig = token.rsplit(".", 1)
        if not hmac.compare_digest(sig, _sign(payload)):
            return None
        data = json.loads(base64.urlsafe_b64decode(payload))
        return data if data["exp"] > time.time() else None
    except Exception:
        return None


def current_user(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return read_token(authorization[len("Bearer "):])


def require(role):
    def dep(user=Depends(current_user)):
        if user is None:
            raise HTTPException(401, "Please log in (session missing or expired)")
        if ROLE_LEVEL[user["r"]] < ROLE_LEVEL[role]:
            raise HTTPException(403, f"This action needs the '{role}' role")
        return user
    return dep


# ------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app):
    STATE["bundle"] = load_bundle(MODEL_PATH)
    STATE["demo"] = split_wide(pd.read_csv(DEMO_PATH))
    STATE["demo_index"] = {cid: i for i, cid in enumerate(STATE["demo"][0])}
    init_db()
    seed_scores()
    yield
    STATE["engine"].dispose()


app = FastAPI(title="ENZERA API", version="2.0",
              description="Explainable electricity theft and abnormal consumption detection",
              lifespan=lifespan)


def check_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")


def row_to_dict(r):
    d = dict(r._mapping)
    if "reasons" in d:
        d["reasons"] = json.loads(d["reasons"])
    return d


def known_customer(customer_id):
    if customer_id not in STATE["demo_index"]:
        raise HTTPException(404, "Customer not found")
    return customer_id


@app.get("/health")
def health():
    with db() as con:
        n = con.execute(select(func.count()).select_from(risk_scores)).scalar()
    return {"status": "ok", "model_version": STATE["bundle"]["version"], "customers_scored": n,
            "database": STATE["engine"].dialect.name}


# ------------------------------------------------------------ auth
class Login(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def login(body: Login):
    u = USERS.get(body.username)
    if u is None or not hmac.compare_digest(u["hash"], _hash(body.password)):
        raise HTTPException(401, "Wrong username or password")
    return {"token": make_token(body.username, u["role"]), "username": body.username, "role": u["role"],
            "profile": PROFILES.get(body.username, {}), "expires_in_hours": TOKEN_HOURS}


@app.get("/auth/me")
def me(user=Depends(require("inspector"))):
    return {"username": user["u"], "role": user["r"], "profile": PROFILES.get(user["u"], {})}


# ------------------------------------------------------------ customers and scoring
@app.get("/stats", dependencies=[Depends(check_key)])
def stats(user=Depends(current_user)):
    q = select(cases.c.status, func.count())
    if user and user["r"] == "inspector":
        q = q.where(cases.c.inspector == user["u"])  # an inspector only counts their own cases
    with db() as con:
        bands = dict(con.execute(select(risk_scores.c.risk_band, func.count()).group_by(risk_scores.c.risk_band)).all())
        by_status = dict(con.execute(q.group_by(cases.c.status)).all())
    return {"bands": bands, "cases": by_status, "threshold": STATE["bundle"]["threshold"],
            "model_version": STATE["bundle"]["version"]}


@app.get("/customers", dependencies=[Depends(check_key)])
def list_customers(band: Optional[Literal["High", "Medium", "Low"]] = None, limit: int = 100, offset: int = 0):
    limit = max(1, min(limit, 1000))
    q = select(risk_scores)
    if band:
        q = q.where(risk_scores.c.risk_band == band)
    q = q.order_by(risk_scores.c.risk_score.desc()).limit(limit).offset(max(0, offset))
    with db() as con:
        return [row_to_dict(r) for r in con.execute(q).all()]


@app.get("/customers/{customer_id}", dependencies=[Depends(check_key)])
def get_customer(customer_id: str, user=Depends(current_user)):
    known_customer(customer_id)
    q = visible_cases(user).where(cases.c.customer_id == customer_id).order_by(cases.c.case_id.desc())
    with db() as con:
        r = con.execute(select(risk_scores).where(risk_scores.c.customer_id == customer_id)).first()
        cs = con.execute(q).all() if user else []
    _, _, dates, values = STATE["demo"]
    series = values[STATE["demo_index"][customer_id]]
    out = row_to_dict(r)
    out["dates"] = dates.strftime("%Y-%m-%d").tolist()
    out["consumption"] = [None if np.isnan(v) else round(float(v), 3) for v in series]
    out["cases"] = [case_dict(c) for c in cs]
    out["explanation"] = explain_one(STATE["bundle"], STATE["demo_feats"].loc[[customer_id]])
    return out


class WhatIf(BaseModel):
    customer_id: str
    changes: Dict[str, float] = {}


@app.post("/whatif", dependencies=[Depends(check_key)])
def what_if(body: WhatIf):
    """Re-score a customer after changing some of its feature values (the model is not changed)."""
    known_customer(body.customer_id)
    unknown = set(body.changes) - set(FEATURE_NAMES)
    if unknown:
        raise HTTPException(422, f"Unknown features: {sorted(unknown)}")
    b = STATE["bundle"]
    original = STATE["demo_feats"].loc[[body.customer_id]]
    edited = original.copy()
    for f, v in body.changes.items():
        edited[f] = np.float32(v)

    def score(feats):
        risk = float(b["model"].predict_proba(feats[FEATURE_NAMES].to_numpy())[:, 1][0])
        return {"risk_score": round(risk, 4), "theft_probability": round(float(calibrated(b, np.array([risk]))[0]), 4),
                "risk_band": band(risk, b["threshold"])}

    return {"original": score(original), "what_if": score(edited),
            "features": {f: float(edited[f].iloc[0]) for f in FEATURE_NAMES},
            "explanation": explain_one(b, edited)}


@app.get("/customers/{customer_id}/similar", dependencies=[Depends(check_key)])
def similar_customers(customer_id: str, k: int = 5):
    """Nearest demo customers in (log-scaled, standardised) feature space."""
    known_customer(customer_id)
    k = max(1, min(k, 20))
    ids, _, dates, values = STATE["demo"]
    z = STATE["demo_z"]
    i = STATE["demo_index"][customer_id]
    dist = np.sqrt(((z - z[i]) ** 2).sum(axis=1))
    dist[i] = np.inf
    nearest = np.argsort(dist)[:k]
    with db() as con:
        rows = {r.customer_id: r for r in con.execute(
            select(risk_scores).where(risk_scores.c.customer_id.in_([ids[j] for j in nearest]))).all()}
    monthly = nan_monthly_means(values[np.r_[i, nearest]], dates)
    months = dates.to_period("M").unique().astype(str).tolist()
    clean = lambda row: [None if np.isnan(v) else round(float(v), 3) for v in row]
    return {"months": months, "customer_monthly": clean(monthly[0]),
            "similar": [{"customer_id": ids[j], "distance": round(float(dist[j]), 3),
                         "risk_score": rows[ids[j]].risk_score, "risk_band": rows[ids[j]].risk_band,
                         "monthly": clean(monthly[n + 1])} for n, j in enumerate(nearest)]}


@app.post("/predict", dependencies=[Depends(check_key), Depends(require("manager"))])
async def predict(file: UploadFile = File(...)):
    """Score a CSV: long format (customer_id, date, consumption) or SGCC wide format.
    Also returns a drift report: how different the uploaded customers are from the training data."""
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
    feats = compute_features(values, dates)
    scored = score_features(STATE["bundle"], feats, ids)
    return {"model_version": STATE["bundle"]["version"], "days": len(dates),
            "results": scored.sort_values("risk_score", ascending=False).to_dict("records"),
            "drift": drift_report(STATE["bundle"], feats)}


@app.get("/drift", dependencies=[Depends(check_key), Depends(require("manager"))])
def drift_demo():
    """Drift of the demo customers vs the training data (expected: stable, same source)."""
    return {"population": "demo customers", "customers": len(STATE["demo_feats"]),
            "features": drift_report(STATE["bundle"], STATE["demo_feats"])}


@app.get("/features")
def features():
    return [{"feature": f, "label": FEATURE_INFO[f]} for f in FEATURE_NAMES]


@app.get("/metrics", dependencies=[Depends(require("manager"))])
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


def case_dict(r):
    d = dict(r._mapping)
    d["inspector_name"] = PROFILES.get(d["inspector"] or "", {}).get("full_name", d["inspector"])
    return d


def get_case(con, case_id):
    r = con.execute(select(cases).where(cases.c.case_id == case_id)).first()
    if r is None:
        raise HTTPException(404, "Case not found")
    return case_dict(r)


def visible_cases(user):
    """Managers see every case; an inspector sees only the cases assigned to them."""
    q = select(cases)
    if user and user["r"] == "inspector":
        q = q.where(cases.c.inspector == user["u"])
    return q


def check_own_case(case, user):
    if user["r"] == "inspector" and case["inspector"] != user["u"]:
        raise HTTPException(403, "This case is assigned to another inspector")


INSPECTORS = sorted(u for u, v in USERS.items() if v["role"] == "inspector")


@app.get("/staff", dependencies=[Depends(check_key)])
def staff(user=Depends(require("manager"))):
    """Inspectors a manager can assign cases to."""
    return [{"username": u, "full_name": PROFILES.get(u, {}).get("full_name", u)} for u in INSPECTORS]


@app.get("/cases", dependencies=[Depends(check_key)])
def list_cases(status: Optional[str] = None, user=Depends(require("inspector"))):
    q = visible_cases(user)
    if status:
        q = q.where(cases.c.status == status)
    with db() as con:
        return [case_dict(r) for r in con.execute(q.order_by(cases.c.case_id.desc())).all()]


@app.post("/cases", status_code=201, dependencies=[Depends(check_key)])
def create_case(body: CaseCreate, user=Depends(require("inspector"))):
    known_customer(body.customer_id)
    if user["r"] == "inspector":
        assignee = user["u"]  # an inspector can only open cases for themselves
    else:
        assignee = body.inspector
        if assignee not in INSPECTORS:
            raise HTTPException(422, f"Assign the case to an inspector: {', '.join(INSPECTORS)}")
    stamp = now()
    with db() as con:
        case_id = con.execute(insert(cases).values(
            customer_id=body.customer_id, status="open", inspector=assignee,
            notes=body.notes, created_at=stamp, updated_at=stamp)).inserted_primary_key[0]
        log_event(con, case_id, user, "created", None, "open", body.notes)
        return get_case(con, case_id)


@app.patch("/cases/{case_id}", dependencies=[Depends(check_key)])
def update_case(case_id: int, body: CaseUpdate, user=Depends(require("inspector"))):
    with db() as con:
        old = get_case(con, case_id)
        check_own_case(old, user)
        changes_status = body.status is not None and body.status != old["status"]
        if changes_status and ROLE_LEVEL[user["r"]] < ROLE_LEVEL["manager"]:
            raise HTTPException(403, "Only a manager can record an outcome or reopen a case")
        new_status = body.status or old["status"]
        new_notes = body.notes if body.notes is not None else old["notes"]
        con.execute(update(cases).where(cases.c.case_id == case_id).values(
            status=new_status, notes=new_notes, updated_at=now()))
        if changes_status:
            log_event(con, case_id, user, "status_changed", old["status"], new_status, body.notes)
        elif body.notes is not None:
            log_event(con, case_id, user, "note_updated", old["status"], new_status, body.notes)
        return get_case(con, case_id)


@app.delete("/cases/{case_id}", status_code=204, dependencies=[Depends(check_key)])
def delete_case(case_id: int, user=Depends(require("manager"))):
    with db() as con:
        old = get_case(con, case_id)
        con.execute(delete(cases).where(cases.c.case_id == case_id))
        log_event(con, case_id, user, "deleted", old["status"], None, None)  # history is kept


@app.get("/cases/{case_id}/history", dependencies=[Depends(check_key)])
def case_history(case_id: int, user=Depends(require("inspector"))):
    with db() as con:
        if user["r"] == "inspector":
            check_own_case(get_case(con, case_id), user)
        rows = con.execute(select(case_events).where(case_events.c.case_id == case_id)
                           .order_by(case_events.c.event_id)).all()
    if not rows:
        raise HTTPException(404, "No history for this case")
    return [dict(r._mapping) for r in rows]


@app.get("/labels", dependencies=[Depends(check_key)])
def training_labels(user=Depends(require("manager"))):
    """Confirmed inspection outcomes as new training labels (1 = theft_confirmed, 0 = normal).
    If a customer has several decided cases, the most recently updated one counts. Used by ml/retrain.py."""
    with db() as con:
        rows = con.execute(select(cases).where(cases.c.status.in_(["theft_confirmed", "normal"]))
                           .order_by(cases.c.updated_at)).all()
    latest = {r.customer_id: r for r in rows}
    return [{"customer_id": cid, "label": int(r.status == "theft_confirmed"), "case_id": r.case_id,
             "decided_at": r.updated_at} for cid, r in latest.items()]

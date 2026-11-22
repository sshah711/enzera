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
uploads = Table(  # customers added through "Upload and score": raw readings, re-scored at every start
    "uploaded_customers", meta,
    Column("customer_id", String(64), primary_key=True),
    Column("dates", Text), Column("consumption", Text),
    Column("uploaded_at", String(40)), Column("uploaded_by", String(100)))
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


def customer_data(customer_id):
    """(dates, daily values, 17-feature row) for a demo customer or an uploaded one."""
    if customer_id in STATE["extra"]:
        e = STATE["extra"][customer_id]
        return e["dates"], e["values"], e["feats"]
    _, _, dates, values = STATE["demo"]
    return dates, values[STATE["demo_index"][customer_id]], STATE["demo_feats"].loc[[customer_id]]


def add_uploaded(ids, dates, values, user=None, stamp=None, save=True):
    """Score uploaded customers, keep them in memory, write their scores (and raw readings) to the database.
    Customers that already exist among the demo customers are skipped. Returns the IDs added."""
    new = [i for i, cid in enumerate(ids) if str(cid) not in STATE["demo_index"]]
    if not new:
        return []
    feats = compute_features(values[new], dates)
    new_ids = [str(ids[i]) for i in new]
    feats.index = new_ids
    scored = score_features(STATE["bundle"], feats, new_ids)
    stamp = stamp or now()
    day = dates.strftime("%Y-%m-%d").tolist()
    with db() as con:
        con.execute(delete(risk_scores).where(risk_scores.c.customer_id.in_(new_ids)))
        con.execute(insert(risk_scores), [
            {"customer_id": r.customer_id, "risk_score": r.risk_score, "theft_probability": r.theft_probability,
             "anomaly_score": r.anomaly_score, "risk_band": r.risk_band, "reasons": json.dumps(r.reasons),
             "model_version": STATE["bundle"]["version"], "scored_at": stamp} for r in scored.itertuples()])
        if save:
            con.execute(delete(uploads).where(uploads.c.customer_id.in_(new_ids)))
            con.execute(insert(uploads), [
                {"customer_id": cid, "dates": json.dumps(day),
                 "consumption": json.dumps([None if np.isnan(v) else round(float(v), 3) for v in values[i]]),
                 "uploaded_at": stamp, "uploaded_by": user} for cid, i in zip(new_ids, new)])
    for cid, i in zip(new_ids, new):
        STATE["extra"][cid] = {"dates": dates, "values": values[i], "feats": feats.loc[[cid]]}
    return new_ids


def restore_uploads():
    """Re-score customers uploaded before the last restart (only possible if the database kept them)."""
    with db() as con:
        rows = con.execute(select(uploads)).all()
    for r in rows:
        dates = pd.DatetimeIndex(json.loads(r.dates))
        vals = np.array([np.nan if v is None else v for v in json.loads(r.consumption)], dtype=np.float32)[None, :]
        add_uploaded([r.customer_id], dates, vals, save=False, stamp=r.uploaded_at)


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
    STATE["extra"] = {}
    init_db()
    seed_scores()
    restore_uploads()
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
    if customer_id not in STATE["demo_index"] and customer_id not in STATE["extra"]:
        raise HTTPException(404, "Customer not found")
    return customer_id


def check_customer_access(customer_id, user):
    """Managers may open any customer; an inspector only customers of cases assigned to them."""
    known_customer(customer_id)
    if user["r"] == "inspector":
        with db() as con:
            mine = con.execute(select(cases.c.case_id).where(
                (cases.c.customer_id == customer_id) & (cases.c.inspector == user["u"]))).first()
        if mine is None:
            raise HTTPException(403, "This customer is not in your assigned cases")


DECIDED = ("theft_confirmed", "normal")


def decided_ids(con):
    return select(cases.c.customer_id).where(cases.c.status.in_(DECIDED))


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
        pending = dict(con.execute(select(risk_scores.c.risk_band, func.count())
                                   .where(risk_scores.c.customer_id.not_in(decided_ids(con)))
                                   .group_by(risk_scores.c.risk_band)).all())
        by_status = dict(con.execute(q.group_by(cases.c.status)).all())
        n_decided = con.execute(select(func.count(func.distinct(cases.c.customer_id)))
                                .where(cases.c.status.in_(DECIDED))).scalar()
    return {"bands": bands, "bands_pending": pending, "decided_customers": n_decided, "cases": by_status,
            "uploaded_customers": len(STATE["extra"]), "threshold": STATE["bundle"]["threshold"],
            "model_version": STATE["bundle"]["version"]}


@app.get("/customers", dependencies=[Depends(check_key), Depends(require("manager"))])
def list_customers(band: Optional[Literal["High", "Medium", "Low"]] = None, limit: int = 100, offset: int = 0,
                   pending_only: bool = False):
    """Customers ranked by risk. pending_only=true hides customers whose case already has an outcome."""
    limit = max(1, min(limit, 5000))
    q = select(risk_scores)
    if band:
        q = q.where(risk_scores.c.risk_band == band)
    if pending_only:
        q = q.where(risk_scores.c.customer_id.not_in(decided_ids(None)))
    q = q.order_by(risk_scores.c.risk_score.desc()).limit(limit).offset(max(0, offset))
    with db() as con:
        return [row_to_dict(r) for r in con.execute(q).all()]


@app.get("/customers/{customer_id}", dependencies=[Depends(check_key)])
def get_customer(customer_id: str, user=Depends(require("inspector"))):
    check_customer_access(customer_id, user)
    q = visible_cases(user).where(cases.c.customer_id == customer_id).order_by(cases.c.case_id.desc())
    with db() as con:
        r = con.execute(select(risk_scores).where(risk_scores.c.customer_id == customer_id)).first()
        cs = con.execute(q).all()
    dates, series, feats = customer_data(customer_id)
    out = row_to_dict(r)
    out["dates"] = dates.strftime("%Y-%m-%d").tolist()
    out["consumption"] = [None if np.isnan(v) else round(float(v), 3) for v in series]
    out["cases"] = [case_dict(c) for c in cs]
    out["explanation"] = explain_one(STATE["bundle"], feats)
    out["uploaded"] = customer_id in STATE["extra"]
    return out


class WhatIf(BaseModel):
    customer_id: str
    changes: Dict[str, float] = {}


@app.post("/whatif", dependencies=[Depends(check_key)])
def what_if(body: WhatIf, user=Depends(require("inspector"))):
    """Re-score a customer after changing some of its feature values (the model is not changed)."""
    check_customer_access(body.customer_id, user)
    unknown = set(body.changes) - set(FEATURE_NAMES)
    if unknown:
        raise HTTPException(422, f"Unknown features: {sorted(unknown)}")
    b = STATE["bundle"]
    original = customer_data(body.customer_id)[2]
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
def similar_customers(customer_id: str, k: int = 5, user=Depends(require("inspector"))):
    """Nearest demo customers in (log-scaled, standardised) feature space."""
    check_customer_access(customer_id, user)
    k = max(1, min(k, 20))
    ids, _, dates, values = STATE["demo"]
    z = STATE["demo_z"]
    c_dates, c_values, c_feats = customer_data(customer_id)
    f = c_feats.to_numpy()[0]
    zc = (np.sign(f) * np.log1p(np.abs(f)) - STATE["bundle"]["sim_mean"]) / STATE["bundle"]["sim_std"]
    dist = np.sqrt(((z - zc) ** 2).sum(axis=1))
    if customer_id in STATE["demo_index"]:
        dist[STATE["demo_index"][customer_id]] = np.inf
    nearest = np.argsort(dist)[:k]
    with db() as con:
        rows = {r.customer_id: r for r in con.execute(
            select(risk_scores).where(risk_scores.c.customer_id.in_([ids[j] for j in nearest]))).all()}
    monthly = nan_monthly_means(values[nearest], dates)
    months = dates.to_period("M").unique().astype(str).tolist()
    own = nan_monthly_means(c_values[None, :], c_dates)[0]
    own = pd.Series(own, index=c_dates.to_period("M").unique().astype(str)).reindex(months).to_numpy()
    clean = lambda row: [None if np.isnan(v) else round(float(v), 3) for v in row]
    return {"months": months, "customer_monthly": clean(own),
            "similar": [{"customer_id": ids[j], "distance": round(float(dist[j]), 3),
                         "risk_score": rows[ids[j]].risk_score, "risk_band": rows[ids[j]].risk_band,
                         "monthly": clean(monthly[n])} for n, j in enumerate(nearest)]}


@app.post("/predict", dependencies=[Depends(check_key), Depends(require("manager"))])
async def predict(file: UploadFile = File(...), user=Depends(require("manager"))):
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
    added = add_uploaded(ids, dates, values, user=user["u"])
    return {"model_version": STATE["bundle"]["version"], "days": len(dates),
            "added_to_dashboard": len(added), "already_in_system": int(len(ids) - len(added)),
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
def create_case(body: CaseCreate, user=Depends(require("manager"))):
    """Only a manager opens a case, and always assigns it to an inspector."""
    known_customer(body.customer_id)
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

"""ENZERA frontend (Streamlit). Talks to the FastAPI backend.

Run locally:  streamlit run frontend/app.py
Backend URL:  env BACKEND_URL, or BACKEND_URL in .streamlit/secrets.toml (Streamlit Cloud)
Styling (CSS + card helpers) lives in frontend/style.py, the PDF inspection sheet in frontend/sheet.py.
"""
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sheet  # noqa: E402
import style as ui  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def setting(name, default=None):
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets[name]
    except Exception:
        return default


BACKEND = setting("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_KEY = setting("API_KEY")

st.set_page_config(page_title="ENZERA", page_icon="⚡", layout="wide", initial_sidebar_state="auto")
ui.inject_css()
st.session_state.setdefault("auth", None)  # {"token", "username", "role"} after login

# ---------------------------------------------------------------- backend calls
WAKING_UP = ("The backend may be **asleep**: the free Render plan stops it after 15 minutes without visitors, "
             "and it takes about 30-60 seconds to start again. Wait a moment, then press **Try again**.")


def headers():
    h = {"X-API-Key": API_KEY} if API_KEY else {}
    if st.session_state.auth:
        h["Authorization"] = f"Bearer {st.session_state.auth['token']}"
    return h


def show_error(title, detail):
    st.error(f"**{title}**\n\n{detail}")
    if st.button("Try again"):
        st.rerun()
    st.stop()


def api(method, path, **kw):
    try:
        r = requests.request(method, BACKEND + path, headers=headers(), timeout=90, **kw)
    except requests.Timeout:
        show_error("The backend did not answer within 90 seconds.", WAKING_UP)
    except requests.ConnectionError:
        show_error(f"Cannot connect to the backend at {BACKEND}.",
                   WAKING_UP + "\n\nRunning locally? Start it first with `uvicorn backend.main:app`.")
    except requests.RequestException as e:
        show_error("Request to the backend failed.", str(e))
    if r.status_code in (502, 503, 504):  # what the hosting proxy returns while the app is starting
        show_error(f"The backend is not ready yet (HTTP {r.status_code}).", WAKING_UP)
    if r.status_code == 401 and st.session_state.auth and path.startswith("/cases"):
        st.session_state.auth = None
        show_error("Your session has expired.", "Please log in again in the sidebar.")
    if r.status_code == 401:
        try:
            detail = r.json().get("detail", "")
        except ValueError:
            detail = ""
        show_error("Not authorised.", detail or "Set the same `API_KEY` on the backend and in the frontend secrets.")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        show_error(f"Backend error {r.status_code}", str(detail))
    return r.json() if r.content else None


@st.cache_data(ttl=30, show_spinner=False)
def backend_online():
    """Quick, non-blocking status check for the sidebar badge."""
    try:
        return requests.get(BACKEND + "/health", timeout=4).ok
    except requests.RequestException:
        return False


def role():
    return st.session_state.auth["role"] if st.session_state.auth else None


def reasons_text(reasons):
    return "; ".join(f"{x['label']}: {x['value']}" for x in reasons) or "No strong theft signals"


BAND_ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
BAND_SCALE = alt.Scale(domain=["High", "Medium", "Low"], range=[ui.BAND_COLOR[b] for b in ("High", "Medium", "Low")])
RISK_COLUMNS = {
    "customer_id": "Customer",
    "risk_score": st.column_config.ProgressColumn("Theft risk", min_value=0, max_value=1, format="%.2f"),
    "theft_probability": st.column_config.NumberColumn("Est. probability", format="%.0f%%",
                                                       help="Calibrated: share of similar-scored customers who were thieves"),
    "anomaly_score": st.column_config.ProgressColumn("Anomaly", min_value=0, max_value=1, format="%.2f"),
}


def risk_table(df, height=None):
    df = df.copy()
    df["theft_probability"] = df["theft_probability"] * 100
    df["Risk band"] = df["risk_band"].map(BAND_ICON)
    df["Top reasons"] = df["reasons"].apply(reasons_text)
    cols = ["customer_id", "risk_score", "theft_probability", "anomaly_score", "Risk band", "Top reasons"]
    st.dataframe(df[cols], column_config=RISK_COLUMNS, hide_index=True, use_container_width=True,
                 **({"height": height} if height else {}))
    return df[cols]


# ---------------------------------------------------------------- sidebar
PAGES = {"Risk dashboard": "📊", "Customer detail": "🔍", "Upload and score": "📤", "Inspection planner": "🧭",
         "Inspection cases": "🗂️", "Monitoring": "📡"}
with st.sidebar:
    status_html = ui.brand(backend_online(), BACKEND)
    page = st.radio("Navigate", list(PAGES), format_func=lambda p: f"{PAGES[p]}  {p}", label_visibility="collapsed")
    ui.html('<div class="nav-divider"></div>')
    if st.session_state.auth:
        a = st.session_state.auth
        ui.user_card(a["username"], a["role"])
        if st.button("Sign out", key="logout", use_container_width=True):
            st.session_state.auth = None
            st.rerun()
    else:
        ui.login_header()
        with st.form("login", clear_on_submit=False, border=False):
            user = st.text_input("Username", placeholder="e.g. inspector")
            pw = st.text_input("Password", type="password", placeholder="Your password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            if not user or not pw:
                st.warning("Enter both username and password.")
            else:
                try:
                    with st.spinner("Signing in…"):
                        r = requests.post(BACKEND + "/auth/login", json={"username": user.strip(), "password": pw},
                                          headers=headers(), timeout=90)
                except requests.RequestException:
                    r = None
                if r is not None and r.ok:
                    st.session_state.auth = r.json()
                    st.rerun()
                elif r is not None and r.status_code == 401:
                    st.error("Wrong username or password.")
                else:
                    st.error("Server is not responding (it may be waking up). Wait 30-60 seconds and try again.")
        ui.html('<div class="login-note">Viewing is open to everyone. Sign in to open cases (inspector) or '
                'record outcomes (manager).</div>')
    ui.html(status_html)

# ---------------------------------------------------------------- dashboard
if page == "Risk dashboard":
    ui.hero("Theft risk dashboard",
            "Every customer is scored by the XGBoost model and ranked by theft risk, so inspectors can visit "
            "the most suspicious meters first.", tag="Overview")
    stats = api("GET", "/stats")
    bands, total = stats["bands"], sum(stats["bands"].values())
    ui.kpis([
        ("Customers scored", f"{total:,}", "unseen demo split", "#2563EB"),
        ("High risk", bands.get("High", 0), f"{bands.get('High', 0) / max(total, 1):.1%} - inspect first", ui.BAND_COLOR["High"]),
        ("Medium risk", bands.get("Medium", 0), f"{bands.get('Medium', 0) / max(total, 1):.1%} of customers", ui.BAND_COLOR["Medium"]),
        ("Low risk", bands.get("Low", 0), f"{bands.get('Low', 0) / max(total, 1):.1%} of customers", ui.BAND_COLOR["Low"]),
        ("Open cases", stats["cases"].get("open", 0), f"threshold {stats['threshold']:.2f}", "#7C3AED"),
    ])
    all_rows = pd.DataFrame(api("GET", "/customers", params={"limit": 1000}))
    if not all_rows.empty:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            ui.section("📉", "Risk score distribution")
            hist = alt.Chart(all_rows).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("risk_score:Q", bin=alt.Bin(step=0.05), title="Theft risk score"),
                y=alt.Y("count():Q", title="Customers"),
                color=alt.Color("risk_band:N", scale=BAND_SCALE, legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("count():Q", title="Customers")]).properties(height=260)
            st.altair_chart(hist, use_container_width=True)
        with c2:
            ui.section("🎯", "Customers per risk band")
            bdf = pd.DataFrame({"band": list(bands), "count": list(bands.values())})
            donut = alt.Chart(bdf).mark_arc(innerRadius=60, cornerRadius=4).encode(
                theta="count:Q", color=alt.Color("band:N", scale=BAND_SCALE, legend=alt.Legend(title=None, orient="top")),
                tooltip=["band", "count"]).properties(height=260)
            st.altair_chart(donut, use_container_width=True)

    ui.section("📋", "Customers ranked by theft risk")
    band = st.radio("Risk band", ["All", "High", "Medium", "Low"], horizontal=True)
    df = all_rows if band == "All" else all_rows[all_rows["risk_band"] == band]
    if df.empty:
        st.info("No customers in this band.")
    else:
        risk_table(df.head(500), height=480)
        st.caption("Open a customer in **🔍 Customer detail** to see why it was flagged.")

# ---------------------------------------------------------------- detail
elif page == "Customer detail":
    ui.hero("Customer detail", "Why did the model flag this customer, what would change its score, and who looks "
            "similar? Every reason is an exact SHAP contribution.", tag="Explainability")
    top = api("GET", "/customers", params={"limit": 1000})
    labels = {r["customer_id"]: f"{r['customer_id']}  ·  risk {r['risk_score']:.2f}  ·  {r['risk_band']}" for r in top}
    cid = st.selectbox("Customer (highest risk first)", list(labels), format_func=labels.get)
    c = api("GET", f"/customers/{cid}")
    threshold = api("GET", "/stats")["threshold"]

    ui.risk_card(cid, c["risk_score"], c["risk_band"], c["anomaly_score"], threshold, c["theft_probability"])
    t_why, t_whatif, t_use, t_sim, t_insp = st.tabs(["🧠 Why flagged", "🎛️ What-if", "⚡ Consumption",
                                                     "👥 Similar customers", "🗂️ Inspection"])

    with t_why:
        if c["reasons"]:
            ui.reason_cards(c["reasons"])
        else:
            st.info("No feature pushes this customer towards theft.")
        ui.section("📊", "Full breakdown (SHAP waterfall)")
        e = c["explanation"]
        top_n = e["contributions"][:10]
        rest = sum(x["impact"] for x in e["contributions"][10:])
        steps = [{"step": "Base value (average customer)", "delta": e["base_value"]}] + \
                [{"step": f"{x['label']} = {x['value']}", "delta": x["impact"]} for x in top_n] + \
                [{"step": f"{len(e['contributions']) - 10} other features", "delta": rest}]
        wf = pd.DataFrame(steps)
        wf["end"] = wf["delta"].cumsum()
        wf["start"] = wf["end"] - wf["delta"]
        wf["effect"] = ["base"] + ["raises risk" if d > 0 else "lowers risk" for d in wf["delta"][1:]]
        wf["order"] = range(len(wf))
        chart = alt.Chart(wf).mark_bar(cornerRadius=3).encode(
            y=alt.Y("step:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("start:Q", title="Model output (log-odds)"), x2="end:Q",
            color=alt.Color("effect:N", scale=alt.Scale(domain=["base", "raises risk", "lowers risk"],
                                                        range=["#94A3B8", "#DC2626", "#16A34A"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["step", alt.Tooltip("delta:Q", format="+.3f", title="contribution")]
        ).properties(height=36 * len(wf))
        rule = alt.Chart(pd.DataFrame({"x": [e["log_odds"]]})).mark_rule(strokeDash=[4, 3], color="#0F172A").encode(x="x:Q")
        st.altair_chart(chart + rule, use_container_width=True)
        st.caption(f"Base value {e['base_value']:+.2f} + all contributions = {e['log_odds']:+.2f} log-odds "
                   f"→ risk score 1 / (1 + e^-({e['log_odds']:.2f})) = **{c['risk_score']:.2f}**. "
                   "Red bars push towards theft, green bars away from it.")

    with t_whatif:
        st.write("Change this customer's feature values and see how the model's score reacts. "
                 "Nothing is saved; the model itself is not changed.")
        feats = {x["feature"]: x for x in c["explanation"]["contributions"]}
        RANGES = {"ratio": (0.0, 1.0), "trend_slope": (-0.2, 0.2), "recent_vs_past": (0.0, 5.0),
                  "weekend_ratio": (0.0, 3.0), "weekly_autocorr": (-1.0, 1.0), "dow_strength": (0.0, 1.0),
                  "max_monthly_drop": (0.0, 1.0), "longest_zero_run": (0.0, 365.0)}
        changes = {}
        cols = st.columns(2)
        for i, x in enumerate(c["explanation"]["contributions"][:6]):  # the 6 most influential features
            f, v = x["feature"], float(x["raw_value"])
            lo, hi = RANGES.get(f) or (RANGES["ratio"] if f.endswith("_ratio") else (0.0, max(3 * v, 10.0)))
            lo, hi = min(lo, v), max(hi, v)
            new = cols[i % 2].slider(x["label"], float(lo), float(hi), float(v), key=f"wi_{cid}_{f}",
                                     help=f"Current value: {v:.3f}")
            if abs(new - v) > 1e-9:
                changes[f] = new
        res = api("POST", "/whatif", json={"customer_id": cid, "changes": changes})
        o, w = res["original"], res["what_if"]
        ui.kpis([("Current risk", f"{o['risk_score']:.2f}", f"{o['risk_band']} · prob {o['theft_probability']:.0%}",
                  ui.BAND_COLOR[o["risk_band"]]),
                 ("What-if risk", f"{w['risk_score']:.2f}", f"{w['risk_band']} · prob {w['theft_probability']:.0%}",
                  ui.BAND_COLOR[w["risk_band"]]),
                 ("Change", f"{w['risk_score'] - o['risk_score']:+.2f}", f"{len(changes)} feature(s) changed", "#7C3AED")])
        if changes:
            st.caption("Top contributions after the change: " + "; ".join(
                f"{x['label']} {x['impact']:+.2f}" for x in res["explanation"]["contributions"][:4]))

    with t_use:
        s = pd.Series(c["consumption"], index=pd.to_datetime(c["dates"]), dtype="float")
        view = st.radio("View", ["Monthly average", "Daily"], horizontal=True)
        series = s.resample("MS").mean() if view == "Monthly average" else s
        cdf = series.rename("kwh").rename_axis("date").reset_index()
        base = alt.Chart(cdf).encode(x=alt.X("date:T", title=None), y=alt.Y("kwh:Q", title="kWh per day"),
                                     tooltip=[alt.Tooltip("date:T"), alt.Tooltip("kwh:Q", format=".2f", title="kWh/day")])
        chart = (base.mark_area(opacity=0.18, color="#2563EB", interpolate="monotone")
                 + base.mark_line(color="#2563EB", strokeWidth=2, interpolate="monotone")).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"Missing days: {s.isna().sum()} of {len(s)} (gaps in the line are missing readings).")

    with t_sim:
        sim = api("GET", f"/customers/{cid}/similar", params={"k": 5})
        st.write("The 5 demo customers whose feature profile is closest to this one "
                 "(log-scaled, standardised features; smaller distance = more similar).")
        sdf = pd.DataFrame(sim["similar"])
        sdf["Risk band"] = sdf["risk_band"].map(BAND_ICON)
        st.dataframe(sdf[["customer_id", "distance", "risk_score", "Risk band"]],
                     column_config={"customer_id": "Customer", "risk_score": RISK_COLUMNS["risk_score"]},
                     hide_index=True, use_container_width=True)
        long = [{"customer": "this customer", "month": m, "kwh": v} for m, v in zip(sim["months"], sim["customer_monthly"])]
        for s_ in sim["similar"]:
            long += [{"customer": s_["customer_id"][:8] + "…", "month": m, "kwh": v} for m, v in zip(sim["months"], s_["monthly"])]
        ldf = pd.DataFrame(long)
        ldf["month"] = pd.to_datetime(ldf["month"])
        st.altair_chart(alt.Chart(ldf).mark_line(interpolate="monotone").encode(
            x=alt.X("month:T", title=None), y=alt.Y("kwh:Q", title="kWh/day (monthly mean)"),
            color=alt.Color("customer:N", legend=alt.Legend(orient="top", title=None)),
            strokeWidth=alt.condition(alt.datum.customer == "this customer", alt.value(3.5), alt.value(1.4)),
            tooltip=["customer", alt.Tooltip("kwh:Q", format=".2f")]).properties(height=300), use_container_width=True)

    with t_insp:
        version = api("GET", "/health")["model_version"]
        d1, d2 = st.columns(2)
        d1.download_button("📄 Download inspection sheet (PDF)", sheet.inspection_sheet(c, threshold, version),
                           f"enzera_inspection_{cid[:12]}.pdf", "application/pdf", use_container_width=True)
        shap_csv = pd.DataFrame(c["explanation"]["contributions"])[["feature", "label", "value", "impact"]]
        d2.download_button("⬇️ Download explanation (CSV)", shap_csv.to_csv(index=False).encode(),
                           f"enzera_explanation_{cid[:12]}.csv", "text/csv", use_container_width=True)
        if c["cases"]:
            st.dataframe(pd.DataFrame(c["cases"]), hide_index=True, use_container_width=True)
        if role():
            with st.form("new_case"):
                col1, col2 = st.columns(2)
                inspector = col1.text_input("Inspector name", value=st.session_state.auth["username"])
                notes = col2.text_input("Notes")
                if st.form_submit_button("Open inspection case", type="primary"):
                    api("POST", "/cases", json={"customer_id": cid, "inspector": inspector, "notes": notes})
                    st.success("Case opened.")
                    st.rerun()
        else:
            st.info("🔐 Log in as an inspector (sidebar) to open an inspection case.")

# ---------------------------------------------------------------- upload
elif page == "Upload and score":
    ui.hero("Upload and score", "Score new customers from a CSV file. Results come back in seconds, "
            "each with its top SHAP reasons, plus a check of how similar the data is to the training data.",
            tag="Batch scoring")
    ui.html("""<div class="card"><b>Accepted formats</b><br>
<b>Long</b>: columns <code>customer_id, date, consumption</code>, one row per customer per day &nbsp;·&nbsp;
<b>Wide</b>: SGCC format (<code>CONS_NO</code> then one column per day).<br>
<span style="color:#64748B">At least 30 days per customer, up to 2,000 customers, max 10 MB.</span></div>""")
    sample = ROOT / "data/samples/sample_upload.csv"
    if sample.exists():
        st.download_button("⬇️ Download a sample file", sample.read_bytes(), "sample_upload.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up and st.button("Score customers", type="primary"):
        with st.spinner("Scoring..."):
            st.session_state.last_upload = {"name": up.name, **api(
                "POST", "/predict", files={"file": (up.name, up.getvalue(), "text/csv")})}
    res = st.session_state.get("last_upload")
    if res:
        df = pd.DataFrame(res["results"])
        counts = df["risk_band"].value_counts()
        ui.kpis([("Customers scored", len(df), f"{res['days']} days · {res['name']}", "#2563EB"),
                 ("High risk", counts.get("High", 0), "inspect first", ui.BAND_COLOR["High"]),
                 ("Medium risk", counts.get("Medium", 0), "", ui.BAND_COLOR["Medium"]),
                 ("Low risk", counts.get("Low", 0), f"model {res['model_version']}", ui.BAND_COLOR["Low"])])
        out = risk_table(df)
        st.download_button("⬇️ Download results", out.to_csv(index=False).encode(), "enzera_scores.csv", "text/csv")
        drift = pd.DataFrame(res["drift"])
        n_major = int((drift["status"] == "major shift").sum())
        ui.section("📡", "Data drift check")
        if n_major:
            st.warning(f"{n_major} feature(s) differ strongly from the training data (PSI > 0.25). Scores for this file "
                       "may be less reliable. See **📡 Monitoring** for details.")
        else:
            st.success("No feature differs strongly from the training data (all PSI ≤ 0.25).")

# ---------------------------------------------------------------- planner
elif page == "Inspection planner":
    m = api("GET", "/metrics")
    cc = m["capture_curve"]
    rate = cc["test_thieves"] / cc["test_customers"]
    ui.hero("Inspection planner", "How many inspections should we plan, and how many thieves will they find? "
            f"Estimates use the model's measured performance on {cc['test_customers']:,} held-out test customers.",
            tag="Planning")
    c1, c2 = st.columns(2)
    population = c1.number_input("Customers in the area", min_value=100, value=1000, step=100)
    pct = c2.slider("Inspection budget (% of customers visited, highest risk first)", 1, 100, 10)
    visits = int(population * pct / 100)
    recall, precision = cc["recall"][pct - 1], cc["precision"][pct - 1]
    exp_thieves = population * rate
    ui.kpis([("Inspections", f"{visits:,}", f"{pct}% of {population:,} customers", "#2563EB"),
             ("Thieves found (expected)", f"{recall * exp_thieves:,.0f}", f"of ~{exp_thieves:,.0f} in the area", "#DC2626"),
             ("Share of thieves caught", f"{recall:.0%}", f"random visits: {pct}%", "#7C3AED"),
             ("Hit rate per visit", f"{precision:.0%}", f"random visits: {rate:.0%}", "#16A34A")])
    curve = pd.DataFrame({"pct": cc["pct"], "ENZERA model": [r * 100 for r in cc["recall"]], "Random inspection": cc["pct"]})
    lc = curve.melt("pct", var_name="strategy", value_name="recall")
    chart = alt.Chart(lc).mark_line(strokeWidth=2.5).encode(
        x=alt.X("pct:Q", title="% of customers inspected (highest risk first)"),
        y=alt.Y("recall:Q", title="% of all thieves found", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("strategy:N", scale=alt.Scale(range=["#2563EB", "#94A3B8"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["pct", alt.Tooltip("recall:Q", format=".1f", title="% of thieves found")])
    point = alt.Chart(pd.DataFrame({"pct": [pct], "recall": [recall * 100]})).mark_point(size=160, filled=True, color="#DC2626").encode(x="pct:Q", y="recall:Q")
    st.altair_chart((chart + point).properties(height=320), use_container_width=True)
    st.caption("Capture curve on the test split. Expected numbers assume the area has the same theft rate as SGCC "
               f"({rate:.1%}) and that the model performs as on the test set.")

    ui.section("📋", "Visit list for the demo customers")
    n_demo = st.slider("Number of demo customers to visit", 5, 200, min(100, max(5, int(1000 * pct / 100))))
    vlist = pd.DataFrame(api("GET", "/customers", params={"limit": n_demo}))
    out = risk_table(vlist, height=360)
    st.download_button("⬇️ Download visit list", out.to_csv(index=False).encode(), "enzera_visit_list.csv", "text/csv")

# ---------------------------------------------------------------- cases
elif page == "Inspection cases":
    ui.hero("Inspection cases", "Track field inspections from open to outcome, with a full audit trail. "
            "Confirmed outcomes become new labelled data for retraining the model.", tag="Workflow")
    all_cases = api("GET", "/cases")
    by = pd.Series([x["status"] for x in all_cases]).value_counts() if all_cases else pd.Series(dtype=int)
    ui.kpis([("All cases", len(all_cases), "", "#2563EB"),
             ("Open", by.get("open", 0), "awaiting inspection", "#F59E0B"),
             ("Theft confirmed", by.get("theft_confirmed", 0), "", "#DC2626"),
             ("Normal", by.get("normal", 0), "false alarms", "#16A34A")])
    status = st.radio("Status", ["All", "open", "theft_confirmed", "normal"], horizontal=True)
    cases = all_cases if status == "All" else [x for x in all_cases if x["status"] == status]
    if not cases:
        st.info("No cases here yet. Open one from 🔍 Customer detail → Inspection.")
    else:
        st.dataframe(pd.DataFrame(cases), hide_index=True, use_container_width=True)
        case_id = st.selectbox("Select a case", [x["case_id"] for x in cases],
                               format_func=lambda i: next(f"#{x['case_id']} · {x['customer_id'][:12]}… · {x['status']}"
                                                          for x in cases if x["case_id"] == i))
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            ui.section("✏️", "Update")
            if role() is None:
                st.info("🔐 Log in (sidebar) to update cases.")
            else:
                with st.form("update"):
                    outcomes = ["open", "theft_confirmed", "normal"]
                    current = next(x["status"] for x in cases if x["case_id"] == case_id)
                    new_status = st.selectbox("Outcome", outcomes, index=outcomes.index(current),
                                              disabled=role() != "manager",
                                              help=None if role() == "manager" else "Only managers can record outcomes")
                    notes = st.text_area("Notes")
                    b1, b2 = st.columns(2)
                    save = b1.form_submit_button("Save", type="primary")
                    remove = b2.form_submit_button("Delete case", disabled=role() != "manager")
                    if save:
                        body = {**({"status": new_status} if role() == "manager" else {}), **({"notes": notes} if notes else {})}
                        api("PATCH", f"/cases/{case_id}", json=body)
                        st.success("Case updated.")
                        st.rerun()
                    if remove:
                        api("DELETE", f"/cases/{case_id}")
                        st.success("Case deleted (its history is kept).")
                        st.rerun()
        with c2:
            ui.section("🕓", "History")
            ui.timeline(api("GET", f"/cases/{case_id}/history"))

    ui.section("🔁", "Learning from inspections")
    decided = by.get("theft_confirmed", 0) + by.get("normal", 0)
    st.write(f"**{decided}** decided case(s) so far. Every confirmed outcome (theft or normal) becomes a verified "
             "training label. The model is retrained regularly with these labels, and a new model goes live only "
             "if it performs at least as well as the current one.")
    if role() == "manager":
        labels = api("GET", "/labels")
        if labels:
            st.download_button("⬇️ Download labels (CSV)", pd.DataFrame(labels).to_csv(index=False).encode(),
                               "enzera_labels.csv", "text/csv")

# ---------------------------------------------------------------- monitoring
elif page == "Monitoring":
    ui.hero("Monitoring", "Is the data the model sees today still like the data it was trained on? "
            "If not (data drift), scores become less reliable and the model should be retrained.", tag="MLOps")
    st.markdown("**Population Stability Index (PSI)** per feature, comparing the share of customers in each training-data "
                "decile: < 0.10 stable · 0.10-0.25 moderate shift · > 0.25 major shift.")
    sources = {"Demo customers (live in the app)": api("GET", "/drift")["features"]}
    if st.session_state.get("last_upload"):
        sources[f"Last upload: {st.session_state.last_upload['name']}"] = st.session_state.last_upload["drift"]
    choice = st.radio("Data to check", list(sources), horizontal=True)
    d = pd.DataFrame(sources[choice])
    counts = d["status"].value_counts()
    ui.kpis([("Stable", counts.get("stable", 0), "PSI < 0.10", "#16A34A"),
             ("Moderate shift", counts.get("moderate shift", 0), "0.10 - 0.25", "#F59E0B"),
             ("Major shift", counts.get("major shift", 0), "PSI > 0.25", "#DC2626"),
             ("Largest PSI", f"{d['psi'].max():.3f}", d.iloc[0]["label"], "#2563EB")])
    st.altair_chart(alt.Chart(d).mark_bar(cornerRadius=3).encode(
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)), x=alt.X("psi:Q", title="PSI"),
        color=alt.Color("status:N", scale=alt.Scale(domain=["stable", "moderate shift", "major shift"],
                                                    range=["#16A34A", "#F59E0B", "#DC2626"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["label", alt.Tooltip("psi:Q", format=".3f"), "status"]).properties(height=28 * len(d)),
        use_container_width=True)
    if not st.session_state.get("last_upload"):
        st.caption("Upload a file in 📤 Upload and score to check its drift here as well.")

"""ENZERA frontend (Streamlit). Talks to the FastAPI backend.

Run locally:  streamlit run frontend/app.py
Backend URL:  env BACKEND_URL, or BACKEND_URL in .streamlit/secrets.toml (Streamlit Cloud)
Styling (CSS + card helpers) lives in frontend/style.py.
"""
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ui  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def setting(name, default=None):
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets[name]
    except Exception:
        return default


BACKEND = setting("BACKEND_URL", "http://localhost:8000").rstrip("/")
HEADERS = {"X-API-Key": setting("API_KEY")} if setting("API_KEY") else {}

st.set_page_config(page_title="ENZERA", page_icon="⚡", layout="wide", initial_sidebar_state="auto")
ui.inject_css()

# ---------------------------------------------------------------- backend calls
WAKING_UP = ("The backend may be **asleep**: the free Render plan stops it after 15 minutes without visitors, "
             "and it takes about 30-60 seconds to start again. Wait a moment, then press **Try again**.")


def show_error(title, detail):
    st.error(f"**{title}**\n\n{detail}")
    if st.button("Try again"):
        st.rerun()
    st.stop()


def api(method, path, **kw):
    try:
        r = requests.request(method, BACKEND + path, headers=HEADERS, timeout=90, **kw)
    except requests.Timeout:
        show_error("The backend did not answer within 90 seconds.", WAKING_UP)
    except requests.ConnectionError:
        show_error(f"Cannot connect to the backend at {BACKEND}.",
                   WAKING_UP + "\n\nRunning locally? Start it first with `uvicorn backend.main:app`.")
    except requests.RequestException as e:
        show_error("Request to the backend failed.", str(e))
    if r.status_code in (502, 503, 504):  # what the hosting proxy returns while the app is starting
        show_error(f"The backend is not ready yet (HTTP {r.status_code}).", WAKING_UP)
    if r.status_code == 401:
        show_error("The backend rejected the API key.",
                   "Set the same `API_KEY` on the backend and in the frontend secrets.")
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


def reasons_text(reasons):
    return "; ".join(f"{x['label']}: {x['value']}" for x in reasons) or "No strong theft signals"


BAND_ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
BAND_SCALE = alt.Scale(domain=["High", "Medium", "Low"], range=[ui.BAND_COLOR[b] for b in ("High", "Medium", "Low")])
RISK_COLUMNS = {
    "customer_id": "Customer",
    "risk_score": st.column_config.ProgressColumn("Theft risk", min_value=0, max_value=1, format="%.2f"),
    "anomaly_score": st.column_config.ProgressColumn("Anomaly", min_value=0, max_value=1, format="%.2f"),
}

# ---------------------------------------------------------------- sidebar
PAGES = {"Risk dashboard": "📊", "Customer detail": "🔍", "Upload and score": "📤",
         "Inspection cases": "🗂️", "Model performance": "📈", "About": "ℹ️"}
with st.sidebar:
    status_html = ui.brand(backend_online(), BACKEND)
    page = st.radio("Navigate", list(PAGES), format_func=lambda p: f"{PAGES[p]}  {p}", label_visibility="collapsed")
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
        df = df.head(500).copy()
        df["Risk band"] = df["risk_band"].map(BAND_ICON)
        df["Top reasons"] = df["reasons"].apply(reasons_text)
        st.dataframe(df[["customer_id", "risk_score", "anomaly_score", "Risk band", "Top reasons"]],
                     column_config=RISK_COLUMNS, hide_index=True, use_container_width=True, height=480)
        st.caption("Open a customer in **🔍 Customer detail** to see why it was flagged.")

# ---------------------------------------------------------------- detail
elif page == "Customer detail":
    ui.hero("Customer detail", "Why did the model flag this customer? Each reason is a SHAP contribution: "
            "how much that feature pushed the risk score up.", tag="Explainability")
    top = api("GET", "/customers", params={"limit": 200})
    labels = {r["customer_id"]: f"{r['customer_id']}  ·  risk {r['risk_score']:.2f}  ·  {r['risk_band']}" for r in top}
    cid = st.selectbox("Customer (highest risk first)", list(labels), format_func=labels.get)
    c = api("GET", f"/customers/{cid}")
    threshold = api("GET", "/stats")["threshold"]

    ui.risk_card(cid, c["risk_score"], c["risk_band"], c["anomaly_score"], threshold)

    ui.section("🧠", "Why was this customer flagged?")
    if c["reasons"]:
        ui.reason_cards(c["reasons"])
    else:
        st.info("No feature pushes this customer towards theft.")

    ui.section("⚡", "Consumption history")
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

    ui.section("🗂️", "Inspection")
    if c["cases"]:
        st.dataframe(pd.DataFrame(c["cases"]), hide_index=True, use_container_width=True)
    with st.form("new_case"):
        col1, col2 = st.columns(2)
        inspector = col1.text_input("Inspector name")
        notes = col2.text_input("Notes")
        if st.form_submit_button("Open inspection case", type="primary"):
            api("POST", "/cases", json={"customer_id": cid, "inspector": inspector, "notes": notes})
            st.success("Case opened.")
            st.rerun()

# ---------------------------------------------------------------- upload
elif page == "Upload and score":
    ui.hero("Upload and score", "Score new customers from a CSV file. Results come back in seconds, "
            "each with its top SHAP reasons.", tag="Batch scoring")
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
            res = api("POST", "/predict", files={"file": (up.name, up.getvalue(), "text/csv")})
        df = pd.DataFrame(res["results"])
        counts = df["risk_band"].value_counts()
        ui.kpis([("Customers scored", len(df), f"{res['days']} days of readings", "#2563EB"),
                 ("High risk", counts.get("High", 0), "inspect first", ui.BAND_COLOR["High"]),
                 ("Medium risk", counts.get("Medium", 0), "", ui.BAND_COLOR["Medium"]),
                 ("Low risk", counts.get("Low", 0), f"model {res['model_version']}", ui.BAND_COLOR["Low"])])
        df["Risk band"] = df["risk_band"].map(BAND_ICON)
        df["Top reasons"] = df["reasons"].apply(reasons_text)
        out = df[["customer_id", "risk_score", "anomaly_score", "Risk band", "Top reasons"]]
        st.dataframe(out, column_config=RISK_COLUMNS, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download results", out.to_csv(index=False).encode(), "enzera_scores.csv", "text/csv")

# ---------------------------------------------------------------- cases
elif page == "Inspection cases":
    ui.hero("Inspection cases", "Track field inspections from open to outcome. Confirmed outcomes become new "
            "labelled data for retraining the model.", tag="Workflow")
    all_cases = api("GET", "/cases")
    by = pd.Series([x["status"] for x in all_cases]).value_counts() if all_cases else pd.Series(dtype=int)
    ui.kpis([("All cases", len(all_cases), "", "#2563EB"),
             ("Open", by.get("open", 0), "awaiting inspection", "#F59E0B"),
             ("Theft confirmed", by.get("theft_confirmed", 0), "", "#DC2626"),
             ("Normal", by.get("normal", 0), "false alarms", "#16A34A")])
    status = st.radio("Status", ["All", "open", "theft_confirmed", "normal"], horizontal=True)
    cases = all_cases if status == "All" else [x for x in all_cases if x["status"] == status]
    if not cases:
        st.info("No cases here yet. Open one from 🔍 Customer detail.")
    else:
        st.dataframe(pd.DataFrame(cases), hide_index=True, use_container_width=True)
        ui.section("✏️", "Update a case")
        with st.form("update"):
            col1, col2 = st.columns(2)
            case_id = col1.selectbox("Case", [x["case_id"] for x in cases])
            new_status = col2.selectbox("Outcome", ["open", "theft_confirmed", "normal"])
            notes = st.text_area("Notes (optional)")
            if st.form_submit_button("Save", type="primary"):
                api("PATCH", f"/cases/{case_id}", json={"status": new_status, **({"notes": notes} if notes else {})})
                st.success("Case updated.")
                st.rerun()

# ---------------------------------------------------------------- performance
elif page == "Model performance":
    m = api("GET", "/metrics")
    d = m["dataset"]
    ui.hero("Model performance", f"Model <b>{m['model_version']}</b> trained on SGCC: {d['customers']:,} customers, "
            f"{d['days']:,} days, theft rate {d['theft_rate']:.1%}. All metrics on the held-out test split "
            f"({d['split']['test']:,} customers).", tag="Evaluation")
    x = m["results"]["xgboost"]
    ui.kpis([("PR-AUC", f"{x['pr_auc']:.3f}", f"random = {d['theft_rate']:.3f}", "#2563EB"),
             ("ROC-AUC", f"{x['roc_auc']:.3f}", "random = 0.500", "#7C3AED"),
             ("Theft caught in top 10%", f"{x['recall_at_top10']:.1%}", "random = 10%", "#DC2626"),
             ("Precision / Recall", f"{x['precision']:.2f} / {x['recall']:.2f}", f"threshold {x['threshold']:.2f}", "#16A34A")])

    ui.section("🏆", "Model comparison")
    rows = [{"Model": name.replace("_", " ").title() + (" ⭐" if name == m["main_model"] else ""),
             "Precision": r.get("precision"), "Recall": r.get("recall"), "F1": r.get("f1"),
             "ROC-AUC": r.get("roc_auc"), "PR-AUC": r.get("pr_auc"),
             "Theft caught in top 10%": r.get("recall_at_top10")} for name, r in m["results"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    st.caption(f"Split: {d['split']['train']:,} train / {d['split']['test']:,} test / {d['split']['demo']:,} demo (unseen). "
               "Isolation Forest is unsupervised, so it has no threshold-based metrics.")

    tabs = st.tabs(["📈 Evaluation", "🔎 Exploratory analysis", "🧪 Experiments"])
    groups = [["roc_pr.png", "confusion_matrix.png", "shap_importance.png"],
              ["eda_class_balance.png", "eda_missing.png", "eda_avg_consumption.png", "eda_sample_curves.png"],
              ["experiments.png"]]
    for tab, files in zip(tabs, groups):
        with tab:
            cols = st.columns(2)
            for i, f in enumerate(x for x in files if (REPORTS / x).exists()):
                cols[i % 2].image(str(REPORTS / f), use_container_width=True)

# ---------------------------------------------------------------- about
else:
    ui.hero("ENZERA", "An Explainable AI-Based System for Electricity Theft and Abnormal Consumption Detection",
            tag="About the project")
    ui.section("👥", "Team")
    ui.kpis([("Member A", "Member A name", "role / contribution", "#2563EB"),
             ("Member B", "Member B name", "role / contribution", "#7C3AED")])
    st.caption("M.Tech Data Science · Applied Machine Learning project")

    ui.section("🔄", "How it works")
    ui.steps([("📥", "Smart-meter data", "Daily kWh readings (SGCC)"),
              ("🧹", "Clean + features", "14 consumption features"),
              ("🌲", "XGBoost + IForest", "Theft risk + anomaly score"),
              ("🧠", "TreeSHAP", "Top reasons per customer"),
              ("🗂️", "Inspection", "Open and close cases")])

    ui.section("📚", "Details")
    st.markdown("""
- **Model:** XGBoost classifier on 14 engineered consumption features, compared with Logistic Regression and
  Random Forest baselines. Class imbalance handled with class weights; decision threshold tuned on a validation split.
- **Abnormal consumption:** Isolation Forest trained on normal customers only.
- **Explainability:** exact TreeSHAP values per customer; the top positive contributors are shown as reasons.
- **Data:** SGCC Electricity Theft Detection dataset (State Grid Corporation of China), 42,372 customers.
  Customers shown in this app come from a demo split the model never saw during training.
- **Stack:** FastAPI backend, SQLite database, Streamlit frontend, Docker, GitHub Actions CI.

Reference: Zheng et al., *Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure
Smart Grids*, IEEE Transactions on Industrial Informatics, 2018.
""")
    with st.expander("Backend health"):
        st.json(api("GET", "/health"))

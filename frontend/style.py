"""Visual styling for the ENZERA frontend: one CSS block + small HTML helpers.

Kept separate from app.py so the page logic stays easy to read. All layouts use CSS grid with
`auto-fit`, so cards wrap onto new rows on phones and tablets (responsive design).
"""
import base64
from html import escape
from pathlib import Path

import streamlit as st

LOGO_PNG = Path(__file__).resolve().parent / "assets/logo.png"
LOGO_URI = "data:image/png;base64," + base64.b64encode(LOGO_PNG.read_bytes()).decode()

BAND_COLOR = {"High": "#DC2626", "Medium": "#F59E0B", "Low": "#16A34A"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 3.2rem; max-width: 1280px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1F3A 0%, #12305C 100%); }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .55rem .8rem; border-radius: 10px; margin-bottom: 4px; width: 100%;
  transition: background .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(59,130,246,.28); box-shadow: inset 3px 0 0 #60A5FA;
}
.brand { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .3rem; }
.brand .logo { width:44px; height:44px; border-radius:50%; box-shadow:0 6px 18px rgba(34,211,238,.35); }
/* hide Streamlit's own toolbar, menu and decoration line inside the app */
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu,
[data-testid="stAppDeployButton"], [data-testid="stHeaderActionElements"] { display:none !important; }
footer { visibility:hidden; }
.brand .name { font-size:1.45rem; font-weight:800; letter-spacing:.06em; }
.brand-sub { font-size:.78rem; opacity:.75; line-height:1.35; margin-bottom:1rem; }

/* ---------- hero banner ---------- */
.hero { position:relative; overflow:hidden; border-radius:20px; padding:1.6rem 1.8rem; margin-bottom:1.3rem; color:#fff;
  background: radial-gradient(circle at 85% 20%, rgba(96,165,250,.45), transparent 45%),
              linear-gradient(120deg, #0B1F3A 0%, #1D4ED8 100%);
  box-shadow: 0 12px 30px rgba(29,78,216,.22); }
.hero h1 { color:#fff !important; font-size:clamp(1.4rem, 3.2vw, 2.1rem); font-weight:800; margin:0 0 .3rem; padding:0; }
.hero p { margin:0; opacity:.88; font-size:clamp(.9rem, 1.6vw, 1rem); max-width:760px; }
.hero .tag { display:inline-block; font-size:.72rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
  background:rgba(255,255,255,.16); padding:.25rem .6rem; border-radius:999px; margin-bottom:.6rem; }

/* ---------- KPI cards ---------- */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:14px; margin-bottom:1.2rem; }
.kpi { background:#fff; border-radius:16px; padding:1rem 1.1rem; border:1px solid #E5E9F2;
  box-shadow:0 2px 10px rgba(15,23,42,.05); border-top:4px solid #2563EB;
  transition: transform .15s ease, box-shadow .15s ease; }
.kpi:hover { transform: translateY(-2px); box-shadow:0 8px 22px rgba(15,23,42,.09); }
.kpi .label { font-size:.78rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.kpi .value { font-size:1.75rem; font-weight:800; color:#0F172A; line-height:1.2; margin-top:.2rem; }
.kpi .sub { font-size:.78rem; color:#64748B; margin-top:.15rem; }

/* ---------- generic card + section titles ---------- */
.card { background:#fff; border:1px solid #E5E9F2; border-radius:16px; padding:1.1rem 1.2rem;
  box-shadow:0 2px 10px rgba(15,23,42,.05); margin-bottom:1rem; }
.section { font-size:1.08rem; font-weight:700; color:#0F172A; margin:1.2rem 0 .6rem; display:flex; align-items:center; gap:.45rem; }

/* ---------- risk pill + meter ---------- */
.pill { display:inline-block; padding:.22rem .7rem; border-radius:999px; font-size:.8rem; font-weight:700; border:1px solid; }
.meter { height:12px; border-radius:999px; background:#EEF2F7; overflow:hidden; margin:.5rem 0 .2rem; }
.meter > span { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,#16A34A,#F59E0B 55%,#DC2626); }

/* ---------- SHAP reason cards ---------- */
.reasons { display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:12px; }
.reason { background:#fff; border:1px solid #E5E9F2; border-left:4px solid #DC2626; border-radius:14px; padding:.9rem 1rem; }
.reason .r-label { font-weight:600; color:#0F172A; font-size:.92rem; }
.reason .r-value { font-size:1.35rem; font-weight:800; color:#0F172A; margin:.15rem 0; }
.reason .r-impact { font-size:.78rem; color:#B91C1C; font-weight:600; }
.bar { height:6px; border-radius:999px; background:#FEE2E2; margin-top:.45rem; overflow:hidden; }
.bar > span { display:block; height:100%; background:#DC2626; border-radius:999px; }

/* ---------- pipeline steps ---------- */
.steps { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; }
.step { background:#fff; border:1px solid #E5E9F2; border-radius:14px; padding:.9rem; text-align:center; }
.step .icon { font-size:1.6rem; } .step .t { font-weight:700; margin-top:.25rem; } .step .d { font-size:.8rem; color:#64748B; }

/* ---------- case timeline ---------- */
.tl-item { display:flex; gap:.7rem; padding:.55rem 0; border-bottom:1px dashed #E5E9F2; }
.tl-item:last-child { border-bottom:none; }
.tl-icon { width:30px; height:30px; flex:0 0 30px; border-radius:50%; background:#EEF2FF; display:grid; place-items:center; }
.tl-role { font-size:.7rem; background:#F1F5F9; color:#475569; padding:.05rem .45rem; border-radius:999px; }
.tl-time { font-size:.75rem; color:#64748B; } .tl-note { font-size:.82rem; color:#334155; margin-top:.15rem; font-style:italic; }

/* ---------- widgets ---------- */
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid #E5E9F2; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius:10px; font-weight:600; padding:.5rem 1.1rem; }
[data-testid="stFileUploaderDropzone"] { border-radius:14px; border:2px dashed #93C5FD; background:#F8FBFF; }
[data-testid="stForm"] { background:#fff; border-radius:16px; border:1px solid #E5E9F2; }
[data-testid="stImage"] img { border-radius:12px; border:1px solid #E5E9F2; background:#fff; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.5rem 1rem; font-weight:600; }

/* ---------- sidebar sign-in ---------- */
.nav-divider { height:1px; background:rgba(255,255,255,.12); margin:.9rem 0 .9rem; }
[data-testid="stSidebar"] .stButton > button { background:transparent !important; border:1px solid rgba(255,255,255,.25) !important; }
[data-testid="stSidebar"] .stButton > button:hover { background:rgba(255,255,255,.08) !important; }

/* ---------- signed-in profile card (sidebar) ---------- */
.profile { padding:.9rem .9rem .7rem; border-radius:14px; background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.12); margin-bottom:.6rem; }
.profile .top { display:flex; gap:.7rem; align-items:center; }
.profile .avatar { width:42px; height:42px; flex:0 0 42px; border-radius:50%; display:grid; place-items:center;
  font-weight:800; font-size:1rem; color:#fff !important; }
.profile .n { font-weight:700; font-size:.98rem; line-height:1.2; }
.profile .t { font-size:.74rem; opacity:.75; }
.profile .role { display:inline-block; font-size:.66rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  padding:.08rem .5rem; border-radius:999px; margin-top:.2rem; }
.profile dl { display:grid; grid-template-columns:auto 1fr; gap:.25rem .6rem; margin:.7rem 0 .2rem; font-size:.75rem; }
.profile dt { opacity:.6; } .profile dd { margin:0; word-break:break-word; }
.profile .can { font-size:.72rem; opacity:.75; line-height:1.35; border-top:1px solid rgba(255,255,255,.1);
  padding-top:.5rem; margin-top:.4rem; }

/* ---------- phones ---------- */
@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding-left:.8rem; padding-right:.8rem; padding-top:3rem; }
  .hero { padding:1.1rem 1.1rem; border-radius:16px; }
  .kpi-grid { grid-template-columns:repeat(2, 1fr); gap:10px; }
  .kpi .value { font-size:1.35rem; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def brand():
    html(f"""<div class="brand"><img class="logo" src="{LOGO_URI}" alt="ENZERA logo"><div class="name">ENZERA</div></div>
<div class="brand-sub">Explainable electricity theft &amp; abnormal consumption detection</div>""")


def hero(title, subtitle, tag=None):
    t = f'<div class="tag">{escape(tag)}</div>' if tag else ""
    html(f'<div class="hero">{t}<h1>{escape(title)}</h1><p>{subtitle}</p></div>')


def kpis(items):
    """items: list of (label, value, sub, colour)."""
    cards = "".join(
        f'<div class="kpi" style="border-top-color:{c}"><div class="label">{escape(str(l))}</div>'
        f'<div class="value">{escape(str(v))}</div><div class="sub">{escape(str(s))}</div></div>'
        for l, v, s, c in items)
    html(f'<div class="kpi-grid">{cards}</div>')


def section(icon, text):
    html(f'<div class="section">{icon} {escape(text)}</div>')


def pill(band):
    c = BAND_COLOR.get(band, "#64748B")  # colour + 13% / 33% alpha versions for background and border
    return f'<span class="pill" style="color:{c};background:{c}22;border-color:{c}55">{escape(band)} risk</span>'


def risk_card(cid, risk, band, anomaly, threshold, probability=None):
    prob_tile = "" if probability is None else (
        f'<div class="kpi" style="border-top-color:#0EA5E9"><div class="label">Estimated probability</div>'
        f'<div class="value">{probability:.0%}</div><div class="meter"><span style="width:{probability * 100:.1f}%;'
        f'background:#0EA5E9"></span></div><div class="sub">calibrated on validation data</div></div>')
    html(f"""<div class="card">
<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:.6rem;align-items:center">
  <div><div style="font-size:.78rem;color:#64748B;font-weight:600">CUSTOMER</div>
       <div style="font-weight:800;font-size:1.05rem;word-break:break-all">{escape(cid)}</div></div>
  <div>{pill(band)}</div>
</div>
<div class="kpi-grid" style="margin:1rem 0 .2rem">
  <div class="kpi" style="border-top-color:{BAND_COLOR.get(band)}"><div class="label">Theft risk</div><div class="value">{risk:.2f}</div>
    <div class="meter"><span style="width:{risk * 100:.1f}%"></span></div><div class="sub">High-risk threshold {threshold:.2f}</div></div>
  <div class="kpi" style="border-top-color:#7C3AED"><div class="label">Anomaly score</div><div class="value">{anomaly:.2f}</div>
    <div class="meter"><span style="width:{anomaly * 100:.1f}%;background:#7C3AED"></span></div><div class="sub">Isolation Forest (unsupervised)</div></div>
  {prob_tile}
</div></div>""")


def reason_cards(reasons):
    top = max([r["impact"] for r in reasons] + [1e-9])
    cards = "".join(
        f'<div class="reason"><div class="r-label">{escape(r["label"])}</div>'
        f'<div class="r-value">{escape(str(r["value"]))}</div>'
        f'<div class="r-impact">raises risk by {r["impact"]:+.2f} (SHAP)</div>'
        f'<div class="bar"><span style="width:{r["impact"] / top * 100:.0f}%"></span></div></div>'
        for r in reasons)
    html(f'<div class="reasons">{cards}</div>')


def steps(items):
    html('<div class="steps">' + "".join(
        f'<div class="step"><div class="icon">{i}</div><div class="t">{escape(t)}</div><div class="d">{escape(d)}</div></div>'
        for i, t, d in items) + "</div>")


EVENT_ICON = {"created": "🆕", "status_changed": "🔁", "note_updated": "📝", "deleted": "🗑️"}


def timeline(events):
    """Audit trail of a case, newest last."""
    items = []
    for e in events:
        what = {"created": "opened the case",
                "status_changed": f"changed status {escape(str(e['from_status']))} → <b>{escape(str(e['to_status']))}</b>",
                "note_updated": "added a note", "deleted": "deleted the case"}.get(e["action"], escape(e["action"]))
        note = f'<div class="tl-note">“{escape(e["notes"])}”</div>' if e.get("notes") else ""
        items.append(f'<div class="tl-item"><div class="tl-icon">{EVENT_ICON.get(e["action"], "•")}</div><div>'
                     f'<div><b>{escape(str(e["username"]))}</b> <span class="tl-role">{escape(str(e["role"]))}</span> {what}</div>'
                     f'<div class="tl-time">{escape(str(e["at"]).replace("T", " ")[:16])} UTC</div>{note}</div></div>')
    html('<div class="card">' + "".join(items) + "</div>")


ROLE_STYLE = {"manager": ("#7C3AED", "Record outcomes, delete cases, export labels"),
              "inspector": ("#2563EB", "Open cases and add inspection notes")}


LOGIN_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display:none; }
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at 15% 10%, rgba(96,165,250,.35), transparent 40%),
  radial-gradient(circle at 85% 90%, rgba(245,158,11,.18), transparent 40%), linear-gradient(135deg,#0B1F3A,#12305C); }
[data-testid="stHeader"] { background:transparent; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top:8vh; }
.login-brand { text-align:center; color:#fff; margin-bottom:1.2rem; }
.login-brand .logo { width:150px; height:150px; margin:0 auto .7rem; display:block; border-radius:50%;
  box-shadow:0 10px 30px rgba(34,211,238,.35); }
.login-brand .name { font-size:2rem; font-weight:800; letter-spacing:.08em; }
.login-brand .sub { font-size:.9rem; opacity:.8; margin-top:.2rem; }
.login-brand .h { font-size:1.1rem; font-weight:700; margin-top:1.4rem; }
[data-testid="stForm"] { background:#fff; border-radius:18px; padding:1.4rem 1.4rem .9rem;
  box-shadow:0 20px 50px rgba(2,6,23,.35); border:none; }
[data-testid="InputInstructions"] { display:none; }
[data-testid="stFormSubmitButton"] > button { background:linear-gradient(135deg,#3B82F6,#2563EB); border:none; color:#fff;
  padding:.6rem; box-shadow:0 6px 16px rgba(37,99,235,.35); }
.login-foot { text-align:center; color:#CBD5E1; font-size:.8rem; margin-top:1rem; line-height:1.45; }
[data-testid="stForm"] [data-baseweb="input"] { background:#F1F5F9; border:1px solid #CBD5E1; border-radius:10px; }
[data-testid="stForm"] [data-baseweb="input"]:focus-within { border-color:#3B82F6; box-shadow:0 0 0 3px rgba(59,130,246,.2); }
[data-testid="stForm"] [data-baseweb="base-input"], [data-testid="stForm"] input { background:transparent; }
input::-ms-reveal, input::-ms-clear { display:none; }
[data-testid="stAlert"] { background:#fff; border-radius:12px; box-shadow:0 8px 20px rgba(2,6,23,.25); }
[data-testid="stAlert"] p { font-weight:600; }
</style>
"""


def login_page_css():
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)


def login_brand():
    html(f'<div class="login-brand"><img class="logo" src="{LOGO_URI}" alt="ENZERA logo">'
         '<div class="sub">Electricity theft detection for inspection teams</div>'
         '<div class="h">Staff sign in</div></div>')


def profile_card(username, role, p):
    """Details of the signed-in user only (inspector or manager)."""
    color, can = ROLE_STYLE.get(role, ("#0EA5E9", ""))
    name = p.get("full_name") or username
    initials = "".join(w[0] for w in name.split()[:2]).upper()
    rows = [("Username", username), ("Employee ID", p.get("employee_id")), ("Zone", p.get("zone")),
            ("Phone", p.get("phone")), ("Email", p.get("email"))]
    dl = "".join(f"<dt>{k}</dt><dd>{escape(str(v))}</dd>" for k, v in rows if v)
    html(f'<div class="profile"><div class="top"><div class="avatar" style="background:{color}">{escape(initials)}</div>'
         f'<div><div class="n">{escape(name)}</div><div class="t">{escape(p.get("title", ""))}</div>'
         f'<span class="role" style="background:{color}33;border:1px solid {color}">{escape(role)}</span></div></div>'
         f'<dl>{dl}</dl><div class="can">Can: {escape(can)}</div></div>')

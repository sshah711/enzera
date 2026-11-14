"""Visual styling for the ENZERA frontend: one CSS block + small HTML helpers.

Kept separate from app.py so the page logic stays easy to read. All layouts use CSS grid with
`auto-fit`, so cards wrap onto new rows on phones and tablets (responsive design).
"""
from html import escape

import streamlit as st

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
.brand .bolt { width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-size:1.3rem;
  background: linear-gradient(135deg,#F59E0B,#FBBF24); box-shadow:0 6px 18px rgba(245,158,11,.35); }
.brand .name { font-size:1.45rem; font-weight:800; letter-spacing:.06em; }
.brand-sub { font-size:.78rem; opacity:.75; line-height:1.35; margin-bottom:1rem; }
.status { font-size:.78rem; padding:.45rem .7rem; border-radius:10px; background:rgba(255,255,255,.07); margin-top:.6rem; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }

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

/* ---------- pipeline steps (About) ---------- */
.steps { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; }
.step { background:#fff; border:1px solid #E5E9F2; border-radius:14px; padding:.9rem; text-align:center; }
.step .icon { font-size:1.6rem; } .step .t { font-weight:700; margin-top:.25rem; } .step .d { font-size:.8rem; color:#64748B; }

/* ---------- widgets ---------- */
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid #E5E9F2; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius:10px; font-weight:600; padding:.5rem 1.1rem; }
[data-testid="stFileUploaderDropzone"] { border-radius:14px; border:2px dashed #93C5FD; background:#F8FBFF; }
[data-testid="stForm"] { background:#fff; border-radius:16px; border:1px solid #E5E9F2; }
[data-testid="stImage"] img { border-radius:12px; border:1px solid #E5E9F2; background:#fff; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.5rem 1rem; font-weight:600; }

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


def brand(backend_ok, backend_url):
    color, text = ("#22C55E", "Backend online") if backend_ok else ("#F87171", "Backend offline / waking up")
    html(f"""<div class="brand"><div class="bolt">⚡</div><div class="name">ENZERA</div></div>
<div class="brand-sub">Explainable electricity theft &amp; abnormal consumption detection</div>""")
    return f'<div class="status"><span class="dot" style="background:{color}"></span>{text}<br>' \
           f'<span style="opacity:.6">{escape(backend_url)}</span></div>'


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


def risk_card(cid, risk, band, anomaly, threshold):
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

"""One-page PDF inspection sheet for a customer (feature: downloadable field report).

Built with fpdf2 (pure Python). The consumption chart is drawn with plain lines, so no image library is needed.
"""
from datetime import datetime

import pandas as pd
from fpdf import FPDF

BAND_RGB = {"High": (220, 38, 38), "Medium": (245, 158, 11), "Low": (22, 163, 74)}


def _t(s):
    """Core PDF fonts only support Latin-1 text."""
    return str(s).encode("latin-1", "replace").decode("latin-1")


def inspection_sheet(c, threshold, model_version) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()
    W = pdf.w - 2 * pdf.l_margin

    # header bar
    pdf.set_fill_color(11, 31, 58); pdf.rect(0, 0, pdf.w, 26, "F")
    pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 16); pdf.set_xy(10, 7)
    pdf.cell(0, 7, "ENZERA - Inspection sheet")
    pdf.set_font("Helvetica", "", 9); pdf.set_xy(10, 15)
    pdf.cell(0, 5, _t(f"Generated {datetime.now():%Y-%m-%d %H:%M} | model {model_version}"))
    pdf.set_text_color(15, 23, 42); pdf.set_y(32)

    # summary
    pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 6, _t(f"Customer {c['customer_id']}"), new_x="LMARGIN", new_y="NEXT")
    r, g, b = BAND_RGB.get(c["risk_band"], (100, 116, 139))
    pdf.set_font("Helvetica", "", 10)
    rows = [("Theft risk score", f"{c['risk_score']:.2f}   (High-risk threshold {threshold:.2f})"),
            ("Estimated theft probability", f"{c.get('theft_probability', float('nan')):.0%} (calibrated)"),
            ("Anomaly score", f"{c['anomaly_score']:.2f}   (Isolation Forest, 0-1)")]
    pdf.set_fill_color(r, g, b); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 7, _t(f"{c['risk_band']} risk"), align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(15, 23, 42); pdf.ln(1)
    for k, v in rows:
        pdf.set_font("Helvetica", "B", 9); pdf.cell(55, 6, _t(k))
        pdf.set_font("Helvetica", "", 9); pdf.cell(0, 6, _t(v), new_x="LMARGIN", new_y="NEXT")

    # reasons (SHAP)
    pdf.ln(2); pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Why the model flagged this customer (SHAP)", new_x="LMARGIN", new_y="NEXT")
    contribs = c["explanation"]["contributions"][:8]
    pdf.set_font("Helvetica", "B", 8); pdf.set_fill_color(238, 242, 247)
    for txt, w in [("Feature", 95), ("Value", 30), ("Effect on risk (log-odds)", 0)]:
        pdf.cell(w or W - 125, 6, txt, fill=True, new_x="RIGHT")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for x in contribs:
        pdf.cell(95, 5.5, _t(x["label"])); pdf.cell(30, 5.5, _t(x["value"]))
        pdf.set_text_color(*((185, 28, 28) if x["impact"] > 0 else (21, 128, 61)))
        pdf.cell(0, 5.5, f"{'raises' if x['impact'] > 0 else 'lowers'} {abs(x['impact']):.2f}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(15, 23, 42)

    # consumption chart (monthly means)
    pdf.ln(3); pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Monthly average consumption (kWh/day)", new_x="LMARGIN", new_y="NEXT")
    s = pd.Series(c["consumption"], index=pd.to_datetime(c["dates"]), dtype="float").resample("MS").mean()
    x0, y0, cw, ch = pdf.l_margin, pdf.get_y() + 2, W, 45
    pdf.set_draw_color(203, 213, 225); pdf.rect(x0, y0, cw, ch)
    vals = s.to_numpy()
    top = max(float(pd.Series(vals).max(skipna=True) or 0), 1e-6)
    pdf.set_font("Helvetica", "", 7); pdf.set_xy(x0 + 1, y0 + 1); pdf.cell(20, 3, f"{top:.1f}")
    pdf.set_draw_color(37, 99, 235); pdf.set_line_width(0.5)
    pts = [(x0 + 3 + i * (cw - 6) / max(len(vals) - 1, 1), y0 + ch - 3 - (v / top) * (ch - 8))
           for i, v in enumerate(vals)]
    for (xa, ya), (xb, yb), va, vb in zip(pts, pts[1:], vals, vals[1:]):
        if pd.notna(va) and pd.notna(vb):  # gaps = months without readings
            pdf.line(xa, ya, xb, yb)
    pdf.set_line_width(0.2); pdf.set_xy(x0, y0 + ch + 1)
    pdf.cell(cw / 2, 4, _t(s.index[0].strftime("%b %Y"))); pdf.cell(cw / 2, 4, _t(s.index[-1].strftime("%b %Y")), align="R")
    missing = sum(v is None for v in c["consumption"])
    pdf.set_xy(x0, y0 + ch + 5); pdf.cell(0, 4, _t(f"Missing days: {missing} of {len(c['consumption'])}"), new_x="LMARGIN", new_y="NEXT")

    # previous cases
    pdf.ln(2); pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Inspection history", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    if c["cases"]:
        for k in c["cases"][:6]:
            pdf.cell(0, 5, _t(f"Case #{k['case_id']}  {k['status']}  opened {k['created_at'][:10]} by {k['inspector'] or '-'}"
                              f"  {(k['notes'] or '')[:70]}"), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 5, "No previous cases.", new_x="LMARGIN", new_y="NEXT")

    # field section
    pdf.ln(3); pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Field inspection (to fill in on site)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for line in ["Date of visit: ____________________    Inspector: ______________________________",
                 "Meter seal intact:  [ ] yes  [ ] no        Bypass / tampering found:  [ ] yes  [ ] no",
                 "Outcome:  [ ] theft confirmed   [ ] normal        Signature: ______________________",
                 "Notes: ____________________________________________________________________________"]:
        pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(-14); pdf.set_font("Helvetica", "I", 7); pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 4, "A risk score prioritises inspections; it is not proof of theft. SHAP values explain the model, not causes.",
             align="C")
    return bytes(pdf.output())

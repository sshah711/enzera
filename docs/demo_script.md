# ENZERA - Demo script (3 minutes, + optional 2-minute extension)

## Before you start (do not skip)
- **Open https://enzera-api.onrender.com/health at least 2 minutes early** so Render wakes up (it sleeps after 15 minutes).
  Wait for `"status":"ok"`.
- Open **https://enzera.streamlit.app** and check the sidebar says **Backend online**.
- **Log in as manager** in the sidebar (🔐 Log in) using the demo account from the README, so you can open and close cases without
  switching.
- Have `data/samples/sample_upload.csv` ready (or download it from the Upload page). Zoom the browser to 90-100%.

## Script (≈ 3:00)

**0:00 - 0:25 · Risk dashboard**
> "ENZERA ranks electricity customers by theft risk. These 1,000 customers come from a split the model never saw. 80 are high risk.
> The histogram shows the red tail where inspectors should go."

**0:25 - 1:15 · Customer detail: why flagged**
The first customer `448FE010…` is selected (risk 0.97, estimated probability 80%).
> "Risk 0.97, about an 80% calibrated chance of theft. These cards are exact SHAP contributions: high day-to-day variation, 29% missing
> readings, a rising trend. The waterfall shows all 17: they add up exactly to the score."

Click **⚡ Consumption** → "65% drop in July 2014, nine months low, then back up: the classic theft pattern."
Click **🎛️ What-if** → drag *Day-to-day variation* down → "risk falls from 0.97 to 0.90; the model reacts as expected."

**1:15 - 1:40 · Upload and score**
Upload `sample_upload.csv` → **Score customers**.
> "New readings are scored in seconds, with reasons. The drift check warns that this file covers only one year, so its scores are less
> reliable. That's monitoring built in."

**1:40 - 2:20 · Inspection workflow**
Customer detail → **🗂️ Inspection** → show the **PDF inspection sheet** download → **Open inspection case**.
**Inspection cases** → select it → outcome **theft_confirmed** → **Save** → point at the **history timeline**.
> "Inspectors open cases, managers record outcomes, every action is audited. Confirmed outcomes become new training labels; our retrain
> script only deploys a new model if it's not worse."

**2:20 - 3:00 · Planner and performance**
**🧭 Inspection planner** → set budget 10%:
> "Visiting the top 10% finds about 43% of thieves, against 10% at random."

**Results** (from the report / Word guide, not the website):
> "PR-AUC 0.376, ROC-AUC 0.798, best of four models. Gap-aware features improved it from 0.332; SMOTE and tuning didn't. Thank you."

## Optional extension (if time allows, ≈ 2:00)
- **Similar customers** tab: "the 5 most similar profiles and their curves."
- **Monitoring** page: "PSI per feature; the demo customers are stable."
- Attack test (report / Word guide): "honest limitation: injected attacks aren't detected; the model learned who was caught."
- Deep learning (report / Word guide): "the CNN + XGBoost ensemble reaches 0.433; not deployed, to keep exact explanations."
- Customer `F21D74D3…`: "zero usage for 3 years: theft risk 0.0025 but anomaly 0.90."

## Backup plan
1. **Render slow or down:** wait 60 s, press **Try again** (the app explains the cold start).
2. **Still down → run locally** (two terminals in the project folder):
   ```
   venv\Scripts\python -m uvicorn backend.main:app
   venv\Scripts\python -m streamlit run frontend/app.py
   ```
   Open http://localhost:8501.
3. **No laptop / no internet:** play the pre-recorded **screen recording** of this script (Win + Alt + R, or OBS).

Note: without a `DATABASE_URL`, cases opened on Render disappear when it restarts. That is expected on the free plan.

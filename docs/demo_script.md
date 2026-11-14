# ENZERA - 3-minute demo script

## Before you start (do not skip)
- **Open https://enzera-api.onrender.com/health at least 2 minutes before the demo** so Render wakes up (free plan sleeps
  after 15 minutes). Wait until it shows `"status":"ok"`.
- Then open **https://enzera.streamlit.app** and check the sidebar badge says **Backend online**.
- Have `data/samples/sample_upload.csv` ready on the desktop (or download it from the Upload page).
- Close other tabs; zoom the browser to 90-100%.

## Script (≈ 3:00)

**0:00 - 0:30 · Risk dashboard**
> "ENZERA ranks electricity customers by theft risk. These 1,000 customers come from a demo split the model never saw.
> 73 are high risk, 261 medium. The histogram shows most customers are low risk; the red tail is where inspectors should go."

Point at the KPI cards, the histogram, then the ranked table and its "Top reasons" column.

**0:30 - 1:15 · Top customer and SHAP reasons**
Go to **Customer detail** (the first customer is selected: `57531380…`, risk 0.98).
> "Risk 0.98, far above the 0.67 threshold. Why? These cards are exact SHAP contributions: very high day-to-day variation,
> 57% missing readings, and a very high average of 220 kWh per day. The consumption history confirms it: an erratic,
> patchy, very large consumer."

Optional: pick `448FE010…` (second) and show the drop in mid-2014 followed by recovery: "the classic theft pattern".

**1:15 - 1:50 · Upload a CSV**
Go to **Upload and score** → upload `sample_upload.csv` → **Score customers**.
> "A utility can upload new readings in long or wide format. 50 customers are scored in seconds, each with its reasons, and
> the results can be downloaded."

**1:50 - 2:25 · Open and close a case**
Back to **Customer detail** → enter inspector name + note → **Open inspection case**.
Go to **Inspection cases** → select the case → outcome **theft_confirmed** → **Save**.
> "Inspectors track each visit. Confirmed outcomes become new labelled data for retraining."

**2:25 - 3:00 · Model performance**
Go to **Model performance**.
> "On the held-out test set, XGBoost reaches PR-AUC 0.332 and ROC-AUC 0.788. Inspecting the top 10% finds 40.5% of all
> thieves, four times better than random. It beats Logistic Regression and Random Forest; SMOTE and tuning did not help.
> All charts come straight from our training pipeline."

End: "Backend FastAPI on Render, frontend Streamlit, tests and CI on GitHub. Thank you."

## Backup plan
1. **Render is slow or down:** wait 60 s and press **Try again** in the app (the error message explains it).
2. **Still down → run locally** (two terminals in the project folder):
   ```
   venv\Scripts\python -m uvicorn backend.main:app
   venv\Scripts\python -m streamlit run frontend/app.py
   ```
   and open http://localhost:8501.
3. **No laptop / no internet:** play the pre-recorded **screen recording** of this exact script (record it beforehand,
   e.g. with the Windows Game Bar: Win + Alt + R, or OBS).

Note: on the free plan the case you open may disappear later (SQLite resets when Render restarts), which is expected.

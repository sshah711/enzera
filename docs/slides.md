# ENZERA - Slides outline (14 slides, ~12 minutes)

Numbers come from `reports/metrics.json` (model `xgb-20261114-1557`), `reports/*.csv` and `docs/explainability.md`.
Speaker notes are in *italics*.

---

## 1. Title
**ENZERA: An Explainable AI-Based System for Electricity Theft and Abnormal Consumption Detection**
- Sakshi Shah (26MCD021) · Dhruvi Patel (26MCD015), M.Tech Data Science, Applied Machine Learning
- Live: enzera.streamlit.app · github.com/sshah711/enzera
- Visual: dashboard screenshot

## 2. Problem
- Theft = non-technical losses; inspectors can visit only a few customers
- **Who should they visit first, why, and how many visits are enough?**
- Deliverable: ranked, explained, calibrated inspection list + inspection workflow

## 3. Data: SGCC
- 42,372 customers × 1,034 days (2014-2016), **8.53% theft**: accuracy is useless
- 25.7% readings missing; **22.3% of meters have no data before 2016**
- Split 70 / 15 / 15: train / test / **demo (never seen, shown in the app)**
- Visual: `eda_class_balance.png`, `eda_missing.png`

## 4. EDA insight
- Theft customers use **more**: median 8.32 vs 5.19 kWh/day, more variable, rising in 2016
- Zero days rare in both classes; best single feature AUC only 0.705
- Visual: `eda_avg_consumption.png`

## 5. Method
- Gap-aware cleaning → **17 features** (level, variability, low usage, trend, weekly pattern, data quality)
- LR, RF, **XGBoost (main)**, Isolation Forest; class weights 10.7; threshold 0.656 tuned on validation
- Platt calibration → real probabilities (Brier 0.122 → 0.065)
- Every change gated by a rule fixed in advance (+0.01 PR-AUC and top-10%, not worse on validation)

## 6. Results
| Model | F1 | ROC-AUC | PR-AUC | Theft in top 10% |
|---|---|---|---|---|
| Logistic Regression | 0.261 | 0.699 | 0.218 | 28.2% |
| Random Forest | 0.341 | 0.781 | 0.337 | 40.2% |
| **XGBoost** | **0.394** | **0.798** | **0.376** | **43.5%** |
| Isolation Forest | - | 0.586 | 0.109 | 13.8% |
- **Top 10% → 43.5% of thieves (4.3× random)**; top 1% → 76% of visits find a thief
- Visual: `roc_pr.png` + `confusion_matrix.png`

## 7. What improved the model (and what did not)
- ✅ Gap-aware + weekly features: PR-AUC 0.332 → **0.376** (CI [+0.017, +0.067])
- ❌ SMOTE (0.296), ❌ hyperparameter tuning (ties at 0.332)
- *Note: weekly features alone hurt; they only help once gaps are not filled with flat lines.*

## 8. Deep learning
- 1D-CNN on raw daily series: better on test but **worse on validation** → rejected alone
- **XGBoost + CNN ensemble: PR-AUC 0.433**, top-10% 46.6% (passes the rule)
- Not deployed: loses exact SHAP explanations + PyTorch too big for free hosting → future work (ONNX)

## 9. Honest limitation: synthetic attacks
- Injected Jokar-style attacks into 4,471 honest customers
- Mostly **not** detected (e.g. flat-average attack: median risk 0.174 → 0.082)
- The model learned **who was caught**, not how theft works → needs rules + retraining

## 10. Explainability
- TreeSHAP: base 0.173 + Σ contributions = exact score; waterfall + what-if
- `448FE010…`: risk 0.97 (80% probability), 65% drop → 9 months low → recovery (actually theft)
- `F21D74D3…`: zero usage for 3 years → risk 0.0025 but **anomaly 0.90**: the two detectors complement each other
- Visual: Customer detail waterfall screenshot + `shap_importance.png`

## 11. Product
- 6 pages: dashboard, customer detail (reasons, waterfall, what-if, similar, PDF sheet), upload + drift check,
  **inspection planner**, cases (roles + audit timeline), **monitoring (PSI)**
- Retraining from confirmed cases with a promotion gate (demo: +80 labels → candidate worse → correctly not promoted)
- Visual: planner + case timeline screenshots

## 12. Architecture and deployment
- FastAPI + SQLAlchemy (SQLite or **PostgreSQL**) on Render (Docker) ↔ Streamlit Cloud
- MLflow tracking; GitHub Actions CI: 21 tests + Docker build
- Visual: architecture diagram (report Section 8)

## 13. Limitations and future work
- ~0.39 precision/recall; label bias; attack blindness; short histories drift; free-tier resets/cold starts; demo-grade auth
- Future: deploy the ensemble (ONNX), train with synthetic attacks + self-comparison rules, MLflow registry + scheduled retraining,
  drift alerts, real user management

## 14. Conclusion
- Explained, calibrated, ranked inspection list + full workflow, live and reproducible
- 43.5% of thieves in the top 10%; High band = 42.5% thieves on unseen demo data
- Rigour: improvements kept only when they held up under fixed rules
- Thank you · Questions?

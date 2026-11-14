# ENZERA - Slides outline (12 slides, ~10 minutes)

Every number comes from `reports/metrics.json`, `reports/experiments.csv` or `docs/explainability.md`.
Speaker notes are in *italics*.

---

## 1. Title
**ENZERA: An Explainable AI-Based System for Electricity Theft and Abnormal Consumption Detection**
- Member A name · Member B name, M.Tech Data Science, Applied Machine Learning
- Live: enzera.streamlit.app · github.com/sshah711/enzera
- Visual: app dashboard screenshot as background

## 2. The problem
- Electricity theft = non-technical losses for utilities
- Inspectors can visit only a few customers. **Who should they visit first, and why?**
- Goal: rank customers by theft risk from daily smart-meter data, and explain each flag
- *Note: the deliverable is a ranked, explained inspection list, not just a classifier.*

## 3. Data: SGCC
- 42,372 customers × 1,034 days (Jan 2014 - Oct 2016), labels from inspections
- **8.53% theft** → accuracy is meaningless (all "normal" = 91.5%)
- 25.7% readings missing; 22.3% of customers have no data before 2016
- Split 70 / 15 / 15: train / test / **demo (never seen, shown in the app)**
- Visual: `eda_class_balance.png` + `eda_missing.png`

## 4. EDA insight
- Theft customers use **more**, not less: median 8.32 vs 5.19 kWh/day, more variable, rising in 2016
- Zero-usage days rare in both classes
- Best single feature (`std_daily`) has AUC only 0.705 → need a model that combines features
- Visual: `eda_avg_consumption.png`

## 5. Method
- Pipeline: clean (interpolate, clip, 3σ cap) → **14 features** → models → SHAP → API → UI
- 14 features in 4 groups: level · variability · suspicious low usage · change over time (+ missing data, weekend ratio)
- Models: Logistic Regression, Random Forest, **XGBoost (main)**, Isolation Forest (unsupervised)
- Imbalance: `scale_pos_weight` = 10.7; threshold tuned for best F1 on **validation** (0.673)
- Visual: pipeline diagram from the About page

## 6. Results
| Model | F1 | ROC-AUC | PR-AUC | Theft in top 10% |
|---|---|---|---|---|
| Logistic Regression | 0.242 | 0.695 | 0.205 | 25.0% |
| Random Forest | 0.338 | 0.776 | 0.293 | 36.6% |
| **XGBoost** | **0.365** | **0.788** | **0.332** | **40.5%** |
| Isolation Forest | - | 0.595 | 0.115 | 16.0% |
- Headline: **inspect the top 10% → catch 40.5% of thieves (4× random)**
- Visual: `roc_pr.png`

## 7. Experiments: can we do better?
- SMOTE: worse (PR-AUC 0.296 vs 0.332)
- 3 hyperparameter settings: best ties (0.332, bootstrap CI ±0.015) → keep the simpler model
- Rule fixed before running; nothing qualified
- Takeaway: limited by features, not tuning
- Visual: `experiments.png`

## 8. Explainability
- TreeSHAP: exact contribution of each feature; base + Σ SHAP = model output
- Example `448FE010…`: risk 0.963, 65% drop July 2014 → 9 months low → recovery (actually theft)
- Global: missing readings, variability, peak usage, trend matter most
- Visual: Customer detail screenshot (risk meter + reason cards) + `shap_importance.png`

## 9. Abnormal consumption (Isolation Forest)
- Trained on normal customers only; no labels needed
- Weak for theft (AUC 0.595); uncorrelated with risk (Spearman 0.001)
- Value: second signal for unusual behaviour (faults, new theft methods)
- *Note: show honesty here; examiners like it.*

## 10. Product demo flow
- Dashboard → top customer + SHAP reasons → upload CSV → open & close case → model performance
- Stack: FastAPI + SQLite (Render, Docker) ↔ Streamlit (Streamlit Cloud); GitHub Actions CI (10 tests + Docker build)
- Visual: architecture diagram (report Section 8) + live demo

## 11. Limitations and future work
- Moderate precision/recall (~0.36); a third of demo thieves scored Low; label bias
- Handcrafted features; interpolated long gaps; SQLite resets on free hosting; cold starts
- Future: 1D-CNN/LSTM, MLflow registry, retraining from confirmed cases, drift monitoring, PostgreSQL

## 12. Conclusion
- Explained, ranked inspection list from raw meter data, live and reproducible
- XGBoost: PR-AUC 0.332, 40.5% of thieves in the top 10%; High band = 42.5% thieves on unseen demo data
- Thank you · Questions?

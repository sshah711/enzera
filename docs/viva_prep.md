# ENZERA - Viva preparation (18 questions)

Short answers grounded in our code (`ml/`, `backend/`, `frontend/`) and our generated results.

---

**1. Why XGBoost?**
Our data is tabular (14 features per customer). Gradient-boosted trees are the strongest standard choice for tabular data.
They capture non-linear effects and interactions (each feature alone is weak: best single-feature AUC 0.705), need no
scaling, train in seconds on a CPU, and give **exact** SHAP values via `pred_contribs=True`. Empirically, XGBoost beat Random
Forest and Logistic Regression on every ranking metric (PR-AUC 0.332 vs 0.293 vs 0.205).

**2. Why not accuracy?**
Only 8.53% are thieves. Predicting "normal" for everyone gives 91.5% accuracy and catches zero thieves. We use precision,
recall, F1, ROC-AUC, **PR-AUC** (random = 0.085) and **theft caught in the top 10%**, which matches how inspections are planned
(a fixed budget of visits).

**3. What is the demo split?**
We split customers 70% train / 15% test / 15% demo (stratified, seed 42). The demo customers are used for neither training nor
evaluation. 1,000 of them are loaded into the app, so every score in the demo is a genuine prediction on unseen data. Their
true labels are stored separately (`data/demo/demo_labels.csv`) and never shown in the UI.

**4. How was the threshold chosen?**
From the training split we hold out 15% as validation (4,449 customers). After fitting on the remaining 25,211, we compute
the precision-recall curve on validation and pick the threshold with the best F1 (`best_f1_threshold`): **0.673** for XGBoost.
The test set is never used for this, otherwise the test metrics would be optimistic. Risk bands: High ≥ 0.673, Medium ≥ 0.337.

**5. Why is the threshold 0.673 and not 0.5?**
`scale_pos_weight` re-balances training, so the model's probabilities are inflated relative to the true 8.5% rate (its base
value is 0.54). The F1-optimal cut-off is therefore higher than 0.5.

**6. How are the 14 features computed?**
`ml/features.py`. First `clean()` interpolates missing days, clips negatives to 0 and caps outliers at mean + 3σ. Then
`compute_features()` computes:
- level: mean, max
- variability: std, coefficient of variation, spike share, between-month CV
- suspicious low usage: zero share, longest zero run, share of days below 20% of the mean, largest month-to-month drop
- change over time: slope of monthly means, last-90-days vs earlier ratio
- weekend/weekday ratio
- missing-reading share (computed *before* cleaning)

The same function is used in training and in the backend, so there is no train/serve skew.

**7. What does TreeSHAP `pred_contribs` do?**
`booster.predict(DMatrix, pred_contribs=True)` returns, for each customer, one SHAP value per feature plus a bias column. SHAP
values come from game theory (Shapley values): each feature's fair share of the difference between this prediction and the
average. For trees, TreeSHAP computes them **exactly** and fast. They are additive in log-odds:
`risk = sigmoid(base + Σ SHAP)`. For example, customer `57531380…`: 0.159 + contributions = 4.066 → 0.983. We drop the bias
column and show the top 3 positive values as reasons (`ml/scoring.py`).

**8. What does Isolation Forest do?**
It builds random trees that split on random features at random values. Anomalies are isolated in fewer splits (shorter
average path), so they get higher scores. We fit it on **normal** training customers only and rescale scores to 0-1. It needs
no labels. Results: ROC-AUC 0.595, top-10% recall 16.0%, uncorrelated with the risk score. It is a weak theft detector but a
useful second signal for unusual behaviour.

**9. How do the backend and frontend communicate?**
Over HTTP/JSON. The Streamlit frontend calls the FastAPI backend with `requests` (`api()` in `frontend/app.py`), using the URL
in the `BACKEND_URL` secret and an optional `X-API-Key` header. The frontend never loads the model. The backend loads the model
at startup, scores the demo customers into SQLite and serves `/customers`, `/predict`, `/cases`, etc.

**10. How would retraining from confirmed cases work?**
Inspectors set a case to `theft_confirmed` or `normal` (`PATCH /cases/{id}`). These are new, high-quality labels. A retraining
job would:
1. export the cases joined with each customer's readings
2. append them to the training data
3. run `python -m ml.train`
4. compare the new model with the current one on a fixed test set
5. deploy only if PR-AUC and top-10% recall improve (ideally via an MLflow registry)

This is designed but **not implemented**; it is future work.

**11. How did you handle class imbalance? Why not SMOTE?**
`scale_pos_weight = #normal / #theft = 10.721` weights each thief 10.7× in the loss. We tested SMOTE in
`notebooks/02_experiments.ipynb`: it was worse on everything (PR-AUC 0.296 vs 0.332, validation 0.271 vs 0.338). Thieves and
normal customers overlap, so interpolated synthetic thieves land among normal customers and add noise.

**12. Did you tune hyperparameters?**
Yes, three alternatives (shallow/800 trees, deep/regularised, slow/1500 trees). Rule fixed in advance: switch only if test
PR-AUC and top-10% recall both improve by ≥ 0.01. None did. The best tied (PR-AUC -0.0009; bootstrap 95% CI [-0.015, +0.014]),
so we kept the simpler model.

**13. What surprised you in the EDA?**
Theft customers use **more** electricity (median 8.32 vs 5.19 kWh/day), are more variable and rise in 2016. Zero days are rare
for both classes. A likely reason is label bias: inspections target larger, irregular customers.

**14. How do you handle missing data?**
25.7% of readings are missing. We interpolate linearly along time and fill the rest with 0, but first record `missing_ratio`
as a feature (it has the highest mean |SHAP|, 0.485). Limitation: 22.3% of customers have no data before 2016, so interpolation
creates a long flat segment.

**15. Interpret your confusion matrix.**
Test set: TN 5,464, FP 349, FN 344, TP 199. Of 548 flagged, 199 are thieves (precision 0.363); of 543 thieves, 199 are caught
(recall 0.366). In practice we recommend ranking (top-k) rather than a hard cut-off.

**16. What are the main limitations?**
Moderate precision/recall; handcrafted features; interpolated gaps; label bias; Isolation Forest weak; SQLite resets on
Render's free plan (no persistent disk; DB in `/tmp`); cold starts; no user authentication.

**17. How is the system tested and deployed?**
10 pytest tests (features, input formats, regression for all-missing customers, API endpoints, case lifecycle). GitHub Actions
runs them plus a Docker build on every push. Backend: Docker on Render (free). Frontend: Streamlit Community Cloud with a
`BACKEND_URL` secret.

**18. How does your work relate to Zheng et al. and Jokar et al.?**
Zheng et al. (2018) released SGCC and used a Wide & Deep CNN on raw 1-D and weekly 2-D series. We use their data but
interpretable features + XGBoost for speed and exact explanations; a CNN is our future work. Jokar et al. (2016) defined theft
attack types (scaling down, zero reporting, constant values, re-ordering). Our drop, trend, zero-run and variability features
target the first three; re-ordering needs sub-daily data, which SGCC does not have.

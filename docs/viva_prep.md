# ENZERA - Viva preparation (24 questions)

Short answers grounded in our code and generated results (deployed model `xgb-20261114-1557`, 17 features).

---

## Model and evaluation

**1. Why XGBoost?**
Our data is tabular (17 features per customer). Gradient-boosted trees are the strongest standard choice for tabular data: they
capture non-linear effects and interactions (each feature is weak alone, best single-feature AUC 0.705), need no scaling, train in
seconds on a CPU, and give **exact** SHAP values via `pred_contribs=True`. Empirically, it beat Random Forest and Logistic
Regression on F1 and every ranking metric (PR-AUC 0.376 vs 0.337 vs 0.218).

**2. Why not accuracy?**
8.53% are thieves; predicting "normal" for everyone gives 91.5% accuracy and catches nobody. We use precision, recall, F1, ROC-AUC,
**PR-AUC** (random = 0.085), **theft caught in the top 10%**, and the full capture curve, because inspections are a fixed budget of
visits.

**3. What is the demo split?**
Customers are split 70% train / 15% test / 15% demo (stratified, seed 42). Demo customers are used in neither training nor evaluation.
1,000 of them are in the app, so every score shown is a genuine out-of-sample prediction. Their true labels
(`data/demo/demo_labels.csv`) are never shown in the UI. We used them only to check that 34 of the 80 High-band customers are
thieves (42.5%).

**4. How was the threshold chosen?**
15% of the training split (4,449 customers) is held out as validation. After fitting on the remaining 25,211, we pick the threshold
with the best F1 on validation (`best_f1_threshold`): **0.656**. The test set is never used for it. Bands: High ≥ 0.656,
Medium ≥ 0.328.

**5. Risk score vs estimated probability, what's the difference?**
Class weights (`scale_pos_weight` 10.7) inflate raw scores, so a 0.97 score is not a 97% chance. We fit **Platt scaling** (a logistic
regression on the logit of the score) on the validation split. The top customer's 0.97 becomes about 80%. On test, calibration
lowers the Brier score from 0.122 to 0.065. We rejected isotonic calibration because its top step claimed 100% based on only
4 validation customers. Ranking and bands use the raw score; the probability is for interpretation.

**6. How are the 17 features computed?**
`ml/features.py`. `clean_inside()` interpolates gaps *between* readings only (no filling before the first or after the last reading),
clips negatives and caps outliers at mean + 3σ. Then `compute_features()` uses NaN-aware statistics on observed days:
- level: mean, max
- variability: std, CV, spike share, between-month CV
- low usage: zero share, longest zero run, share below 20% of mean, largest monthly drop
- change over time: monthly trend slope, last-90-days vs earlier
- weekly pattern: weekend ratio, lag-7 autocorrelation, day-of-week strength
- data quality: missing share, active-reporting span

The same code runs in training and in the backend, so there is no train/serve skew.

**7. Why "gap-aware"? What did it change?**
22.3% of meters have no data before 2016. Interpolating in both directions turned 2014-2015 into a flat copy of their first value,
which distorted variability and trend. Leaving those days missing, plus two weekly features, improved test PR-AUC from 0.332 to
**0.376** (bootstrap CI [+0.017, +0.067]) and top-10% recall from 40.5% to 43.5%, and validation improved too, so it passed our rule.

**8. How did you avoid fooling yourselves with the test set?**
Every change had a rule written **before** running: ≥ +0.01 test PR-AUC **and** top-10% recall, and not worse on validation, plus
paired bootstrap intervals. That rejected SMOTE, tuning, the CNN alone and a retrained candidate, and accepted the gap-aware features.

**9. Why not SMOTE?**
Tested in notebook 02: worse on every metric (PR-AUC 0.296 vs 0.332, validation 0.271 vs 0.338). The classes overlap heavily, so
synthetic thieves interpolated between real ones fall among normal customers and add label noise. Class weights reach the same
balance without inventing data.

**10. Did deep learning help?**
A 1D-CNN on the raw 1,034-day series (2 channels: log usage + missing mask) got test PR-AUC 0.391 but was worse on validation (0.346
vs 0.379), so it was rejected alone. An **ensemble** (average of the two models' percentile ranks) reached PR-AUC **0.433** and top-10%
46.6%, and passed the rule. We did not deploy it: exact SHAP explanations exist only for the tree part, and PyTorch does not fit the free
512 MB host. It could be deployed via ONNX later.

**11. Does your model detect the attack types of Jokar et al.?**
Mostly **no**, and we report it. Notebook 04 injected six attack types into 4,471 honest test customers. Only zero periods raised scores
(in the top-10% zone: 8.1% vs 7.0% before). Reducing or flattening usage *lowered* risk (flat average: median 0.174 → 0.082). The model
learned what *caught* SGCC thieves look like (large, variable, rising usage), not manipulation mechanics. A production system needs
self-comparison rules and retraining on new outcomes.

**12. Interpret the confusion matrix.**
Test: TN 5,484, FP 329, FN 329, TP 214. Of 543 flagged, 214 are thieves (precision 0.394); of 543 thieves, 214 are caught (recall 0.394).
In practice we recommend ranking with a budget (planner) rather than a hard cut-off.

## Explainability

**13. What does TreeSHAP `pred_contribs` do?**
`booster.predict(DMatrix, pred_contribs=True)` returns, per customer, one SHAP value per feature plus a bias column. SHAP values are
Shapley values from game theory, each feature's fair share of the prediction relative to the average, and TreeSHAP computes them
**exactly** for trees. They add up in log-odds: for `448FE010…`, 0.173 + Σ contributions = 3.517 → sigmoid = 0.971, the displayed
score. A test checks this additivity.

**14. What do the waterfall and what-if show?**
The waterfall stacks all 17 contributions from the base value to the final log-odds. The what-if sends changed feature values to
`POST /whatif` and re-scores. For example, lowering the top customer's day-to-day variation from 13.86 to 1.96 kWh moves its risk from
0.97 to 0.90 (probability 80% → 55%). The model itself is never changed.

**15. What does Isolation Forest do?**
Random trees split on random features at random values; anomalies are isolated in fewer splits. It is fitted on **normal** training
customers only and scaled to 0-1. Results: ROC-AUC 0.586, top-10% 13.8%, uncorrelated with risk (Spearman −0.039). So it is weak for
ranking theft but useful for label-free anomalies. Example: `F21D74D3…` reads zero for three years, theft risk 0.0025, anomaly 0.90.

**16. How does "similar customers" work?**
Features are log-scaled (`sign(x)·log(1+|x|)`) and standardised with the training mean/std (stored in the model bundle), then the
5 nearest demo customers by Euclidean distance are returned, with their risk and monthly curves.

## Product and engineering

**17. How do the backend and frontend communicate?**
HTTP/JSON. Streamlit calls FastAPI with `requests` (`api()` in `frontend/app.py`) at `BACKEND_URL`, sending a Bearer token after
login (and an optional `X-API-Key`). The frontend never loads the model.

**18. How do login and roles work?**
`POST /auth/login` checks the username and a SHA-256 password hash, then returns a token = base64(username, role, expiry) + an HMAC-SHA256
signature with `SECRET_KEY`. Each protected endpoint verifies the signature and expiry. Inspectors open cases and add notes; managers
also set outcomes, delete cases and export labels. Viewing is public. It is demo-grade (no salts, demo accounts): production would use
bcrypt and SSO.

**19. What is the audit trail?**
The `case_events` table logs every create, status change, note and delete (who, role, when, from → to, note). History is kept even
after a case is deleted, and the UI shows it as a timeline.

**20. How does retraining from confirmed cases work?**
Managers export outcomes (`GET /labels`: 1 = theft_confirmed, 0 = normal). `python -m ml.retrain` rebuilds the same split, adds these
customers to the training data (never the test split; an inspection outcome overrides the SGCC label), retrains with the same settings,
and compares with the current model on the fixed test set. `--promote` deploys only if test PR-AUC is not worse. Demo: 80 simulated
outcomes → candidate PR-AUC 0.371 < 0.376 → correctly **not promoted** (the labels agreed with SGCC, so they added no information).

**21. What is drift monitoring and PSI?**
Training feature deciles are stored in the bundle. For new data, `PSI = Σ (new% − train%) · ln(new% / train%)` per feature: < 0.1
stable, 0.1-0.25 moderate, > 0.25 major. The demo customers are stable (max 0.021). The one-year sample upload has major shifts on 14
of 17 features, a correct warning that short histories are scored less reliably.

**22. Why SQLAlchemy? Does PostgreSQL really work?**
The same table definitions and queries run on SQLite (default) and PostgreSQL (set `DATABASE_URL`), so cases can persist on free
hosting. We ran the whole 20-test suite against a real PostgreSQL 16 server. It caught a bug: SQLAlchemy 2.1 picks the psycopg-3
driver for `postgresql://`, so we force `postgresql+psycopg2://`.

**23. What is logged in MLflow?**
Each training run logs every model's parameters and test metrics, and for XGBoost the figures, `metrics.json` and the model bundle as
artifacts (SQLite backend `mlflow.db`; view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

**24. What are the main limitations?**
Moderate precision/recall (~0.39); about 3 in 10 thieves scored Low; label bias and attack blindness; handcrafted features; short-history
drift; Isolation Forest is weak; free-tier resets (unless PostgreSQL) and cold starts; demo-grade authentication.

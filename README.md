# ENZERA ⚡
**An Explainable AI-Based System for Electricity Theft and Abnormal Consumption Detection**

ENZERA scores electricity customers for theft risk from their daily smart-meter readings, explains each
score with SHAP, and lets inspectors open and close inspection cases.

**Live demo:** https://enzera.streamlit.app · **API:** https://enzera-api.onrender.com ([interactive docs](https://enzera-api.onrender.com/docs))

> The free Render backend sleeps after 15 minutes without visitors. The first request can take 30-60 seconds while it wakes up.

[![CI](https://github.com/sshah711/enzera/actions/workflows/ci.yml/badge.svg)](https://github.com/sshah711/enzera/actions)

## How it works
```
SGCC data ──> ml/train.py ──> models/enzera_model.joblib ──> backend (FastAPI + SQLite) ──> frontend (Streamlit)
              (features, 3 models,                             /customers /predict          risk dashboard, customer
               SHAP, Isolation Forest)                         /cases /metrics              detail, upload, cases
```
- **Features:** 14 consumption features (zero-usage runs, monthly drops, trend, variability, missing data...)
- **Models:** Logistic Regression and Random Forest baselines, **XGBoost** main model (class weights for the
  8.5% theft imbalance, threshold tuned on a validation split), **Isolation Forest** for abnormal consumption
- **Explainability:** exact TreeSHAP values per customer -> top 3 reasons shown to inspectors
- **Honest demo:** data split 70% train / 15% test / 15% demo. The app only shows demo customers the model never saw.

## Results (held-out test set, 6,356 customers)
From `reports/metrics.json` (model `xgb-20261114-1143`). Thresholds tuned for best F1 on a validation split.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Theft caught in top 10% |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.163 | 0.475 | 0.242 | 0.695 | 0.205 | 25.0% |
| Random Forest | 0.324 | 0.354 | 0.338 | 0.776 | 0.293 | 36.6% |
| **XGBoost (deployed)** | **0.363** | 0.366 | **0.365** | **0.788** | **0.332** | **40.5%** |
| Isolation Forest (unsupervised) | - | - | - | 0.595 | 0.115 | 16.0% |

Random ranking would give PR-AUC ≈ 0.085 (the theft rate) and catch 10% of thieves in the top 10%.
`notebooks/02_experiments.ipynb` compares SMOTE and three other XGBoost settings on the same split. None beats
the deployed model (SMOTE: PR-AUC 0.296; best alternative ties at 0.332, within bootstrap noise), so it is kept.
EDA: `notebooks/01_eda.ipynb`. Explanations: `docs/explainability.md`.

## Run it
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

# 1. Put SGCC data.csv in data/raw/ , then train (about 2-5 minutes)
python -m ml.train

# 2. Backend      -> http://localhost:8000/docs
uvicorn backend.main:app --reload

# 3. Frontend (second terminal) -> http://localhost:8501
streamlit run frontend/app.py

# Windows: if `activate` is blocked, skip it and use the venv's Python directly:
#   venv\Scripts\python -m uvicorn backend.main:app
#   venv\Scripts\python -m streamlit run frontend/app.py
# Notebooks: pip install -r notebooks/requirements.txt

# Tests
pytest -q
```
Or with Docker: `docker compose up --build`.

## Dataset
SGCC Electricity Theft Detection: https://github.com/henryRDlab/ElectricityTheftDetection
(Kaggle single-file mirror: https://www.kaggle.com/datasets/sreen28g10/electricity-theft-detection)

Zheng et al., *Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart
Grids*, IEEE Transactions on Industrial Informatics, 2018.

## Future work
Deep learning (1D-CNN/LSTM), MLflow model registry with automated retraining from confirmed inspection
outcomes, drift monitoring, PostgreSQL in production, Jenkins pipelines.

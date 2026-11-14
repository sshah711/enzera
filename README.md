# ENZERA ⚡
**An Explainable AI-Based System for Electricity Theft and Abnormal Consumption Detection**

ENZERA scores electricity customers for theft risk from their daily smart-meter readings, explains each
score with SHAP, and lets inspectors open and close inspection cases.

**Live demo:** _add your Streamlit link here_ · **API docs:** _add your Render link_/docs

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

## Results (held-out test set)
_Paste the table from the Model performance page here after training._

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

# ENZERA ⚡
**An Explainable AI-Based System for Electricity Theft and Abnormal Consumption Detection**

ENZERA scores electricity customers for theft risk from their daily smart-meter readings, explains every score with exact SHAP
values, and supports the whole inspection workflow: planning visits, opening and closing cases (with roles and an audit trail),
retraining from confirmed outcomes, and monitoring data drift.

**Live demo:** https://enzera.streamlit.app · **API:** https://enzera-api.onrender.com ([interactive docs](https://enzera-api.onrender.com/docs))

> The free Render backend sleeps after 15 minutes without visitors. The first request can take 30-60 seconds while it wakes up.

[![CI](https://github.com/sshah711/enzera/actions/workflows/ci.yml/badge.svg)](https://github.com/sshah711/enzera/actions)

## Features
| | |
|---|---|
| **Risk model** | XGBoost on 17 gap-aware consumption features; threshold tuned on validation; Platt-calibrated probabilities |
| **Explainability** | Top SHAP reasons, full SHAP waterfall, what-if simulator, similar customers |
| **Abnormal consumption** | Isolation Forest trained on normal customers only |
| **Inspection planner** | How many thieves a given inspection budget will find (capture curve) + downloadable visit list |
| **Cases** | Inspector / manager roles, full audit trail, PDF inspection sheet |
| **Retraining** | `GET /labels` + `python -m ml.retrain`: retrain with confirmed outcomes, promote only if not worse |
| **Monitoring** | Population Stability Index per feature, for the live customers and for every upload |
| **Engineering** | FastAPI + SQLAlchemy (SQLite or PostgreSQL), Streamlit, MLflow, Docker, 20 tests + CI |

## Results (held-out test set, 6,356 customers)
From `reports/metrics.json` (model `xgb-20261114-1557`). Thresholds tuned for best F1 on a validation split.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Theft caught in top 10% |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.175 | 0.506 | 0.261 | 0.699 | 0.218 | 28.2% |
| Random Forest | 0.409 | 0.293 | 0.341 | 0.781 | 0.337 | 40.2% |
| **XGBoost (deployed)** | 0.394 | 0.394 | **0.394** | **0.798** | **0.376** | **43.5%** |
| Isolation Forest (unsupervised) | - | - | - | 0.586 | 0.109 | 13.8% |

Random ranking gives PR-AUC ≈ 0.085 and catches 10% of thieves in the top 10%. Calibration: Brier 0.122 → 0.065.

**Experiments** (each change adopted only under a rule fixed before running):
| Notebook | Question | Outcome |
|---|---|---|
| `01_eda` | What does the data look like? (train split only) | 22.3% of meters have no data before 2016; thieves use *more* |
| `02_experiments` | SMOTE? Other hyperparameters? | No: SMOTE worse (0.296), tuning ties (0.332) |
| `03_feature_experiments` | Gap-aware cleaning + weekly features? | **Yes**: PR-AUC 0.332 → 0.376 (adopted) |
| `04_synthetic_attacks` | Does the model detect injected theft attacks? | Mostly no: it learned who was *caught* (limitation) |
| `05_cnn_baseline` | 1D-CNN on raw daily data? | CNN alone: no. **XGBoost + CNN ensemble: 0.433** (not deployed, to keep exact SHAP) |

Write-ups: `docs/report.md`, `docs/explainability.md`, `docs/viva_prep.md`, `docs/slides.md`, `docs/demo_script.md`.

## Demo accounts
The app opens on a sign-in page. After signing in, the sidebar shows only the signed-in person's details
(name, role, employee ID, zone, contact). All names and details below are dummy demo data.

| Username | Password | Shown as | Role: can do |
|---|---|---|---|
| `inspector` | `inspector123` | Rahul Patel, Field Inspector (INS-1021, Ahmedabad East) | inspector: open cases, add notes |
| `inspector2` | `inspector456` | Priya Shah, Field Inspector (INS-1034, Gandhinagar) | inspector: open cases, add notes |
| `manager` | `manager123` | Anil Mehta, Revenue Protection Manager (MGR-2007, all zones) | manager: + record outcomes, delete cases, export labels |

These are demo credentials. For real use, set `USERS` and `SECRET_KEY` on the backend (see `.env.example`).

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
Windows: if `activate` is blocked, skip it and use `venv\Scripts\python -m uvicorn backend.main:app` and
`venv\Scripts\python -m streamlit run frontend/app.py`. Or with Docker: `docker compose up --build`.

**Notebooks, MLflow, retraining** (extra packages: `pip install -r notebooks/requirements.txt`; the CNN notebook also needs
`pip install torch --index-url https://download.pytorch.org/whl/cpu`):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db                                   # experiment runs -> http://localhost:5000
python -m ml.retrain --api http://localhost:8000 --user manager --password manager123   # add --promote to deploy if not worse
```

## Deployment
- **Backend (Render):** Web Service → Docker → Dockerfile `backend/Dockerfile`, build context `.`, Free plan.
  Optional environment variables: `DATABASE_URL` (PostgreSQL, e.g. free Neon/Render Postgres, so cases survive restarts),
  `SECRET_KEY` (keeps logins valid across restarts), `USERS`, `API_KEY`.
- **Frontend (Streamlit Community Cloud):** main file `frontend/app.py`, Python 3.11, secret `BACKEND_URL = "https://<app>.onrender.com"`.

## Dataset and references
SGCC Electricity Theft Detection: https://github.com/henryRDlab/ElectricityTheftDetection

- Zheng et al., *Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids*,
  IEEE Transactions on Industrial Informatics, 2018.
- Jokar et al., *Electricity Theft Detection in AMI Using Customers' Consumption Patterns*, IEEE Transactions on Smart Grid, 2016.

## Future work
Deploy the XGBoost + CNN ensemble (ONNX), train with synthetic attacks and self-comparison rules, MLflow model registry with
scheduled retraining, drift alerts, production-grade user management.

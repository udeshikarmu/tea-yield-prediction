# 🍵 Sri Lanka Tea Harvest Yield Prediction & Fertilizer Advisory System

An end-to-end Machine Learning, Spatio-Temporal Forecasting, and Agronomic Decision Support System designed for Sri Lanka's tea plantations across distinct agro-climatic zones (Up Country, Mid Country, and Low Country).

Built in full compliance with the Machine Learning Assignment Guidelines:
- **Novel Algorithm:** CatBoost Regressor (Oblivious Trees, Ordered Boosting, Native Categorical Support)
- **Explainable AI (XAI):** TreeSHAP Feature Attribution & Partial Dependence Plots (PDP)
- **Agronomic Advisory:** Tea Research Institute of Sri Lanka (TRI) Fertilizer & Nutrient Engine
- **MLOps & Tracking:** DagsHub (DVC for dataset versioning + Hosted MLflow for experiment tracking)
- **Cloud Backend:** FastAPI microservice deployed on Microsoft Azure
- **Interactive Dashboard:** Modern Next.js (App Router, TypeScript, Tailwind CSS)

---

## 📌 Project Architecture

```
User (Planter/Officer) 
        │
        ▼ [Selects Zone & Month]
Next.js Interactive Dashboard
        │
        ▼ [POST /predict-and-advise]
FastAPI Microservice (Azure App Service)
   ├── 1. CatBoost Model (Predicts Harvest Yield in kg/ha)
   ├── 2. TreeSHAP Engine (Explains top weather drivers)
   └── 3. TRI Advisory Rules (Calculates exact fertilizer mixture & bags)
```

---

## 🚀 Quick Start & Implementation

For the complete, phase-by-phase implementation instructions, see:
👉 **[STEP_BY_STEP_GUIDE.md](STEP_BY_STEP_GUIDE.md)**

### Repository Layout
```text
tea-yield-prediction/
├── data/
│   ├── raw/                # Tracked by DVC (e.g., sri_lanka_tea_yield.csv)
│   └── processed/
├── ml_service/             # Machine Learning & API
│   ├── src/
│   │   ├── preprocess.py   # Feature engineering & temporal split
│   │   ├── train.py        # CatBoost training & DagsHub MLflow logging
│   │   ├── explain.py      # SHAP & PDP visual generation
│   │   └── advisory.py     # TRI fertilizer & operational advisory logic
│   ├── app.py              # FastAPI REST API
│   ├── Dockerfile          # Container configuration for Azure
│   └── requirements.txt
├── frontend/               # Next.js Web Dashboard
│   ├── src/
│   └── package.json
├── STEP_BY_STEP_GUIDE.md   # Complete step-by-step implementation guide
└── README.md
```

# End-to-End Implementation Guide
## Predicting Monthly Tea Harvest Yields Across Agro-Climatic Zones in Sri Lanka

This project implements a complete, production-grade Machine Learning and Agronomic Decision Support System. It predicts monthly tea yields across Sri Lanka's tea-growing zones, explains the forecasts using Explainable AI (SHAP), provides official Tea Research Institute (TRI) fertilizer recommendations, and presents an interactive Next.js dashboard deployed on Azure.

---

### Project Architecture Overview

```
                      +---------------------------------------+
                      |       1. DATA & VERSIONING            |
                      |  - Local Sri Lanka Tea Dataset (.csv) |
                      |  - DVC tracking to DagsHub Storage    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       2. MACHINE LEARNING ENGINE      |
                      |  - CatBoost Regressor (Symmetric GBDT)|
                      |  - Temporal Train/Val/Test Split      |
                      |  - DagsHub Remote MLflow Logging      |
                      |  - TreeSHAP & PDP Explainability (XAI)|
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     3. TRI FERTILIZER ADVISORY        |
                      |  - Nutrient replacement formula       |
                      |  - Elevation mixtures (U709, U615)    |
                      |  - Weather safety constraint checks   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     4. REST API (FASTAPI / AZURE)     |
                      |  - POST /predict-and-advise           |
                      |  - Dockerized on Azure App Service    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     5. INTERACTIVE DASHBOARD          |
                      |  - Next.js (App Router, Tailwind CSS) |
                      |  - Zone + Month dropdown inputs       |
                      |  - Real-time forecast & SHAP charts   |
                      |  - Fertilizer & operational checklist |
                      +---------------------------------------+
```

---

## Phase 1: Environment & Directory Structure Setup

### 1.1 Create Folder Layout
In your terminal (`d:\my project\tea-yield-prediction`):

```bash
mkdir -p data/raw data/processed
mkdir -p ml_service/src ml_service/models ml_service/reports
mkdir -p frontend
```

### 1.2 Python Virtual Environment & Dependencies
Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

Create `ml_service/requirements.txt`:
```text
catboost>=1.2.0
lightgbm>=4.1.0
scikit-learn>=1.3.0
pandas>=2.1.0
numpy>=1.26.0
shap>=0.44.0
matplotlib>=3.8.0
seaborn>=0.13.0
mlflow>=2.10.0
dagshub>=0.3.0
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.6.0
joblib>=1.3.2
python-dotenv>=1.0.0
```
Install them:
```bash
pip install -r ml_service/requirements.txt
```

---

## Phase 2: DagsHub & GitHub Setup (Data & Experiment Tracking)

### 2.1 Initialize Git & DVC
```bash
git init
dvc init
```

### 2.2 Connect DagsHub
1. Go to [DagsHub](https://dagshub.com) and create a new repository: `tea-yield-prediction` (or connect your GitHub repository).
2. Configure DagsHub remote storage for DVC:
```bash
dvc remote add origin https://dagshub.com/<username>/tea-yield-prediction.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <dagshub_token>
```

---

## Phase 3: Dataset Preparation (Sri Lankan Tea & Weather Data)

Tea yield depends on elevation zones, rainfall, temperature, sunshine hours, and historical lag effects.

### 3.1 Dataset Schema
Each record in `data/raw/sri_lanka_tea_yield.csv` contains:
* `zone`: Agro-climatic zone (`Low Country Wet`, `Mid Country`, `Up Country Wet`, `Up Country Intermediate`)
* `district`: `Ratnapura`, `Galle`, `Kandy`, `Nuwara Eliya`, `Badulla`
* `elevation_m`: Elevation in meters (e.g., 100m, 650m, 1800m, 1200m)
* `year`, `month`: 2015 to 2024
* `rainfall_mm`: Total monthly precipitation
* `rainy_days`: Number of rainy days in the month
* `avg_temp_c`: Mean monthly temperature
* `sunshine_hours`: Daily average sunshine hours
* `relative_humidity`: Monthly average relative humidity (%)
* `yield_kg_per_ha`: Monthly made tea yield per hectare (Target Variable)

### 3.2 Track Data with DVC
```bash
dvc add data/raw/sri_lanka_tea_yield.csv
git add data/raw/sri_lanka_tea_yield.csv.dvc .gitignore
git commit -m "Track tea dataset using DVC"
dvc push -r origin
```

---

## Phase 4: Feature Engineering & Preprocessing

Tea shoots (*two leaves and a bud*) take 35 to 60 days to flush after rain. Lagged weather features are essential.

Create `ml_service/src/preprocess.py`:
* **Lag Features:** `rainfall_lag_1m`, `rainfall_lag_2m`, `temp_lag_1m`
* **Rolling Drought Indicators:** `rolling_3m_rainfall` (rolling sum over 3 months)
* **Cyclical Month Encoding:** `month_sin = sin(2 * pi * month / 12)`, `month_cos = cos(2 * pi * month / 12)`
* **Extreme Weather Flags:** `is_drought = rainfall_mm < 60`, `is_excess_rain = rainfall_mm > 350`
* **Temporal Splitting (No Data Leakage):**
  * Train Set: 2015 – 2022
  * Validation Set: 2023
  * Test Set: 2024

---

## Phase 5: CatBoost Model Training & MLflow Tracking

### Why CatBoost for this Assignment?
1. **Oblivious (Symmetric) Trees:** Regularizes against overfitting on regional agricultural time-series.
2. **Native Categorical Handling:** Encodes zones without artificial ordering or one-hot sparsity.
3. **Ordered Boosting:** Solves target leakage that affects traditional Gradient Boosted Trees.

Create `ml_service/src/train.py`:
* Connects to DagsHub MLflow server:
  `mlflow.set_tracking_uri("https://dagshub.com/<username>/tea-yield-prediction.mlflow")`
* Logs parameters: `depth`, `learning_rate`, `iterations`, `l2_leaf_reg`.
* Trains CatBoost Regressor.
* Evaluates on Test Set: logs **RMSE**, **MAE**, and **R² Score**.
* Saves model artifact to `ml_service/models/catboost_tea_model.cbm`.

---

## Phase 6: Explainable AI (XAI) with SHAP

Create `ml_service/src/explain.py`:
* **TreeSHAP:** Computes exact Shapley values.
* **SHAP Summary Beeswarm Plot:** Visualizes which climate features push yield up or down across zones.
* **SHAP Waterfall Plot:** Explains individual monthly predictions for specific estates.
* **Partial Dependence Plots (PDP):** Shows the non-linear relationship between temperature/rainfall and tea yield.
* **Agronomic Validation:** Compares SHAP feature importance with Tea Research Institute (TRI) shoot growth literature.

---

## Phase 7: TRI Fertilizer & Operational Advisory Engine

Create `ml_service/src/advisory.py`:
* **Mixture Selection:**
  * Low Country -> `TRI Mixture U709` or `U834` (high Nitrogen for rapid foliage regeneration).
  * Mid Country / Up Country -> `TRI Mixture U615` (balanced NPK + Mg).
* **Dosage Calculation:**
  $$\text{Dosage (kg/ha)} = \frac{\text{Predicted Yield (kg/ha)}}{100} \times 9.5$$
  Calculates number of standard 50kg bags needed.
* **Weather Safety Rules:**
  * Rainfall < 60mm: **Do Not Apply** (drought scorch risk).
  * Rainfall > 350mm: **Do Not Apply** (monsoon runoff / leaching risk).
  * 100mm - 250mm: **Optimal Window** (ideal nutrient absorption).
* **Operational Checklist:**
  * Plucking round frequency (6–7 days during high flush, 9–10 days during low flush).
  * Estimated pluckers required per hectare.

---

## Phase 8: FastAPI Backend Service

Create `ml_service/app.py`:
* Exposes `POST /predict-and-advise`:
  * **Inputs:** `zone`, `month`, optional custom rainfall/temperature sliders.
  * **Processes:** Fetches climate profile, constructs features, predicts yield with CatBoost, computes SHAP impact, and runs advisory rules.
  * **Outputs:** Structured JSON containing:
    1. Predicted Yield (kg/ha) + % difference from 5-year average.
    2. Top 3 climate drivers (XAI).
    3. Recommended fertilizer mixture, dosage (bags), and safety status.
    4. Operational labour and plucking interval advisory.

Create `ml_service/Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Phase 9: Azure Deployment

1. **Push Container to Azure Container Registry (ACR)**:
   ```bash
   az acr build --registry <your_acr_name> --image tea-yield-api:v1 ./ml_service
   ```
2. **Deploy to Azure App Service / Container App**:
   ```bash
   az webapp create --resource-group tea-rg --plan tea-plan --name tea-yield-api --deployment-container-image-name <your_acr_name>.azurecr.io/tea-yield-api:v1
   ```
   Backend URL: `https://tea-yield-api.azurewebsites.net`

---

## Phase 10: Next.js Frontend Dashboard

Inside `frontend/`:
1. Initialize Next.js with Tailwind CSS:
   ```bash
   npx create-next-app@latest ./ --typescript --tailwind --eslint --app
   ```
2. Build UI Components:
   * `ZoneSelector.tsx`: Dropdown for Sri Lanka's tea zones.
   * `MonthSelector.tsx`: Dropdown for months (January to December).
   * `YieldCard.tsx`: Metric card with predicted yield, confidence intervals, and flush category.
   * `XaiInsightCard.tsx`: Visual breakdown of climate drivers.
   * `FertilizerCard.tsx`: Official TRI mixture badge, dosage in bags, and weather safety status.
   * `OperationsCard.tsx`: Plucking interval and labor allocations.
   * `ClimateSimulationModal.tsx`: Sliders to simulate rainfall scenarios ("What if rainfall drops by 25%?").

---

## Assignment Deliverables Checklist

| Deliverable | Description | Location |
| :--- | :--- | :--- |
| **Written Report (PDF)** | Problem definition, CatBoost justification, evaluation tables, SHAP & PDP graphs, critical discussion | `reports/Final_Report.pdf` |
| **Source Code (GitHub)** | Clean modular code: `preprocess.py`, `train.py`, `explain.py`, `advisory.py`, `app.py`, `frontend/` | GitHub Repository |
| **DagsHub Tracking** | DVC data tracking link & MLflow experiment comparison table | DagsHub Project Link |
| **Demo Video (3–5 mins)**| Walkthrough: Problem overview -> DagsHub MLflow -> Live Next.js Web App predicting yield and fertilizer | Video link / file |

# An Integrated Agro-Climatic Intelligence System for Sri Lankan Tea Plantations
### Combining Harvest Readiness Classification and Yield Estimation with Explainable AI (XAI) and Closed-Loop MLOps

---

## Introduction

Tea is one of Sri Lanka's most important export crops, and its productivity depends heavily on local climate. Rainfall, temperature, humidity, soil moisture and canopy health all influence when a field is ready to be plucked and how much green leaf it will produce. These conditions differ sharply between the cool, misty high-grown estates of Nuwara Eliya and the warm, humid low-grown estates of Galle and Matara, so a one-size-fits-all rule of thumb rarely works.

This project builds an **agro-climatic decision-support system** that helps plantation managers answer two practical questions:

1. **Is this field ready for harvest?** A binary classification of harvest readiness, with a confidence probability.
2. **How much yield can we expect?** An estimate of green leaf yield in kilograms per hectare.

Every prediction comes with an **explanation**. SHAP (SHapley Additive exPlanations) shows which climatic factors pushed the decision towards "Ready" or "Not Ready", so the system is a transparent advisor that agronomists can check against their own field knowledge rather than a black box.

The model sits inside a complete **MLOps lifecycle**. Data is versioned, experiments are tracked, models must pass a quality gate before release, deployment to the cloud is automated, and production data is continuously monitored for drift. When the climate shifts, retraining is triggered automatically.

---

## 1. System Architecture Overview

The system is designed for tea plantations across **11 agro-ecological regions** of Sri Lanka, covering all three elevation categories:

| Elevation Zone | Example Regions |
| :--- | :--- |
| **High-grown** | Nuwara Eliya, Dimbula, Haputale, Uda Pussellawa |
| **Mid-grown** | Kandy, Badulla, Ella |
| **Low-grown** | Galle, Ratnapura, Matara, Kegalle |

The diagram below shows how code, data and models move from a developer's machine to production, and how production monitoring feeds back into training.

```
   +-----------------------------------------------------------------------------------+
   |                                1. Developer Workstation                           |
   |   (Local / GitHub Codespaces: Data Contract, Preprocessing, Unit Tests, DVC Push) |
   +-----------------------------------------+-----------------------------------------+
                                             |  git push / pull request
                                             v
   +-----------------------------------------------------------------------------------+
   |                               2. GitHub CI Pipeline                               |
   |              (GitHub Actions: Tests -> CatBoost Train -> Quality Gate)            |
   +--------------------+------------------------------------+-------------------------+
                        |                                    |
     DVC Data & MLflow  |                                    | Promoted upon passing
     Tracking/Registry  v                                    v F1 >= 0.85 & AUC >= 0.88
          +-----------------------+               +-----------------------+
          |        DagsHub        |               |    Build & Publish    |
          |  MLflow Experiments   |               | Docker Image -> GHCR  |
          |   & DVC Remote Data   |               +-----------+-----------+
          +-----------------------+                           |
                                                              | OIDC Keyless Auth + Human Approval
                                                              v
   +-----------------------------------------------------------------------------------+
   |                           3. Production Cloud Deployment                          |
   |           Azure Container Apps (FastAPI Model Service + SHAP Explainer)           |
   |                                         ^                                         |
   |                         REST API Calls  |  Telemetry Ingestion                    |
   |                                         v                                         |
   |     +-------------------------+                   +-------------------------+     |
   |     |    Next.js Web UI       |                   |  App Insights / Azure   |     |
   |     | (Interactive Dashboard) |                   |      Log Analytics      |     |
   |     +-------------------------+                   +------------+------------+     |
   +----------------------------------------------------------------|------------------+
                                                                    |
                                                                    v
   +-----------------------------------------------------------------------------------+
   |                     4. Continuous Monitoring & Closed-Loop Retrain                |
   |           Evidently AI Drift Detection (KS-Test & Population Stability Index)     |
   |                                         |                                         |
   |                           Drift Flagged | (p < 0.05 / PSI > 0.25)                 |
   |                                         v                                         |
   |             Automated Retraining GitHub Action -> Opens New Pull Request          |
   +-----------------------------------------------------------------------------------+
```

### How the stages fit together

**Stage 1: Developer Workstation.** Development happens locally or in GitHub Codespaces. Incoming data is validated against a strict **Pydantic data contract** ([src/data_contract.py](src/data_contract.py)) that defines the allowed regions, zones and seasons and a valid range for every numeric feature (for example, rainfall 0–1200 mm, temperature 4–42 °C, NDVI 0–1). Records that break the contract are rejected before they can corrupt training or produce misleading predictions. Raw and processed datasets are versioned with **DVC**.

**Stage 2: Continuous Integration.** Every push or pull request starts a GitHub Actions workflow ([.github/workflows/ci.yml](.github/workflows/ci.yml)) that runs the test suite, retrains the CatBoost model and runs the **quality gate**. The gate only lets a model through if it reaches **F1 ≥ 0.85** and **ROC-AUC ≥ 0.88** on held-out data. Experiment runs, parameters and metrics are logged to **MLflow on DagsHub**, which also hosts the DVC remote storage.

**Stage 3: Continuous Deployment.** Once a model passes the gate, the CD workflow ([.github/workflows/cd.yml](.github/workflows/cd.yml)) packages the FastAPI service into a Docker image, publishes it to the **GitHub Container Registry (GHCR)** and deploys it to **Azure Container Apps**. Authentication uses **OpenID Connect (OIDC)**, so no long-lived cloud passwords are stored in the repository. The workflow runs in a protected `production` environment, so a human reviewer must approve each production release. The running service sends telemetry to **Azure Application Insights / Log Analytics**.

**Stage 4: Monitoring and closed-loop retraining.** A scheduled workflow ([.github/workflows/monitor.yml](.github/workflows/monitor.yml), weekly on Sundays) compares recent production inputs with the training distribution using **Evidently AI**. Two statistical tests are used:
- **Kolmogorov–Smirnov (KS) test**: flags a feature as drifted when p < 0.05.
- **Population Stability Index (PSI)**: flags a significant population shift when PSI > 0.25.

When drift is detected, the retraining workflow ([.github/workflows/retrain.yml](.github/workflows/retrain.yml)) retrains the model on updated data and **opens a new pull request**, so the refreshed model goes through the same review, testing and quality gate as any other change.

### Project structure

```
tea-yield-prediction/
├── app/                 # FastAPI model service (main.py), telemetry and Dockerfile
├── config/config.yaml   # Central configuration: features, temporal splits, model hyperparameters
├── data/
│   ├── raw/             # Original TRI dataset (DVC-tracked)
│   └── processed/       # Cleaned, split train/validation/test data
├── frontend/            # Next.js + Tailwind CSS interactive web dashboard
├── models/              # Serialized champion and baseline models
├── monitoring/          # Evidently AI drift detection script
├── reports/             # Generated figures and metric files
├── src/
│   ├── data_contract.py # Pydantic schema for input validation
│   ├── preprocess.py    # Cleaning, feature preparation and temporal splitting
│   ├── train.py         # CatBoost and baseline training, evaluation and SHAP plots
│   └── quality_gate.py  # Production threshold enforcement
├── tests/               # Unit and integration tests (data contract, inference, API)
├── streamlit_app.py     # Standalone Streamlit dashboard with live SHAP explanations
└── docker-compose.yml   # Local multi-container setup (API + frontend)
```

---

## 2. Dataset and Features

The model is trained on a real-world dataset, `sri_lanka_tea_tri.gov_ds.csv`, which contains **3,168 records covering 2001 to 2024**, based on agro-climatic data associated with the **Tea Research Institute (TRI) of Sri Lanka**.

| Feature | Type | Description |
| :--- | :--- | :--- |
| `region` | Categorical | One of the 11 tea-growing districts |
| `zone` | Categorical | Elevation category: high-grown, mid-grown or low-grown |
| `season` | Categorical | Monsoon season: **Maha** (northeast monsoon) or **Yala** (southwest monsoon) |
| `elevation_m` | Numeric | Elevation above sea level (50–2600 m) |
| `rainfall_mm` | Numeric | Monthly precipitation in millimetres |
| `temperature_c` | Numeric | Mean monthly temperature in °C |
| `humidity_percent` | Numeric | Mean relative humidity |
| `soil_moisture_percent` | Numeric | Volumetric soil moisture |
| `ndvi_index` | Numeric | Normalized Difference Vegetation Index, a satellite measure of canopy greenness |
| `month` | Numeric | Month of observation (1–12) |

**Target variable:** `harvest_ready`, where 1 means the field is ready for harvest and 0 means it is not.

### Temporal data split

A random split would let the model "see the future": it could train on 2023 data and then be tested on 2015 data. To avoid this **lookahead leakage**, the data is split chronologically:

| Split | Years | Purpose |
| :--- | :--- | :--- |
| Training | 2001 – 2018 | Learning model parameters |
| Validation | 2019 – 2021 | Early stopping and model selection |
| Test | 2022 – 2024 | Final, unseen evaluation |

This mirrors how the model is actually used: it learns from the past and predicts future seasons.

---

## 3. Assignment Rubric Compliance (100 Marks + 10 Bonus Marks)

The table below maps each assessment criterion to where it is implemented in this codebase.

| Rubric Section | Marks | Implementation in this Codebase |
| :--- | :---: | :--- |
| **1. Problem Definition & Dataset Collection** | **15** | Addresses agro-climatic challenges for Sri Lankan tea across 11 regions. Uses a real dataset (`sri_lanka_tea_tri.gov_ds.csv`, 3,168 records, 2001–2024). Agricultural data is handled ethically and validated with a Pydantic data contract (`src/data_contract.py`). |
| **2. Selection of a "New" ML Algorithm** | **15** | **CatBoost (Categorical Boosting)**, which was not covered in lectures. It uses symmetric (oblivious) decision trees and **ordered boosting**, which reduces target leakage and overfitting on tabular climate data, and it handles categorical features such as region and season natively without one-hot encoding. |
| **3. Model Training & Evaluation** | **20** | A **temporal train/validation/test split** prevents lookahead leakage. CatBoost is compared with **Decision Tree** and **Logistic Regression** baselines and reaches **0.9903 F1** and **0.9998 ROC-AUC** on the unseen 2022–2024 test set. |
| **4. Explainability & Interpretation (XAI)** | **20** | **SHAP TreeExplainer** produces global beeswarm plots (`reports/shap_summary.png`) and local, per-prediction feature attributions. These explanations are checked against Tea Research Institute (TRI) agronomic guidance to confirm the model has learned sensible relationships. |
| **5. Critical Discussion** | **10** | Discusses the limits of monthly temporal aggregation, regional agro-ecological bias, gaps in satellite NDVI caused by cloud cover, and the socioeconomic impact on plantation superintendents and workers. |
| **6. Report Quality & Technical Clarity** | **10** | Publication-quality ROC curves, confusion matrices, SHAP plots and clear architecture documentation. |
| **7. Bonus: Front-End Integration** | **10** | A **Next.js web application**, a built-in FastAPI dashboard and a **Streamlit** app, all offering interactive predictions with real-time SHAP attribution bars and drift simulation. |

### Why CatBoost?

CatBoost was chosen as the champion algorithm for three reasons:

- **Native categorical handling.** Region, zone and season carry strong agronomic meaning. CatBoost encodes them with ordered target statistics instead of sparse one-hot vectors.
- **Ordered boosting.** Standard gradient boosting can leak target information when it computes residuals. CatBoost's ordered scheme reduces this, which matters on a modest-sized dataset.
- **Robust defaults.** Symmetric trees act as a form of regularisation and give fast, stable inference in production.

The configured hyperparameters (see [config/config.yaml](config/config.yaml)) are 600 iterations, learning rate 0.04, tree depth 6, Logloss as the loss function, F1 as the evaluation metric and early stopping after 50 rounds without improvement.

---

## 4. Quickstart: Running the Project Locally

Run these steps in order from the project root. Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

### Step 1: Preprocess the data and train the models

`preprocess.py` cleans the raw dataset and creates the temporal splits. `train.py` then trains the champion CatBoost model and both baseline models, evaluates them, and saves the ROC curve, confusion matrix and SHAP plots to `reports/`.

```bash
python src/preprocess.py
python src/train.py
```

### Step 2: Check the quality gate

This step confirms that the trained champion model meets the production thresholds (**F1 ≥ 0.85** and **ROC-AUC ≥ 0.88**). If it does not, the script exits with an error, which is also what blocks a weak model from being deployed in CI.

```bash
python src/quality_gate.py
```

### Step 3: Run the test suite

This runs 9 unit and integration tests. They check that the data contract rejects invalid inputs, that the trained model returns well-formed predictions, and that the API endpoints respond correctly.

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Step 4: Start the FastAPI backend and built-in dashboard

Start the production server:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **`http://localhost:8000`** in your browser.

The **Agro-Climatic Intelligence Dashboard** lets you:
* Adjust sliders for rainfall, temperature, soil moisture and NDVI to describe field conditions.
* Pick a region (for example, high-grown *Nuwara Eliya* or low-grown *Galle*) and compare how the same weather affects different elevations.
* See the **harvest readiness classification**, its **probability** and the **estimated yield** (kg/ha) update in real time.
* Read the live **SHAP feature attribution bars**, which show which factors pushed the prediction up or down and give an agronomic reason for each decision.

#### API endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the built-in interactive dashboard |
| `GET` | `/health` | Health check used by the container platform |
| `GET` | `/metrics` | Returns the model's evaluation metrics |
| `GET` | `/explain/global` | Returns global SHAP feature importance |
| `POST` | `/predict` | Returns harvest readiness, probability, estimated yield and local SHAP contributions for one set of field conditions |
| `POST` | `/simulate-drift` | Injects shifted climate data to demonstrate drift detection |

Interactive Swagger documentation is generated automatically at **`http://localhost:8000/docs`**.

---

## 5. Front-End Applications (Bonus)

### Next.js web application

The Next.js application in `frontend/` (styled with Tailwind CSS) gives the system a modern, responsive interface:

```bash
cd frontend
npm install
npm run dev
```

Open **`http://localhost:3000`** in your browser. The frontend sends its requests to the FastAPI backend on port 8000, so start the backend (Step 4 above) first.

### Streamlit dashboard

`streamlit_app.py` is a standalone dashboard that loads the trained model directly and computes SHAP explanations live, without needing the API server:

```bash
streamlit run streamlit_app.py
```

---

## 6. MLOps CI/CD and Cloud Setup (Azure and DagsHub)

### DagsHub: MLflow tracking and DVC data storage

DagsHub hosts the experiment tracking server (MLflow) and the remote storage for versioned datasets (DVC), so every model can be traced back to the exact data and parameters that produced it.

1. Create a free repository on [DagsHub.com](https://dagshub.com) named `tea-harvest-mlops`.
2. Connect your local Git and DVC setup to it:
   ```bash
   dvc remote add origin https://dagshub.com/<username>/tea-harvest-mlops.dvc
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user <username>
   dvc remote modify origin --local password <dagshub_token>
   dvc push
   ```
   The `--local` flag stores your credentials in `.dvc/config.local`, which is not committed to Git.
3. Add these secrets to your GitHub repository (**Settings → Secrets and variables → Actions**) so CI can log experiments:
   * `DAGSHUB_USERNAME`
   * `DAGSHUB_TOKEN`

### Azure Container Apps: keyless OIDC deployment

Deployment uses OpenID Connect federation. GitHub Actions proves its identity to Azure with a short-lived token, so no Azure passwords or keys are stored anywhere.

1. In Azure, create an **App Registration** and add a **Federated Credential** that trusts your GitHub repository (`main` branch).
2. Add the following GitHub Actions secrets:
   * `AZURE_CLIENT_ID`: the App Registration's client ID
   * `AZURE_TENANT_ID`: your Azure Active Directory tenant ID
   * `AZURE_SUBSCRIPTION_ID`: the subscription that hosts the Container App
   * `GHCR_USERNAME`: the GitHub username that owns a package-read token
   * `GHCR_TOKEN`: a GitHub token with read-only access to the container package
3. Pushing to `main` then builds the Docker image, publishes it to **GHCR** and, after approval, deploys it to **Azure Container Apps** using **canary releases**. A small share of traffic goes to the new version first, so problems can be caught before every user is affected.

### Drift monitoring and closed-loop retraining

Climate patterns change over time, through shifting monsoons, unusual dry spells or long-term warming. A model trained on historical data can quietly lose accuracy as conditions move away from what it learned. The monitoring component detects this automatically.

To run a drift check manually:

```bash
python monitoring/run_drift.py
```

**Incident drill (end-to-end demonstration):**
1. Click **"Simulate Drift Drill"** on the dashboard, or send a `POST` request to `/simulate-drift`. This injects climate readings that are deliberately shifted from the training distribution.
2. Run `python monitoring/run_drift.py`.
3. The script detects the agro-climatic shift (KS-test p < 0.05), triggers the automated retraining workflow and **opens a new GitHub pull request** with the retrained model for review.

---

## 7. Generated Report Artifacts

All publication-quality figures and metric files are written to `reports/` when `src/train.py` runs:

| Artifact | Description |
| :--- | :--- |
| `reports/roc_curve.png` | ROC curves comparing all three models. CatBoost reaches **AUC 0.9998**, against **0.9871** for the Decision Tree and **0.6679** for Logistic Regression. The weak Logistic Regression result shows that harvest readiness depends on non-linear interactions between climate variables. |
| `reports/confusion_matrix.png` | Normalized confusion matrix on the unseen test records (2022–2024), showing correct and incorrect classifications for each class. |
| `reports/shap_summary.png` | Global SHAP beeswarm plot. It ranks features by overall importance and shows, for each feature, whether high or low values push predictions towards "Ready" or "Not Ready". |
| `reports/metrics_comparison.json` | Full comparison of Accuracy, Precision, Recall, F1 and ROC-AUC for all models. |

---

## 8. Limitations and Future Work

- **Temporal granularity:** The features are monthly aggregates, so short weather events such as a week of heavy rain or a brief drought may be smoothed out. Daily or weekly data would capture them better.
- **Regional bias:** Some regions have more records than others, so predictions for under-represented districts may be less reliable.
- **Satellite data gaps:** NDVI readings can be affected by cloud cover, which is common in Sri Lanka's monsoon seasons.
- **Yield estimation:** Yield is calculated with a rule-based agronomic formula. It starts from a base yield for each elevation zone and adjusts it for NDVI, rainfall, soil moisture and harvest readiness. A regression model trained on measured estate yields would make it more precise.
- **Human-in-the-loop:** The system supports estate superintendents and agronomists in their decisions. It does not replace them.

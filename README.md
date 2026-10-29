# An Integrated Agro-Climatic Intelligence System for Sri Lankan Tea Plantations
### Combining Harvest Readiness Classification and Yield Estimation with Explainable AI (XAI) and Closed-Loop MLOps

---

## 1. System Architecture Overview

This project implements an end-to-end, production-grade Machine Learning Operations (MLOps) architecture designed for Sri Lankan tea plantations across 11 agro-ecological regions (spanning High-Grown, Mid-Grown, and Low-Grown elevations).

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

---

## 2. Assignment Rubric Compliance (100 Marks + 10 Bonus Marks)

| Rubric Section | Marks | Implementation in this Codebase |
| :--- | :---: | :--- |
| **1. Problem Definition & Dataset Collection** | **15** | Sri Lankan tea agro-climatic challenges across 11 regions. Real dataset (`sri_lanka_tea_tri.gov_ds.csv`, 3,168 records, 2001–2024). Ethical agricultural data with Pydantic contract (`src/data_contract.py`). |
| **2. Selection of a "New" ML Algorithm** | **15** | **CatBoost (Categorical Boosting)**. Uses symmetric oblivious trees and ordered boosting to prevent target leakage on tabular time-series climate data. Distinct from lecture models. |
| **3. Model Training & Evaluation** | **20** | **Temporal train/validation/test split** (prevents lookahead leakage). Evaluated against **Decision Tree** and **Logistic Regression** baselines. CatBoost achieved **0.9903 F1** and **0.9998 ROC-AUC**. |
| **4. Explainability & Interpretation (XAI)** | **20** | **SHAP TreeExplainer** generates global summary beeswarm plots and local waterfall feature attributions (`reports/shap_summary.png`), validated against Tea Research Institute (TRI) agronomy. |
| **5. Critical Discussion** | **10** | Detailed analysis of temporal aggregation limits, regional agro-ecological biases, sensor cloud cover issues, and socioeconomic impact for plantation superintendents. |
| **6. Report Quality & Technical Clarity** | **10** | Publication-grade ROC curves, confusion matrices, and clear architecture documentation. |
| **7. Bonus: Front-End Integration** | **10** | Modern **Next.js Web Application** + FastAPI interactive dashboard with real-time SHAP feature attribution bars and drift simulation. |

---

## 3. Quickstart: How to Run Locally Right Now

### Step 1: Run the Preprocessing & Model Training
The training script trains the champion CatBoost model and baselines, generates ROC curves, confusion matrices, and SHAP plots:
```bash
python src/preprocess.py
python src/train.py
```

### Step 2: Verify the Quality Gate
Ensures the champion model satisfies production thresholds (F1 >= 0.85, ROC-AUC >= 0.88):
```bash
python src/quality_gate.py
```

### Step 3: Run the Test Suite
Executes 9 unit and integration tests across data contracts, model predictions, and API endpoints:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Step 4: Launch the FastAPI Backend & Built-In Web Dashboard
Start the production server:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

You will see the interactive **Agro-Climatic Intelligence Dashboard** where you can:
* Adjust rainfall, temperature, soil moisture, and NDVI sliders.
* Select regions (e.g. *Nuwara Eliya* vs *Galle*).
* View real-time **Harvest Readiness Classification**, **Probability**, and **Estimated Yield**.
* Inspect live **SHAP feature attribution bars** explaining the agronomic rationale behind the decision.

---

## 4. Running the Next.js Frontend (Bonus Web UI)

The dedicated Next.js application is located in `frontend/`:
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:3000`** in your browser. The Next.js frontend connects directly to the FastAPI backend running on port 8000.

---

## 5. MLOps CI/CD & Cloud Setup (Azure & DagsHub)

### DagsHub (MLflow Tracking & DVC Data Storage)
1. Create a free repo on [DagsHub.com](https://dagshub.com) named `tea-harvest-mlops`.
2. Connect your local Git and DVC:
   ```bash
   dvc remote add origin https://dagshub.com/<username>/tea-harvest-mlops.dvc
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user <username>
   dvc remote modify origin --local password <dagshub_token>
   dvc push
   ```
3. Set GitHub Secrets:
   * `DAGSHUB_USERNAME`
   * `DAGSHUB_TOKEN`

### Azure Container Apps (Keyless OIDC Deployment)
1. Configure an Azure App Registration with Federated Credentials pointing to your GitHub repository (`main` branch).
2. Store the following GitHub Actions secrets:
   * `AZURE_CLIENT_ID`
   * `AZURE_TENANT_ID`
   * `AZURE_SUBSCRIPTION_ID`
   * `GHCR_USERNAME`
   * `GHCR_TOKEN` (classic PAT with `read:packages` and `write:packages`)
3. Pushing code to `main` will build the Docker container, push to **GitHub Container Registry (GHCR)**, and deploy keylessly to Azure Container Apps with canary releases.

### Drift Monitoring & Closed-Loop Retraining
Run the drift check:
```bash
python monitoring/run_drift.py
```
To run the incident drill:
1. Click **"Simulate Drift Drill"** on the dashboard (or send a POST to `/simulate-drift`).
2. Run `python monitoring/run_drift.py`.
3. The script detects the agro-climatic shift (p < 0.05) and triggers automated retraining, automatically creating a new GitHub Pull Request.

---

## 6. Generated Report Artifacts
All publication-quality figures are saved in `reports/`:
* `reports/roc_curve.png`: Comparative ROC curve (CatBoost AUC: 0.9998 vs Decision Tree: 0.9871 vs Logistic Regression: 0.6679).
* `reports/confusion_matrix.png`: Normalized confusion matrix on unseen test records (2022–2024).
* `reports/shap_summary.png`: Global SHAP beeswarm plot illustrating feature importance and impact direction.
* `reports/metrics_comparison.json`: Complete comparison table across Accuracy, Precision, Recall, F1, and AUC.

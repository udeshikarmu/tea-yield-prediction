# Machine Learning Assignment: Final Written Report

**Project Title:** Spatio-Temporal Tea Harvest Yield Forecasting and Agronomic Nutrient Advisory Across Agro-Climatic Zones in Sri Lanka  
**Author:** [Your Name / Student ID]  
**Algorithm Chosen:** CatBoost Regressor (Oblivious Trees, Ordered Boosting, Native Target Statistics)  
**Explainability Technique:** TreeSHAP (SHapley Additive exPlanations) & Feature Attribution  
**Front-End Integration:** Interactive Web Dashboard (FastAPI / Next.js)  

---

## 1. Problem Definition & Dataset Collection (15 Marks)

### 1.1 Problem Description & Real-World Relevance
Tea (*Camellia sinensis*) is a foundational pillar of Sri Lanka’s agricultural economy, accounting for over $1.3 billion in foreign export revenue. Unlike temperate crops, tea is perennial and rain-fed, making harvest volume intensely sensitive to micro-climatic oscillations. 

Sri Lanka’s tea cultivation is geographically partitioned into three distinct elevations:
1. **Low Country (< 600m):** Generates >60% of national volume, dominated by smallholders, sensitive to drought and heat stress.
2. **Mid Country (600m - 1200m):** Moderate growth, balancing volume and leaf density.
3. **Up Country (> 1200m):** High-elevation microclimates producing premium aromatic leaves (e.g., Nuwara Eliya, Dimbula, Uva), constrained by cold night temperatures (<13°C) and monsoon cloud cover.

Climate change has triggered unseasonal rainfall variability, sudden dry spells, and delayed monsoons (Yala and Maha). Accurate monthly yield forecasting enables Regional Plantation Companies (RPCs) and factory managers to plan plucking intervals, labour allocation, and factory intake schedules weeks in advance.

### 1.2 Data Sourcing & Compilation
* **Harvest Yield Data:** Compiled from Sri Lanka Tea Board (SLTB) Annual Statistical Bulletins and Central Bank of Sri Lanka (CBSL) agricultural statistics.
* **Agrometeorological Data:** Compiled from the Sri Lanka Department of Meteorology and NASA POWER Agroclimatology API for district coordinates (Ratnapura, Galle, Kandy, Nuwara Eliya, Badulla).
* **Sample Size:** 600 monthly historical records spanning 10 continuous years (2015–2024) across 5 primary tea-producing districts.
* **Ethical Data Considerations:** The dataset comprises aggregated regional agricultural and climatic statistics without individual farmer identities or private proprietary telemetry, complying with ethical data use standards.

### 1.3 Feature Engineering & Preprocessing
To mirror biological flush dynamics documented by the Tea Research Institute (TRI):
1. **Rainfall Lag Features (`rainfall_lag_1m`, `rainfall_lag_2m`):** A tea shoot ("two leaves and a bud") requires 35–60 days to flush following moisture stimulation.
2. **Rolling Drought Buffer (`rolling_3m_rainfall`):** 3-month cumulative rainfall to capture persistent drought stress.
3. **Diurnal Temperature Range (`diurnal_temp_range`):** Difference between daily maximum and minimum temperatures.
4. **Cyclical Encodings:** $\sin$ and $\cos$ transformations for month (1–12) to preserve annual periodicity.
5. **Temporal Splitting (Avoiding Look-ahead Leakage):**
   * **Train:** 2015–2022 (480 records)
   * **Validation:** 2023 (60 records)
   * **Test (Unseen Holdout):** 2024 (60 records)

---

## 2. Selection of a New Machine Learning Algorithm (15 Marks)

### 2.1 Algorithm Selected: CatBoost Regressor
**Why this algorithm is "new" and not covered in lectures:** Traditional lectures cover standard Linear Models, standard Decision Trees, Random Forests, and k-NN. CatBoost (Categorical Boosting) is an advanced gradient-boosted decision tree library specifically selected to address the core challenges of regional agricultural panel data.

### 2.2 Theoretical Justification & Differences from Standard Models

| Dimension | Standard Models (Decision Trees / Random Forest) | CatBoost Regressor |
| :--- | :--- | :--- |
| **Tree Architecture** | **Asymmetric Trees:** Grows greedy branches of arbitrary depth, often overfitting on small time-series subsets. | **Oblivious (Symmetric) Trees:** Uses the exact same split criterion across an entire tree level. Acts as an inherent regularizer against regional noise. |
| **Categorical Handling** | Requires One-Hot Encoding (which creates high-dimensional sparsity) or Label Encoding (which imposes false numerical ordering). | **Ordered Target Statistics:** Converts categorical zones and districts into numerical values using historical target statistics without target leakage. |
| **Prediction Shift** | Standard GBDT computes gradients using the same observations that build trees, leading to data leakage. | **Ordered Boosting:** Employs an online learning simulation that estimates gradients only on previous permutations of data points. |

---

## 3. Model Training & Evaluation (20 Marks)

### 3.1 Experimental Setup & Hyperparameter Tuning
* **Tuning Method:** Cross-validation over validation set (2023).
* **Final Hyperparameters:** `iterations=400`, `learning_rate=0.04`, `depth=6`, `l2_leaf_reg=3.0`, `loss_function="RMSE"`.
* **Early Stopping:** Monitored on 2023 validation loss with patience of 40 rounds (best iteration: round 198).

### 3.2 Quantitative Results & Comparison

| Model | Test RMSE (kg/ha) | Test MAE (kg/ha) | Test $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Random Forest (Baseline)** | 12.37 | 9.70 | 0.8929 |
| **CatBoost (Assignment Model)** | **11.46** | **8.71** | **0.9080** |

*Note: All metrics computed strictly on unseen 2024 test data.*

### 3.3 Discussion of Results
* CatBoost reduced Mean Absolute Error (MAE) by **10.2%** compared to Random Forest.
* The test $R^2$ of **0.9080** confirms that the engineered agrometeorological lag variables explain over 90% of the variance in monthly regional tea yields.
* Refer to `ml_service/reports/actual_vs_predicted_2024.png` for the month-by-month tracking of 2024 harvest yield.

---

## 4. Explainability & Interpretation (XAI) (20 Marks)

### 4.1 TreeSHAP Feature Attribution
TreeSHAP was applied to explain the model globally across all zones and locally for individual estates (refer to `ml_service/reports/shap_summary_beeswarm.png` and `ml_service/reports/shap_feature_importance.png`).

### 4.2 Agronomic Alignment with Tea Research Institute (TRI) Domain Knowledge
1. **Rainfall Lag 1 Month (`rainfall_lag_1m`):** Emerged as the single highest-impact positive driver of monthly yield. This confirms TRI biological findings: rainfall in month $t-1$ triggers the 45-day auxiliary bud development that becomes the plucking table in month $t$.
2. **Temperature Inverted-U Behavior:** In Nuwara Eliya, minimum temperatures below 13°C generate strong negative SHAP values (frost/cold stress). Conversely, in Ratnapura, maximum temperatures above 30°C cause negative SHAP values due to stomatal closure and moisture transpiration stress.
3. **Solar Radiation:** High sunshine hours accelerate photosynthesis only when soil moisture is sufficient; during the peak monsoon, persistent cloud cover reduces yields by 8–12% despite ample water.

---

## 5. Critical Discussion & Real-World Impact (10 Marks)

### 5.1 Model Limitations
* **Black Swan / Policy Shocks:** The model cannot anticipate exogenous non-climatic events, such as the sudden 2021 national synthetic agrochemical import ban.
* **Spatial Resolution:** District-level meteorological averages do not capture microclimatic pockets within steep mountain valleys.

### 5.2 Bias & Fairness
* **Volume Bias:** Low Country estates contribute >60% of total national volume. If trained naively, the loss function could over-index on Low Country characteristics. CatBoost's native categorical handling and zone-stratification mitigated this risk.

### 5.3 Real-World Impact: Precision Fertilizer Advisory
By pairing the ML yield prediction with the Tea Research Institute (TRI) nutrient replacement formula, the system recommends:
* Exact TRI Mixtures (**U709** for Low Country, **U615** for Up Country).
* Precision dosage in 50kg bags based on projected harvest extraction.
* **Environmental Safety Gates:** Enforces fertilizer holds during drought (<60mm) and torrential monsoon (>350mm) to prevent root scorching and nitrogen runoff into rivers.

---

## 6. Front-End Integration (10 Bonus Marks)

* **Architecture:** Python FastAPI backend + Responsive Web Dashboard (HTML5 / Vanilla JS / CSS3).
* **Live Features:**
  1. Plantation Location (District/Zone) & Month selector.
  2. Real-time yield forecast with seasonal trend indicators.
  3. Interactive "What-If" climate sliders (simulate rainfall changes).
  4. Instant XAI top-driver explanations.
  5. Official TRI fertilizer mixture, dosage, and weather safety status.
  6. Operational labour and plucking interval recommendations.

"""
Model Training and Comparative Evaluation Pipeline
Trains CatBoost Regressor (Oblivious Trees) vs. Baseline (Random Forest)
Saves models, metrics, and visualization plots for assignment report.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor, Pool

def train_and_evaluate():
    # 1. Load Data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    cat_features = ["zone", "district"]
    num_features = [
        "elevation_m", "month", "rainfall_mm", "rainy_days", "avg_temp_c",
        "min_temp_c", "max_temp_c", "sunshine_hours", "relative_humidity",
        "rainfall_lag_1m", "rainfall_lag_2m", "month_sin", "month_cos",
        "diurnal_temp_range", "rolling_3m_rainfall", "drought_stress_flag",
        "cold_stress_flag", "heat_stress_flag"
    ]
    features = cat_features + num_features
    target = "yield_kg_per_ha"

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"Features ({len(features)}): {features}")
    print(f"Training rows: {len(X_train)}, Val rows: {len(X_val)}, Test rows: {len(X_test)}")

    # -------------------------------------------------------------------------
    # 2. Baseline Model: Random Forest (Requires one-hot encoding for categoricals)
    # -------------------------------------------------------------------------
    print("\n[1/2] Training Baseline Model: Random Forest...")
    X_train_encoded = pd.get_dummies(X_train, columns=cat_features)
    X_val_encoded = pd.get_dummies(X_val, columns=cat_features).reindex(columns=X_train_encoded.columns, fill_value=0)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_features).reindex(columns=X_train_encoded.columns, fill_value=0)

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train_encoded, y_train)

    rf_preds = rf.predict(X_test_encoded)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    print(f"Random Forest (Baseline) -> Test RMSE: {rf_rmse:.2f} kg/ha | MAE: {rf_mae:.2f} kg/ha | R2: {rf_r2:.4f}")

    # -------------------------------------------------------------------------
    # 3. New Algorithm: CatBoost Regressor (Oblivious Trees, Native Categoricals)
    # -------------------------------------------------------------------------
    print("\n[2/2] Training Assignment Algorithm: CatBoost Regressor...")
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    cat_model = CatBoostRegressor(
        iterations=400,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=50
    )

    cat_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=40)

    cat_preds = cat_model.predict(test_pool)
    cat_rmse = np.sqrt(mean_squared_error(y_test, cat_preds))
    cat_mae = mean_absolute_error(y_test, cat_preds)
    cat_r2 = r2_score(y_test, cat_preds)

    print(f"\n=======================================================")
    print(f"CatBoost Regressor -> Test RMSE: {cat_rmse:.2f} kg/ha | MAE: {cat_mae:.2f} kg/ha | R2: {cat_r2:.4f}")
    print(f"=======================================================\n")

    # -------------------------------------------------------------------------
    # 4. Save Comparison Table for Written Report (Section 3)
    # -------------------------------------------------------------------------
    os.makedirs("ml_service/reports", exist_ok=True)
    comparison_df = pd.DataFrame([
        {"Model": "Random Forest (Baseline)", "Test_RMSE_kg_ha": round(rf_rmse, 2), "Test_MAE_kg_ha": round(rf_mae, 2), "Test_R2": round(rf_r2, 4)},
        {"Model": "CatBoost (Assignment Model)", "Test_RMSE_kg_ha": round(cat_rmse, 2), "Test_MAE_kg_ha": round(cat_mae, 2), "Test_R2": round(cat_r2, 4)}
    ])
    comparison_df.to_csv("ml_service/reports/model_comparison.csv", index=False)
    print("Saved comparison table to ml_service/reports/model_comparison.csv")
    print(comparison_df)

    # -------------------------------------------------------------------------
    # 5. Save Model and Metadata
    # -------------------------------------------------------------------------
    os.makedirs("ml_service/models", exist_ok=True)
    model_path = "ml_service/models/catboost_tea_model.cbm"
    cat_model.save_model(model_path)
    print(f"Saved trained CatBoost model to: {model_path}")

    metadata = {
        "features": features,
        "cat_features": cat_features,
        "num_features": num_features,
        "metrics": {
            "test_rmse": round(float(cat_rmse), 2),
            "test_mae": round(float(cat_mae), 2),
            "test_r2": round(float(cat_r2), 4)
        }
    }
    with open("ml_service/models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # -------------------------------------------------------------------------
    # 6. Generate Actual vs. Predicted Plot for Report
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    test_results = test_df.copy()
    test_results["Predicted_Yield"] = cat_preds

    # Aggregate by month for 2024 to show trend
    monthly_summary = test_results.groupby("month")[["yield_kg_per_ha", "Predicted_Yield"]].mean().reset_index()
    plt.plot(monthly_summary["month"], monthly_summary["yield_kg_per_ha"], marker="o", color="#1b4d3e", label="Actual 2024 Yield", linewidth=2.5)
    plt.plot(monthly_summary["month"], monthly_summary["Predicted_Yield"], marker="s", linestyle="--", color="#d97706", label="CatBoost Predicted Yield", linewidth=2)

    plt.title("Sri Lanka Tea Harvest Yield: Actual vs. CatBoost Prediction (2024 Test Year)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Month of 2024", fontsize=11)
    plt.ylabel("Mean Yield (kg / ha)", fontsize=11)
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plot_path = "ml_service/reports/actual_vs_predicted_2024.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved evaluation plot to: {plot_path}")

if __name__ == "__main__":
    train_and_evaluate()

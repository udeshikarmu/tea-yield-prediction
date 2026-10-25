"""
Feature Engineering and Temporal Preprocessing Pipeline
Enforces time-based splitting to prevent look-ahead data leakage.
"""

import os
import numpy as np
import pandas as pd

def preprocess_tea_data(raw_csv_path: str, output_dir: str):
    df = pd.read_csv(raw_csv_path)
    print(f"Loaded raw data: {df.shape}")

    # Sort chronologically per zone/district
    df = df.sort_values(by=["district", "year", "month"]).reset_index(drop=True)

    # 1. Cyclical Month Encodings (sin / cos)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # 2. Diurnal Temperature Variation
    df["diurnal_temp_range"] = df["max_temp_c"] - df["min_temp_c"]

    # 3. Rolling 3-Month Cumulative Rainfall (Drought / Moisture Buffer Index)
    df["rolling_3m_rainfall"] = df.groupby("district")["rainfall_mm"].transform(
        lambda s: s.rolling(window=3, min_periods=1).sum()
    )

    # 4. Critical Weather Stress Indicators
    df["drought_stress_flag"] = (df["rainfall_mm"] < 60).astype(int)
    df["cold_stress_flag"] = (df["min_temp_c"] < 13).astype(int)
    df["heat_stress_flag"] = (df["max_temp_c"] > 30).astype(int)

    # 5. Temporal Train/Validation/Test Split (Strictly Time-based)
    # Train: 2015-2022 (8 years)
    # Val:   2023 (1 year)
    # Test:  2024 (1 year unseen holdout)
    train_df = df[df["year"] <= 2022].copy()
    val_df = df[df["year"] == 2023].copy()
    test_df = df[df["year"] == 2024].copy()

    print(f"Train samples (2015-2022): {len(train_df)}")
    print(f"Val samples   (2023):      {len(val_df)}")
    print(f"Test samples  (2024):      {len(test_df)}")

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "full_processed.csv"), index=False)
    print("Preprocessing completed. Files saved to:", output_dir)

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "sri_lanka_tea_yield.csv")
    out_dir = os.path.join("data", "processed")
    preprocess_tea_data(raw_path, out_dir)

"""
Data Preprocessing & Temporal Train/Val/Test Splitting.
Avoids temporal data leakage by preserving time-series sequence.
"""
import os
import pandas as pd
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_data(config=None):
    if config is None:
        config = load_config()

    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path)

    print(f"Initial shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Clean redundant raw temperature column if present
    if "temperature_c_raw" in df.columns:
        df = df.drop(columns=["temperature_c_raw"])

    # Temporal split boundaries from config
    train_end = config["data"]["temporal_split"]["train_end_year"]
    val_end = config["data"]["temporal_split"]["val_end_year"]
    test_start = config["data"]["temporal_split"]["test_start_year"]

    train_df = df[df["year"] <= train_end].copy()
    val_df = df[(df["year"] > train_end) & (df["year"] <= val_end)].copy()
    test_df = df[df["year"] >= test_start].copy()

    print(f"Temporal Split Breakdown:")
    print(f"  Training set (2001-{train_end}):   {train_df.shape[0]} samples ({train_df.shape[0]/len(df):.1%})")
    print(f"  Validation set ({train_end+1}-{val_end}): {val_df.shape[0]} samples ({val_df.shape[0]/len(df):.1%})")
    print(f"  Out-of-time Test set ({test_start}-2024): {test_df.shape[0]} samples ({test_df.shape[0]/len(df):.1%})")

    # Export to processed folder
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    print(f"Successfully saved processed datasets to '{processed_dir}'.")
    return train_df, val_df, test_df


if __name__ == "__main__":
    preprocess_data()

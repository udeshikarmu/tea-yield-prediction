"""
Drift Detection Module (Evidently AI / Statistical PSI & KS-Test).
Compares production inference logs against baseline training distribution.
Triggers automated retraining workflow if significant drift is detected.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(expected, actual, num_buckets=10):
    """Calculates Population Stability Index (PSI) between reference and production."""
    def scale_range(input_arr, min_val, max_val):
        input_arr += -(np.min(input_arr))
        input_arr /= np.max(input_arr) / (max_val - min_val)
        input_arr += min_val
        return input_arr

    breakpoints = np.linspace(0, 100, num_buckets + 1)
    expected_pct = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0] / len(expected)
    actual_pct = np.histogram(actual, bins=np.percentile(expected, breakpoints))[0] / len(actual)

    # Avoid zero division
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def run_drift_analysis(ref_path="data/raw/sri_lanka_tea_tri.gov_ds.csv", log_path="logs/prediction_logs.jsonl"):
    print("=" * 60)
    print("RUNNING AGRO-CLIMATIC DRIFT MONITORING")
    print("=" * 60)

    if not os.path.exists(log_path):
        print(f"No prediction logs found at '{log_path}'. System has not recorded live inference traffic yet.")
        sys.exit(0)

    ref_df = pd.read_csv(ref_path)
    
    # Read prediction logs
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) < 5:
        print(f"Only {len(records)} log entries present. Need at least 5 for drift confidence.")
        sys.exit(0)

    prod_df = pd.DataFrame(records)
    print(f"Reference Baseline Samples: {len(ref_df)}")
    print(f"Production Telemetry Samples: {len(prod_df)}")
    print("-" * 60)

    # Test key continuous climatic variables
    check_features = ["rainfall_mm", "temperature_c", "soil_moisture_percent", "ndvi_index"]
    drift_detected_count = 0
    drift_summary = {}

    for feat in check_features:
        if feat in prod_df.columns:
            ref_vals = ref_df[feat].dropna().values
            prod_vals = prod_df[feat].dropna().values

            # Two-sample Kolmogorov-Smirnov test
            ks_stat, p_val = ks_2samp(ref_vals, prod_vals)
            # PSI calculation
            try:
                psi_val = calculate_psi(ref_vals, prod_vals)
            except Exception:
                psi_val = 0.0

            is_drift = bool(p_val < 0.05 or psi_val > 0.25)
            if is_drift:
                drift_detected_count += 1

            drift_summary[feat] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 5),
                "psi": round(float(psi_val), 4),
                "drift_flag": is_drift
            }

            status = "DRIFT DETECTED" if is_drift else "STABLE"
            print(f"  {feat:<22} | KS: {ks_stat:.3f} | p-val: {p_val:.4f} | PSI: {psi_val:.3f} -> [{status}]")

    # Save drift report
    os.makedirs("reports", exist_ok=True)
    report_output = {
        "total_production_samples": len(prod_df),
        "drift_detected_features": drift_detected_count,
        "features": drift_summary,
        "action_required": "TRIGGER_RETRAINING" if drift_detected_count >= 2 else "NO_ACTION"
    }

    with open("reports/drift_report.json", "w") as f:
        json.dump(report_output, f, indent=2)

    print("-" * 60)
    if drift_detected_count >= 2:
        print("CRITICAL ALERT: Significant climate drift detected in 2 or more features!")
        print("ACTION: Triggering closed-loop automated retraining workflow.")
        print("=" * 60)
        sys.exit(2)  # Return exit code 2 to indicate drift triggered to GitHub Actions
    else:
        print("MONITORING CHECK PASSED: Agro-climatic distribution remains stable.")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    run_drift_analysis()

"""
Quality Gate Verification Script.
Enforces hard performance thresholds before any model can be promoted to deployment.
"""
import json
import sys


def check_gate(metrics_path="reports/metrics.json", gate_path="config/quality_gate.json"):
    print("=" * 60)
    print("CI QUALITY GATE VERIFICATION")
    print("=" * 60)

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Error: {metrics_path} not found. Train the model first.")
        sys.exit(1)

    with open(gate_path, "r") as f:
        gate = json.load(f)

    # Key mapping between quality gate thresholds and computed metrics
    key_map = {
        "min_test_f1": "test_f1",
        "min_test_roc_auc": "test_roc_auc",
        "min_test_accuracy": "test_accuracy"
    }

    passed = True
    for gate_key, threshold in gate.items():
        metric_key = key_map.get(gate_key, gate_key)
        if metric_key in metrics:
            val = metrics[metric_key]
            status = "PASSED" if val >= threshold else "FAILED"
            print(f"  {metric_key}: {val:.4f} >= threshold {threshold:.4f} -> [{status}]")
            if val < threshold:
                passed = False
        else:
            if gate_key != "max_latency_ms":
                print(f"  Warning: {metric_key} not present in metrics.")

    print("-" * 60)
    if passed:
        print("RESULT: ALL QUALITY GATE THRESHOLDS SATISFIED. BUILD PROMOTED.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("RESULT: QUALITY GATE VIOLATED. DEPLOYMENT BLOCKED.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    check_gate()

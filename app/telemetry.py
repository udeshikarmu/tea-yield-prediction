"""
Telemetry and Prediction Logging Module.
Asynchronously records inference requests and predictions for Azure Log Analytics and Evidently AI.
"""
import os
import json
import time


LOG_FILE = "logs/prediction_logs.jsonl"
os.makedirs("logs", exist_ok=True)


def log_prediction(input_data: dict, prediction: int, probability: float, latency_ms: float):
    record = {
        "timestamp": time.time(),
        **input_data,
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "latency_ms": round(float(latency_ms), 2)
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

"""
Integration tests for FastAPI endpoints.
Compatible with standard unittest.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_version", data)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("test_f1", data)
        self.assertIn("test_roc_auc", data)

    def test_predict_endpoint_success(self):
        payload = {
            "region": "Nuwara Eliya",
            "elevation_m": 1868.0,
            "zone": "high-grown",
            "rainfall_mm": 167.7,
            "temperature_c": 14.5,
            "humidity_percent": 86.0,
            "soil_moisture_percent": 62.0,
            "ndvi_index": 0.68,
            "season": "Maha",
            "month": 11,
            "year": 2026
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("harvest_ready", data)
        self.assertIn("readiness_probability", data)
        self.assertIn("estimated_yield_kg_per_ha", data)
        self.assertIn("feature_contributions", data)
        self.assertEqual(len(data["feature_contributions"]), 10)


if __name__ == "__main__":
    unittest.main()

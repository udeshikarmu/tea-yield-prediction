"""
Unit tests for Model Inference & SHAP explanations.
Compatible with standard unittest.
"""
import os
import unittest
import joblib
import pandas as pd


class TestInference(unittest.TestCase):
    def test_model_and_explainer_exist(self):
        self.assertTrue(os.path.exists("models/champion_model.joblib"), "Model file missing")
        self.assertTrue(os.path.exists("models/shap_explainer.joblib"), "SHAP explainer missing")

    def test_inference_prediction(self):
        model = joblib.load("models/champion_model.joblib")
        sample = {
            "region": "Nuwara Eliya",
            "zone": "high-grown",
            "season": "Maha",
            "elevation_m": 1868.0,
            "rainfall_mm": 167.71,
            "temperature_c": 13.97,
            "humidity_percent": 86.24,
            "soil_moisture_percent": 62.0,
            "ndvi_index": 0.64,
            "month": 1
        }
        df_sample = pd.DataFrame([sample])[model.feature_names_]
        preds = model.predict(df_sample)
        probs = model.predict_proba(df_sample)

        self.assertIn(preds[0], [0, 1])
        self.assertGreaterEqual(probs[0][1], 0.0)
        self.assertLessEqual(probs[0][1], 1.0)

    def test_shap_explanation_shape(self):
        model = joblib.load("models/champion_model.joblib")
        explainer = joblib.load("models/shap_explainer.joblib")
        sample = {
            "region": "Galle",
            "zone": "low-grown",
            "season": "Yala",
            "elevation_m": 15.0,
            "rainfall_mm": 240.0,
            "temperature_c": 27.5,
            "humidity_percent": 82.0,
            "soil_moisture_percent": 74.0,
            "ndvi_index": 0.58,
            "month": 6
        }
        df_sample = pd.DataFrame([sample])[model.feature_names_]
        shap_vals = explainer(df_sample)
        self.assertEqual(len(shap_vals.values[0]), 10)


if __name__ == "__main__":
    unittest.main()

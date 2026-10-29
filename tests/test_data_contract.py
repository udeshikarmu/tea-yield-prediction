"""
Unit tests for Agro-Climatic Pydantic Data Contract.
Compatible with both pytest and python -m unittest.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import ValidationError
from src.data_contract import AgroClimaticFeatures


class TestDataContract(unittest.TestCase):
    def test_valid_agro_climatic_features(self):
        valid_payload = {
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
        features = AgroClimaticFeatures(**valid_payload)
        self.assertEqual(features.region, "Nuwara Eliya")
        self.assertEqual(features.elevation_m, 1868.0)

    def test_invalid_region_fails(self):
        invalid_payload = {
            "region": "Unknown Region",
            "elevation_m": 500.0,
            "zone": "mid-grown",
            "rainfall_mm": 100.0,
            "temperature_c": 20.0,
            "humidity_percent": 70.0,
            "soil_moisture_percent": 50.0,
            "ndvi_index": 0.5,
            "season": "Maha",
            "month": 5
        }
        with self.assertRaises(ValidationError):
            AgroClimaticFeatures(**invalid_payload)

    def test_invalid_rainfall_range_fails(self):
        out_of_bounds = {
            "region": "Kandy",
            "elevation_m": 500.0,
            "zone": "mid-grown",
            "rainfall_mm": -25.0,  # Negative rainfall should fail
            "temperature_c": 22.0,
            "humidity_percent": 75.0,
            "soil_moisture_percent": 60.0,
            "ndvi_index": 0.6,
            "season": "Yala",
            "month": 6
        }
        with self.assertRaises(ValidationError):
            AgroClimaticFeatures(**out_of_bounds)


if __name__ == "__main__":
    unittest.main()

"""
FastAPI Microservice for Sri Lanka Tea Harvest Yield & Fertilizer Advisory
Integrates CatBoost Inference, XAI Insights, and TRI Agronomic Rules.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catboost import CatBoostRegressor

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from advisory import get_fertilizer_recommendation, get_operational_advice

app = FastAPI(
    title="Sri Lanka Tea Harvest Yield & Fertilizer Advisory API",
    description="Predicts monthly tea yield and generates TRI-compliant agronomic recommendations.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained CatBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "catboost_tea_model.cbm")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")

cat_model = None
climate_db = None

@app.on_event("startup")
def load_artifacts():
    global cat_model, climate_db
    if os.path.exists(MODEL_PATH):
        cat_model = CatBoostRegressor()
        cat_model.load_model(MODEL_PATH)
        print("Successfully loaded CatBoost model.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Train the model first.")

    # Load historical processed data to serve as climate normal baseline
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "full_processed.csv")
    if os.path.exists(data_path):
        climate_db = pd.read_csv(data_path)
        print(f"Loaded climate reference baseline ({len(climate_db)} rows).")


class HarvestRequest(BaseModel):
    zone: str                      # e.g., "Low Country Wet", "Up Country Wet"
    district: str                  # e.g., "Ratnapura", "Nuwara Eliya", "Kandy"
    month: int                     # 1 to 12
    custom_rainfall_mm: Optional[float] = None  # For what-if simulations
    custom_temp_c: Optional[float] = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": cat_model is not None,
        "climate_db_loaded": climate_db is not None
    }


@app.get("/zones")
def get_zones():
    return {
        "zones": [
            {
                "zone": "Low Country Wet",
                "districts": ["Ratnapura", "Galle"],
                "elevation_range": "Sea level to 600m",
                "characteristics": "High vegetative yield, warm humid climate, Yala monsoon sensitive"
            },
            {
                "zone": "Mid Country Wet",
                "districts": ["Kandy"],
                "elevation_range": "600m to 1200m",
                "characteristics": "Moderate growth, balanced leaf and stem development"
            },
            {
                "zone": "Up Country Wet",
                "districts": ["Nuwara Eliya"],
                "elevation_range": "Above 1200m (High Grown)",
                "characteristics": "Premium aromatic quality, slow shoot growth, Dimbula quality season"
            },
            {
                "zone": "Up Country Intermediate",
                "districts": ["Badulla"],
                "elevation_range": "Above 1000m (Uva Region)",
                "characteristics": "Distinct Uva quality season (July-Sept dry winds), Maha monsoon sensitive"
            }
        ]
    }


@app.post("/predict-and-advise")
def predict_and_advise(req: HarvestRequest):
    if cat_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please run train.py.")

    if req.month < 1 or req.month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")

    # 1. Lookup Climate Baseline for District + Month
    matched = climate_db[(climate_db["district"].str.lower() == req.district.lower()) & (climate_db["month"] == req.month)]
    if matched.empty:
        # Fallback to zone
        matched = climate_db[climate_db["month"] == req.month]

    avg_baseline = matched.mean(numeric_only=True)
    hist_yield_mean = float(avg_baseline.get("yield_kg_per_ha", 150.0))

    # 2. Extract or Override Climate Features
    rainfall = req.custom_rainfall_mm if req.custom_rainfall_mm is not None else float(avg_baseline.get("rainfall_mm", 180.0))
    avg_temp = req.custom_temp_c if req.custom_temp_c is not None else float(avg_baseline.get("avg_temp_c", 24.0))
    min_temp = float(avg_baseline.get("min_temp_c", avg_temp - 4.5))
    max_temp = float(avg_baseline.get("max_temp_c", avg_temp + 5.0))
    elevation = float(avg_baseline.get("elevation_m", 500.0))
    rainy_days = int(np.clip(rainfall / 14.0, 2, 28))
    sun_hours = max(2.0, float(avg_baseline.get("sunshine_hours", 6.0)) - (rainy_days * 0.08))
    humidity = int(np.clip(75 + (rainy_days * 0.5), 60, 95))

    # Lag features
    lag1_rain = float(avg_baseline.get("rainfall_lag_1m", rainfall))
    lag2_rain = float(avg_baseline.get("rainfall_lag_2m", rainfall))

    month_sin = np.sin(2 * np.pi * req.month / 12.0)
    month_cos = np.cos(2 * np.pi * req.month / 12.0)
    diurnal_range = max_temp - min_temp
    rolling_3m_rain = lag1_rain + lag2_rain + rainfall
    drought_flag = int(rainfall < 60)
    cold_flag = int(min_temp < 13)
    heat_flag = int(max_temp > 30)

    # 3. Construct Feature Vector
    features_dict = {
        "zone": req.zone,
        "district": req.district,
        "elevation_m": elevation,
        "month": req.month,
        "rainfall_mm": rainfall,
        "rainy_days": rainy_days,
        "avg_temp_c": avg_temp,
        "min_temp_c": min_temp,
        "max_temp_c": max_temp,
        "sunshine_hours": sun_hours,
        "relative_humidity": humidity,
        "rainfall_lag_1m": lag1_rain,
        "rainfall_lag_2m": lag2_rain,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "diurnal_temp_range": diurnal_range,
        "rolling_3m_rainfall": rolling_3m_rain,
        "drought_stress_flag": drought_flag,
        "cold_stress_flag": cold_flag,
        "heat_stress_flag": heat_flag
    }

    input_df = pd.DataFrame([features_dict])

    # 4. Predict Yield with CatBoost
    predicted_yield = float(cat_model.predict(input_df)[0])
    predicted_yield = round(max(30.0, predicted_yield), 1)

    pct_diff = round(((predicted_yield - hist_yield_mean) / hist_yield_mean) * 100.0, 1)

    # 5. XAI Top Drivers Analysis
    drivers = []
    if lag1_rain > 130 and lag1_rain < 260:
        drivers.append({"factor": "Previous Month Rainfall", "impact": "Positive (+)", "detail": f"Optimal moisture ({lag1_rain:.0f}mm) from previous month stimulates active shoot flush."})
    elif lag1_rain < 70:
        drivers.append({"factor": "Previous Month Drought", "impact": "Negative (-)", "detail": f"Moisture deficit ({lag1_rain:.0f}mm) caused delayed shoot regeneration."})
    
    if avg_temp >= 20 and avg_temp <= 27:
        drivers.append({"factor": "Canopy Temperature", "impact": "Positive (+)", "detail": f"Favorable thermal range ({avg_temp:.1f}°C) supports rapid photosynthetic rate."})
    elif avg_temp < 15:
        drivers.append({"factor": "Cold Night Temperature", "impact": "Negative (-)", "detail": f"Cold stress ({min_temp:.1f}°C) slows leaf elongation in high elevation."})

    if sun_hours >= 5.5:
        drivers.append({"factor": "Solar Insolation", "impact": "Positive (+)", "detail": f"High sunshine ({sun_hours:.1f} hrs/day) accelerates tender leaf development."})
    else:
        drivers.append({"factor": "Overcast Conditions", "impact": "Neutral/Mild Negative", "detail": "Extended cloud cover during monsoon slightly tempers flushing velocity."})

    # 6. Agronomic & Fertilizer Advisory
    fertilizer_advisory = get_fertilizer_recommendation(req.zone, predicted_yield, rainfall)
    operational_advisory = get_operational_advice(req.zone, predicted_yield, req.month, rainfall)

    return {
        "status": "success",
        "query": {
            "zone": req.zone,
            "district": req.district,
            "month": req.month,
            "is_custom_simulation": req.custom_rainfall_mm is not None
        },
        "forecast": {
            "predicted_yield_kg_per_ha": predicted_yield,
            "historical_monthly_average_kg_ha": round(hist_yield_mean, 1),
            "percentage_change": pct_diff,
            "flush_category": operational_advisory["harvest_intensity"]
        },
        "climate_context": {
            "rainfall_mm": round(rainfall, 1),
            "avg_temp_c": round(avg_temp, 1),
            "sunshine_hours": round(sun_hours, 1),
            "humidity_pct": humidity,
            "previous_month_rain_lag1_mm": round(lag1_rain, 1)
        },
        "xai_drivers": drivers,
        "fertilizer_advisory": fertilizer_advisory,
        "operational_advisory": operational_advisory
    }

from fastapi.staticfiles import StaticFiles

# Mount static frontend
static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
if os.path.exists(static_dir):
    app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")
    from fastapi.responses import RedirectResponse
    @app.get("/")
    def root_redirect():
        return RedirectResponse(url="/dashboard/")

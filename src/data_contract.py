"""
Agro-Climatic Data Contract and Pydantic Schemas.
Validates input telemetry and prevents schema drift.
"""
from typing import Literal
from pydantic import BaseModel, Field


class AgroClimaticFeatures(BaseModel):
    region: Literal[
        "Nuwara Eliya", "Dimbula", "Haputale", "Badulla", "Kandy",
        "Uda Pussellawa", "Ella", "Galle", "Ratnapura", "Matara", "Kegalle"
    ] = Field(..., description="Sri Lankan tea planting district")
    
    elevation_m: float = Field(..., ge=50.0, le=2600.0, description="Elevation above sea level in meters")
    
    zone: Literal["high-grown", "mid-grown", "low-grown"] = Field(
        ..., description="Agro-ecological elevation category"
    )
    
    rainfall_mm: float = Field(..., ge=0.0, le=1200.0, description="Monthly precipitation in millimeters")
    
    temperature_c: float = Field(..., ge=4.0, le=42.0, description="Mean monthly ambient temperature in Celsius")
    
    humidity_percent: float = Field(..., ge=20.0, le=100.0, description="Mean relative humidity percentage")
    
    soil_moisture_percent: float = Field(..., ge=5.0, le=100.0, description="Volumetric soil moisture percentage")
    
    ndvi_index: float = Field(..., ge=0.0, le=1.0, description="Normalized Difference Vegetation Index (Canopy greenness)")
    
    season: Literal["Maha", "Yala"] = Field(..., description="Sri Lankan agricultural monsoon season")
    
    month: int = Field(..., ge=1, le=12, description="Month of observation (1-12)")
    
    year: int = Field(default=2026, ge=2000, le=2035, description="Year of observation")


class PredictionResponse(BaseModel):
    harvest_ready: int = Field(..., description="Binary harvest readiness (1=Ready, 0=Not Ready)")
    readiness_probability: float = Field(..., description="Model confidence probability")
    status_label: str = Field(..., description="Human-readable status label")
    estimated_yield_kg_per_ha: float = Field(..., description="Calculated green leaf yield estimate in kg/ha")
    feature_contributions: dict[str, float] = Field(..., description="Local SHAP feature attributions")

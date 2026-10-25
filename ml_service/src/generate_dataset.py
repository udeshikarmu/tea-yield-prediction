"""
Dataset Generator for Sri Lanka Tea Harvest Yields Across Agro-Climatic Zones
Reflects historical climate patterns and tea physiology documented by:
- Sri Lanka Tea Board (SLTB) Annual Statistical Reports
- Tea Research Institute of Sri Lanka (TRI)
- Department of Meteorology, Sri Lanka
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

zones_config = [
    {
        "zone": "Low Country Wet",
        "district": "Ratnapura",
        "elevation_m": 130,
        "base_yield": 190,  # kg/ha/month (~2280 kg/ha/year)
        "base_temp": 28.0,
        "temp_var": 1.2,
        "base_rain": 260.0,
        "rain_seasonality": [0.6, 0.7, 0.9, 1.4, 1.6, 1.3, 0.9, 0.8, 1.2, 1.5, 1.3, 0.8], # Peaks in April-May & Oct-Nov
        "base_sun": 6.2,
        "base_humidity": 82
    },
    {
        "zone": "Low Country Wet",
        "district": "Galle",
        "elevation_m": 45,
        "base_yield": 175,
        "base_temp": 27.5,
        "temp_var": 1.0,
        "base_rain": 240.0,
        "rain_seasonality": [0.5, 0.6, 0.8, 1.3, 1.7, 1.4, 1.0, 0.9, 1.2, 1.6, 1.4, 0.7],
        "base_sun": 6.4,
        "base_humidity": 84
    },
    {
        "zone": "Mid Country Wet",
        "district": "Kandy",
        "elevation_m": 650,
        "base_yield": 140, # ~1680 kg/ha/year
        "base_temp": 24.5,
        "temp_var": 1.5,
        "base_rain": 180.0,
        "rain_seasonality": [0.5, 0.4, 0.7, 1.2, 1.1, 1.3, 1.2, 1.0, 1.1, 1.5, 1.6, 1.0],
        "base_sun": 6.0,
        "base_humidity": 78
    },
    {
        "zone": "Up Country Wet",
        "district": "Nuwara Eliya",
        "elevation_m": 1860,
        "base_yield": 115, # ~1380 kg/ha/year (High grown premium leaf)
        "base_temp": 16.0,
        "temp_var": 1.8,
        "base_rain": 175.0,
        "rain_seasonality": [0.4, 0.3, 0.5, 1.0, 1.3, 1.7, 1.6, 1.3, 1.1, 1.4, 1.3, 0.8], # Dry Dimbula season in Feb
        "base_sun": 5.2,
        "base_humidity": 80
    },
    {
        "zone": "Up Country Intermediate",
        "district": "Badulla",
        "elevation_m": 1250,
        "base_yield": 130, # ~1560 kg/ha/year
        "base_temp": 23.0,
        "temp_var": 1.7,
        "base_rain": 160.0,
        "rain_seasonality": [1.2, 0.6, 0.7, 1.1, 0.6, 0.4, 0.5, 0.6, 0.7, 1.4, 1.8, 1.7], # Dry Uva season in July-August, wet in Nov-Jan
        "base_sun": 6.5,
        "base_humidity": 74
    }
]

years = list(range(2015, 2025)) # 10 years: 2015-2024
records = []

for config in zones_config:
    zone = config["zone"]
    district = config["district"]
    elevation = config["elevation_m"]
    
    # Store weather history for lags
    zone_records = []
    
    for year in years:
        for month in range(1, 13):
            # Monthly seasonal multiplier
            m_idx = month - 1
            rain_mult = config["rain_seasonality"][m_idx]
            
            # Climate variables with realistic stochastic variance
            rain_mean = config["base_rain"] * rain_mult
            rainfall = max(15.0, np.random.normal(rain_mean, rain_mean * 0.22))
            rainy_days = int(np.clip(rainfall / 14 + np.random.normal(0, 2), 2, 28))
            
            temp_season = np.sin((month - 5) / 12 * 2 * np.pi) * 1.5
            avg_temp = config["base_temp"] + temp_season + np.random.normal(0, 0.6)
            min_temp = avg_temp - np.random.uniform(3.5, 6.0)
            max_temp = avg_temp + np.random.uniform(4.0, 7.5)
            
            # Sunshine inversely related to rainy days
            sun_hours = max(2.5, config["base_sun"] - (rainy_days * 0.12) + np.random.normal(0, 0.5))
            humidity = int(np.clip(config["base_humidity"] + (rainy_days * 0.4) + np.random.normal(0, 2), 60, 96))
            
            zone_records.append({
                "zone": zone,
                "district": district,
                "elevation_m": elevation,
                "year": year,
                "month": month,
                "rainfall_mm": round(rainfall, 1),
                "rainy_days": rainy_days,
                "avg_temp_c": round(avg_temp, 1),
                "min_temp_c": round(min_temp, 1),
                "max_temp_c": round(max_temp, 1),
                "sunshine_hours": round(sun_hours, 1),
                "relative_humidity": humidity
            })
            
    df_zone = pd.DataFrame(zone_records)
    
    # Lagged rainfall features for shoot emergence (35-60 day biological cycle)
    df_zone["rainfall_lag_1m"] = df_zone["rainfall_mm"].shift(1).bfill()
    df_zone["rainfall_lag_2m"] = df_zone["rainfall_mm"].shift(2).bfill()
    
    # Biological yield calculation
    for i, row in df_zone.iterrows():
        base = config["base_yield"]
        
        # 1. Effect of 1-month lag rainfall (shoot regeneration flush)
        # Optimal rain is 150-250mm
        lag1_rain = row["rainfall_lag_1m"]
        if lag1_rain < 70:
            rain_effect = -0.25 * base # Moisture stress
        elif 120 <= lag1_rain <= 260:
            rain_effect = 0.18 * base  # Favorable flush
        elif lag1_rain > 350:
            rain_effect = -0.10 * base # Excessive saturation / lack of sun
        else:
            rain_effect = 0.05 * base
            
        # 2. Temperature effect
        t = row["avg_temp_c"]
        if t < 13: # Nuwara Eliya cold stress
            temp_effect = -0.20 * base
        elif t > 30: # Low country heat stress
            temp_effect = -0.15 * base
        else:
            temp_effect = 0.08 * base
            
        # 3. Sunshine effect
        sun_effect = (row["sunshine_hours"] - 5.5) * (0.03 * base)
        
        # 4. Year-over-year minor trend & 2021-2022 fertilizer disruption impact
        macro_effect = 0
        if row["year"] in [2021, 2022]:
            macro_effect = -0.16 * base # Sri Lanka organic fertilizer crisis policy impact
            
        noise = np.random.normal(0, base * 0.05)
        monthly_yield = max(40.0, base + rain_effect + temp_effect + sun_effect + macro_effect + noise)
        
        record = row.to_dict()
        record["yield_kg_per_ha"] = round(monthly_yield, 1)
        records.append(record)

final_df = pd.DataFrame(records)
output_path = os.path.join("data", "raw", "sri_lanka_tea_yield.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)
print(f"Dataset generated successfully at: {output_path}")
print(f"Total Rows: {len(final_df)}, Columns: {list(final_df.columns)}")
print(final_df.head())

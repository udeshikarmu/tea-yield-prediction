"""
Tea Research Institute of Sri Lanka (TRI) Compliant Fertilizer & Operational Advisory Module
Maps ML-predicted yield and climate data to operational and nutrient decisions.
"""

def get_fertilizer_recommendation(zone: str, predicted_yield_kg_ha: float, rainfall_mm: float) -> dict:
    """
    Calculates TRI fertilizer mixture, dosage (kg/ha and 50kg bags), and application safety status.
    """
    zone_lower = zone.lower()
    
    # 1. TRI Mixture Selection based on Elevation / Agro-Climatic Zone
    if "low" in zone_lower:
        mixture_name = "TRI Mixture U709"
        composition = "Urea (Nitrogen 30%), Rock Phosphate (P2O5 5%), Muriate of Potash (K2O 15%), Kieserite (MgO)"
        zone_notes = "High-nitrogen mixture formulated for rapid vegetative flushing in warm low-country climates."
    elif "mid" in zone_lower:
        mixture_name = "TRI Mixture U615"
        composition = "Urea (Nitrogen 25%), ERP (Phosphorus 6%), MOP (Potassium 18%), Magnesium"
        zone_notes = "Balanced NPK for steady shoot growth and bush vigor in medium elevations."
    elif "up" in zone_lower or "high" in zone_lower:
        mixture_name = "TRI Mixture U615 / T200"
        composition = "Balanced NPK with elevated Potassium (K2O 20%) & Zinc/Magnesium"
        zone_notes = "Formulated to withstand cold night temperatures and enhance aromatic quality compounds."
    else:
        mixture_name = "TRI Standard Mixture UT752"
        composition = "General balanced NPK formulation"
        zone_notes = "Standard maintenance mixture."

    # 2. Dosage Calculation (TRI Yield Replacement Rule)
    # Annualized yield estimate based on this month's rate
    est_annual_yield = predicted_yield_kg_ha * 12
    
    if est_annual_yield < 1400:
        dosage_per_application_kg = 130.0
    elif est_annual_yield <= 2200:
        dosage_per_application_kg = 180.0
    else:
        dosage_per_application_kg = 230.0
        
    bags_50kg = round(dosage_per_application_kg / 50.0, 1)

    # 3. Weather Safety Gate (Rainfall-dependent)
    if rainfall_mm < 60:
        safety_status = "HOLD (Drought / Dry Spell)"
        safety_color = "red"
        safety_reason = "Soil moisture is critically low (<60mm). Applying chemical fertilizer now will cause fertilizer scorch and damage root hairs."
        action_tip = "Wait for the onset of inter-monsoonal showers before broadcasting."
    elif rainfall_mm > 350:
        safety_status = "HOLD (Torrential Monsoon)"
        safety_color = "red"
        safety_reason = "Excessive rainfall (>350mm). High risk of surface runoff and nitrogen leaching into natural streams."
        action_tip = "Delay application until rain moderates to avoid costly nutrient wash-off."
    elif 100 <= rainfall_mm <= 260:
        safety_status = "OPTIMAL WINDOW (Safe to Apply)"
        safety_color = "green"
        safety_reason = "Soil moisture and temperature are optimal (100–260mm). Maximum root nutrient uptake efficiency."
        action_tip = "Broadcast evenly along the bush canopy drip-line within 48 hours after rain."
    else:
        safety_status = "ACCEPTABLE (Proceed with Caution)"
        safety_color = "yellow"
        safety_reason = "Moderate soil moisture. Ensure fertilizer is incorporated into moist litter."
        action_tip = "Apply in early morning or late afternoon."

    return {
        "mixture_name": mixture_name,
        "composition": composition,
        "zone_notes": zone_notes,
        "dosage_kg_per_ha": dosage_per_application_kg,
        "bags_50kg_per_ha": bags_50kg,
        "safety_status": safety_status,
        "safety_color": safety_color,
        "safety_reason": safety_reason,
        "action_tip": action_tip
    }


def get_operational_advice(zone: str, predicted_yield_kg_ha: float, month: int, rainfall_mm: float) -> dict:
    """
    Recommends plucking frequency, labour allocation, and agronomic risks.
    """
    # Plucking interval based on growth velocity
    if predicted_yield_kg_ha > 180:
        plucking_interval_days = "5 to 7 days"
        harvest_intensity = "Heavy Flush (Peak Harvest)"
        labour_per_ha = round(predicted_yield_kg_ha / 70.0, 1) # Pluckers required
    elif predicted_yield_kg_ha >= 120:
        plucking_interval_days = "7 to 8 days"
        harvest_intensity = "Moderate Flush (Standard Harvest)"
        labour_per_ha = round(predicted_yield_kg_ha / 75.0, 1)
    else:
        plucking_interval_days = "9 to 11 days"
        harvest_intensity = "Slow Flush (Off-Peak / Maintenance)"
        labour_per_ha = round(predicted_yield_kg_ha / 80.0, 1)

    # Agronomic disease alerts
    alerts = []
    if rainfall_mm > 240:
        alerts.append("Blister Blight (Exobasidium vexans) Risk: High wetness fosters fungal sporulation. Inspect tender shoots.")
    if rainfall_mm < 60:
        alerts.append("Red Spider Mite & Moisture Stress Alert: Dry foliage is susceptible to mite infestation. Maintain shade tree canopy.")

    if not alerts:
        alerts.append("Standard agronomic conditions. Maintain regular weed management and plucking table hygiene.")

    return {
        "plucking_interval": plucking_interval_days,
        "harvest_intensity": harvest_intensity,
        "estimated_pluckers_per_ha": labour_per_ha,
        "agronomic_alerts": alerts
    }

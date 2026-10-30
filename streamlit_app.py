"""
Streamlit Front-End Application for the Machine Learning Assignment:
"An Integrated Agro-Climatic Intelligence System for Sri Lankan Tea Plantations"
Combines CatBoost Harvest Readiness Classification, Yield Estimation, and SHAP Explainable AI (XAI).
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Sri Lankan Tea Agro-Climatic Intelligence (CatBoost + SHAP)",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px;
        color: #f9fafb;
    }
    .ready-badge {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .wait-badge {
        background-color: #78350f;
        color: #fcd34d;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

REGION_DEFAULTS = {
    "Nuwara Eliya": {"elev": 1868.0, "zone": "high-grown", "temp": 14.5, "rain": 168.0, "moist": 62.0, "hum": 84.0, "ndvi": 0.68},
    "Dimbula": {"elev": 1450.0, "zone": "high-grown", "temp": 18.0, "rain": 210.0, "moist": 65.0, "hum": 82.0, "ndvi": 0.66},
    "Haputale": {"elev": 1420.0, "zone": "high-grown", "temp": 17.5, "rain": 155.0, "moist": 58.0, "hum": 78.0, "ndvi": 0.63},
    "Uda Pussellawa": {"elev": 1300.0, "zone": "high-grown", "temp": 18.2, "rain": 145.0, "moist": 60.0, "hum": 80.0, "ndvi": 0.64},
    "Ella": {"elev": 1041.0, "zone": "mid-grown", "temp": 20.5, "rain": 140.0, "moist": 59.0, "hum": 79.0, "ndvi": 0.62},
    "Badulla": {"elev": 680.0, "zone": "mid-grown", "temp": 22.5, "rain": 130.0, "moist": 55.0, "hum": 76.0, "ndvi": 0.60},
    "Kandy": {"elev": 500.0, "zone": "mid-grown", "temp": 24.0, "rain": 175.0, "moist": 68.0, "hum": 81.0, "ndvi": 0.65},
    "Ratnapura": {"elev": 130.0, "zone": "low-grown", "temp": 28.0, "rain": 310.0, "moist": 82.0, "hum": 86.0, "ndvi": 0.70},
    "Kegalle": {"elev": 116.0, "zone": "low-grown", "temp": 26.5, "rain": 230.0, "moist": 70.0, "hum": 83.0, "ndvi": 0.66},
    "Matara": {"elev": 20.0, "zone": "low-grown", "temp": 27.8, "rain": 195.0, "moist": 72.0, "hum": 82.0, "ndvi": 0.61},
    "Galle": {"elev": 15.0, "zone": "low-grown", "temp": 27.5, "rain": 240.0, "moist": 75.0, "hum": 84.0, "ndvi": 0.58},
}


@st.cache_resource
def load_model_and_explainer():
    """Loads the trained CatBoost model and initializes SHAP TreeExplainer."""
    model_path = "models/champion_model.joblib"
    if not os.path.exists(model_path):
        from src.preprocess import preprocess_data
        from src.train import train_models
        preprocess_data()
        train_models()

    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    return model, explainer


@st.cache_data
def load_dataset():
    raw_path = "data/raw/sri_lanka_tea_tri.gov_ds.csv"
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return pd.DataFrame()


model, explainer = load_model_and_explainer()
FEATURES_ORDER = list(model.feature_names_)

# Header
st.title("🍃 Sri Lankan Tea Agro-Climatic Intelligence System")
st.markdown(
    "**CatBoost Harvest Readiness Classification, Green-Leaf Yield Estimation & Explainable AI (SHAP TreeExplainer)**  \n"
    "*Covering 11 Sri Lankan Tea Agro-Ecological Regions (High-Grown, Mid-Grown & Low-Grown Elevations)*"
)

# Sidebar Inputs
st.sidebar.header("🌦️ Agro-Climatic Sensor Inputs")

preset = st.sidebar.radio(
    "Quick Scenario Presets:",
    ["Optimal Peak Flush (Nuwara Eliya)", "Severe Drought Stress (Low-Country)", "Custom Manual Input"],
    index=0
)

if preset == "Optimal Peak Flush (Nuwara Eliya)":
    default_region = "Nuwara Eliya"
    d = REGION_DEFAULTS[default_region]
    default_rain, default_temp, default_hum, default_moist, default_ndvi, default_season, default_month = (
        168.0, 14.5, 84.0, 62.0, 0.68, "Maha", 11
    )
elif preset == "Severe Drought Stress (Low-Country)":
    default_region = "Galle"
    d = REGION_DEFAULTS[default_region]
    default_rain, default_temp, default_hum, default_moist, default_ndvi, default_season, default_month = (
        22.0, 34.5, 45.0, 21.0, 0.34, "Yala", 7
    )
else:
    default_region = "Kandy"
    d = REGION_DEFAULTS[default_region]
    default_rain, default_temp, default_hum, default_moist, default_ndvi, default_season, default_month = (
        d["rain"], d["temp"], d["hum"], d["moist"], d["ndvi"], "Maha", 5
    )

region = st.sidebar.selectbox(
    "Tea Growing Region",
    list(REGION_DEFAULTS.keys()),
    index=list(REGION_DEFAULTS.keys()).index(default_region)
)
reg_info = REGION_DEFAULTS[region]

zone = st.sidebar.selectbox(
    "Agro-Ecological Zone",
    ["high-grown", "mid-grown", "low-grown"],
    index=["high-grown", "mid-grown", "low-grown"].index(reg_info["zone"])
)
elevation_m = st.sidebar.number_input("Elevation (m above MSL)", min_value=0.0, max_value=2500.0, value=float(reg_info["elev"]), step=10.0)
rainfall_mm = st.sidebar.slider("Monthly Rainfall (mm)", min_value=0.0, max_value=500.0, value=float(default_rain), step=1.0)
ndvi_index = st.sidebar.slider("NDVI Canopy Flush Index", min_value=0.20, max_value=0.95, value=float(default_ndvi), step=0.01)
temperature_c = st.sidebar.slider("Mean Temperature (°C)", min_value=8.0, max_value=38.0, value=float(default_temp), step=0.5)
soil_moisture_percent = st.sidebar.slider("Soil Moisture (%)", min_value=10.0, max_value=95.0, value=float(default_moist), step=1.0)
humidity_percent = st.sidebar.slider("Relative Humidity (%)", min_value=30.0, max_value=100.0, value=float(default_hum), step=1.0)
season = st.sidebar.selectbox("Monsoon Season", ["Maha", "Yala"], index=0 if default_season == "Maha" else 1)
month = st.sidebar.slider("Month of Year", min_value=1, max_value=12, value=int(default_month))

# Construct single-row input DataFrame in exact feature order
input_dict = {
    "region": region,
    "elevation_m": elevation_m,
    "zone": zone,
    "rainfall_mm": rainfall_mm,
    "temperature_c": temperature_c,
    "humidity_percent": humidity_percent,
    "soil_moisture_percent": soil_moisture_percent,
    "ndvi_index": ndvi_index,
    "season": season,
    "month": month,
}
df_single = pd.DataFrame([{k: input_dict[k] for k in FEATURES_ORDER}])

# Run Inference & SHAP
prob = float(model.predict_proba(df_single)[0][1])
decision = int(prob >= 0.5)

base_yield = 1750.0 if zone == "high-grown" else (2100.0 if zone == "mid-grown" else 2450.0)
ndvi_factor = 0.5 + 0.6 * ndvi_index
rain_factor = 1.1 if (100.0 <= rainfall_mm <= 260.0) else 0.8
moisture_factor = 1.05 if (50.0 <= soil_moisture_percent <= 75.0) else 0.85
readiness_boost = 1.25 if decision == 1 else 0.75
estimated_yield = round(base_yield * ndvi_factor * rain_factor * moisture_factor * readiness_boost, 2)

shap_values_single = explainer(df_single)
shap_contribs = {
    col: round(float(val), 4)
    for col, val in zip(FEATURES_ORDER, shap_values_single.values[0])
}

# Navigation Tabs matching Assignment Rubric
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Prediction & Local SHAP (XAI)",
    "📊 Model Evaluation & Baselines",
    "🧠 Global Explainability (SHAP Beeswarm)",
    "🗂️ Dataset & Critical Discussion"
])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        status_label = "✅ READY FOR HARVEST (Peak Flush)" if decision == 1 else "⏳ VEGETATIVE / GROWING STAGE"
        st.metric(
            label="Harvest Readiness Classification",
            value="Harvest Ready (1)" if decision == 1 else "Not Ready (0)",
            delta="Optimal Plucking Round" if decision == 1 else "Hold Plucking"
        )
        st.caption(status_label)
    with col2:
        st.metric(
            label="CatBoost Readiness Probability",
            value=f"{prob * 100:.2f}%",
            delta=f"Threshold: 50.0%"
        )
    with col3:
        st.metric(
            label="Estimated Green-Leaf Yield",
            value=f"{estimated_yield:,.1f} kg/ha",
            delta=f"{zone.title()} Zone Projection"
        )

    st.divider()

    left_col, right_col = st.columns([1.25, 1])

    with left_col:
        st.subheader("🔍 Local Explainable AI (SHAP Feature Contributions)")
        st.caption(
            "Positive SHAP values (emerald green) push the model toward **Harvest Ready (1)**; "
            "negative SHAP values (crimson red) push the prediction toward **Not Ready (0)**."
        )
        contrib_df = pd.DataFrame({
            "Feature": list(shap_contribs.keys()),
            "SHAP Value": list(shap_contribs.values()),
            "Input Value": [str(input_dict[k]) for k in shap_contribs.keys()]
        }).sort_values(by="SHAP Value", key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        colors = ["#ef4444" if v < 0 else "#10b981" for v in contrib_df["SHAP Value"]]
        bars = ax.barh(
            contrib_df["Feature"] + " (" + contrib_df["Input Value"] + ")",
            contrib_df["SHAP Value"],
            color=colors
        )
        ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
        ax.set_xlabel("SHAP Contribution (Log-Odds Impact)")
        ax.set_title(f"Why CatBoost Predicted {'Harvest Ready' if decision == 1 else 'Not Ready'} for {region}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right_col:
        st.subheader("📋 Current Sensor Telemetry & SHAP Table")
        table_df = contrib_df.sort_values(by="SHAP Value", key=abs, ascending=False).reset_index(drop=True)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.info(
            "**Agronomic Domain Alignment (TRI Sri Lanka):**  \n"
            "• **NDVI Canopy Index (> 0.55)** indicates active flush density ('two leaves and a bud').  \n"
            "• **Soil Moisture (50%–75%)** & **Rainfall (100–260 mm/month)** sustain sap flow without waterlogging.  \n"
            "• **Temperature extremes (> 30°C)** induce stomatal closure and dormant 'banji' buds."
        )

with tab2:
    st.subheader("📈 Comparative Evaluation: CatBoost (New Algorithm) vs Lecture Baselines")
    st.markdown(
        "Trained using a strict **Temporal Split** (**Train:** 2001–2018 | **Validation:** 2019–2021 | **Out-of-Time Test:** 2022–2024) "
        "to prevent look-ahead data leakage."
    )

    metrics_comp_path = "reports/metrics_comparison.json"
    if os.path.exists(metrics_comp_path):
        with open(metrics_comp_path, "r") as f:
            comp_data = json.load(f)
        comp_df = pd.DataFrame(comp_data).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("reports/roc_curve.png"):
            st.image("reports/roc_curve.png", caption="Comparative ROC Curves on Unseen Test Set (2022–2024)", use_container_width=True)
    with c2:
        if os.path.exists("reports/confusion_matrix.png"):
            st.image("reports/confusion_matrix.png", caption="CatBoost Champion Confusion Matrix (2022–2024)", use_container_width=True)

with tab3:
    st.subheader("🧠 Global Explainability (SHAP Summary Beeswarm & Feature Importance)")
    g1, g2 = st.columns([1.2, 1])
    with g1:
        if os.path.exists("reports/shap_summary.png"):
            st.image("reports/shap_summary.png", caption="Global SHAP Beeswarm Plot across Test Set", use_container_width=True)
    with g2:
        st.markdown("#### Global CatBoost Feature Importance (%)")
        importances = model.get_feature_importance()
        imp_df = pd.DataFrame({
            "Feature": FEATURES_ORDER,
            "Importance (%)": [round(float(x), 2) for x in importances]
        }).sort_values(by="Importance (%)", ascending=False)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

        st.markdown("""
        #### Why CatBoost Was Selected Over Standard Lecture Models:
        1. **Native Ordered Target Statistics:** Handles categorical features (`region`, `zone`, `season`) without high-dimensional one-hot encoding.
        2. **Oblivious (Symmetric) Decision Trees:** Uses identical splitting criteria across an entire tree depth level, acting as a structural regularizer and enabling rapid inference.
        3. **Ordered Boosting:** Prevents target shift and prediction leakage on sequential agro-climatic time-series data.
        """)

with tab4:
    st.subheader("🗂️ Sri Lankan Tea Agro-Climatic Dataset (2001–2024)")
    df_raw = load_dataset()
    if not df_raw.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", f"{len(df_raw):,}")
        m2.metric("Tea Regions", f"{df_raw['region'].nunique()}")
        m3.metric("Year Span", f"{int(df_raw['year'].min())} – {int(df_raw['year'].max())}")
        m4.metric("Features + Target", f"{df_raw.shape[1]}")
        st.dataframe(df_raw.head(50), use_container_width=True)

    st.markdown("""
    ### ⚖️ Critical Discussion, Limitations & Ethical Considerations
    * **Temporal Resolution Limits:** Monthly aggregation captures seasonal monsoon shifts (*Maha* vs *Yala*) but smooths out short-duration extreme weather events (e.g., 48-hour flash floods or frost nights in Nuwara Eliya).
    * **Satellite NDVI Cloud Cover Bias:** Optical satellite NDVI measurements during peak southwest monsoon months can be attenuated by persistent cloud cover in Up-Country estates.
    * **Fairness Across Smallholder vs Corporate Estates:** Model calibration spans all 11 regions across High, Mid, and Low elevations so smallholder farmers in low-grown regions (Ratnapura, Galle, Matara) receive equally accurate guidance as large corporate plantations.
    * **Ethical Data Compliance:** Contains purely environmental, meteorological, and agronomic telemetry—zero personal, private, or worker-identifiable data is collected or processed.
    """)


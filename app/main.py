"""
FastAPI Model Serving API for Agro-Climatic Tea Intelligence.
Includes real-time SHAP Explainable AI and asynchronous telemetry logging.
"""
import os
import time
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from src.data_contract import AgroClimaticFeatures, PredictionResponse
from app.telemetry import log_prediction

app = FastAPI(
    title="Sri Lankan Tea Agro-Climatic Intelligence API",
    description="Harvest Readiness Classification, Yield Estimation, and Explainable AI (CatBoost + SHAP)",
    version="1.0.0"
)

# Enable CORS for Next.js frontend or external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Champion Model and Explainer
MODEL_PATH = "models/champion_model.joblib"
EXPLAINER_PATH = "models/shap_explainer.joblib"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run 'src/train.py' first.")

champion_model = joblib.load(MODEL_PATH)
try:
    import shap
    explainer = joblib.load(EXPLAINER_PATH)
except Exception:
    import shap
    explainer = shap.TreeExplainer(champion_model)

# Automatically use the exact feature order learned during training
FEATURES_ORDER = champion_model.feature_names_


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "sri-lanka-tea-harvest-predictor",
        "model_version": "1.0.0-catboost",
        "timestamp": time.time()
    }


@app.get("/metrics")
def get_metrics():
    metrics_path = "reports/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"message": "Metrics not yet generated."}


@app.get("/explain/global")
def get_global_importance():
    feature_names = champion_model.feature_names_
    importances = champion_model.get_feature_importance()
    return {
        name: round(float(imp), 4)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_harvest(payload: AgroClimaticFeatures):
    start_time = time.time()
    data = payload.model_dump()
    
    # Build single-row DataFrame with exact feature order
    input_row = {k: data[k] for k in FEATURES_ORDER}
    df_single = pd.DataFrame([input_row])

    try:
        # 1. Model Inference
        prob = float(champion_model.predict_proba(df_single)[0][1])
        decision = int(prob >= 0.5)

        # 2. Agro-Climatic Yield Estimation Formula (kg/ha green leaf)
        base_yield = 1750.0 if data["zone"] == "high-grown" else (2100.0 if data["zone"] == "mid-grown" else 2450.0)
        ndvi_factor = 0.5 + 0.6 * data["ndvi_index"]
        rain_factor = 1.1 if (100.0 <= data["rainfall_mm"] <= 260.0) else 0.8
        moisture_factor = 1.05 if (50.0 <= data["soil_moisture_percent"] <= 75.0) else 0.85
        readiness_boost = 1.25 if decision == 1 else 0.75

        estimated_yield = round(base_yield * ndvi_factor * rain_factor * moisture_factor * readiness_boost, 2)

        # 3. Local SHAP Attribution
        shap_vals = explainer(df_single)
        feature_contributions = {
            col: round(float(val), 4)
            for col, val in zip(FEATURES_ORDER, shap_vals.values[0])
        }

        latency_ms = (time.time() - start_time) * 1000

        # 4. Asynchronous Telemetry Log for Azure Log Analytics & Evidently Drift
        log_prediction(data, decision, prob, latency_ms)

        status_text = "READY FOR HARVEST (Peak Flush)" if decision == 1 else "VEGETATIVE / GROWING STAGE"

        return PredictionResponse(
            harvest_ready=decision,
            readiness_probability=round(prob, 4),
            status_label=status_text,
            estimated_yield_kg_per_ha=estimated_yield,
            feature_contributions=feature_contributions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate-drift")
def simulate_drift(num_records: int = 50):
    """Generates synthetic anomalous climate traffic to simulate drift for the MLOps monitoring pipeline."""
    import random
    regions = ["Nuwara Eliya", "Kandy", "Galle", "Ratnapura", "Dimbula"]
    for _ in range(num_records):
        # Simulate severe drought and temperature spike (climate drift)
        record = {
            "region": random.choice(regions),
            "elevation_m": 1200.0,
            "zone": "mid-grown",
            "rainfall_mm": random.uniform(5.0, 30.0),            # severe drought
            "temperature_c": random.uniform(33.0, 39.0),          # heatwave
            "humidity_percent": random.uniform(30.0, 50.0),       # low humidity
            "soil_moisture_percent": random.uniform(12.0, 25.0),  # dry soil
            "ndvi_index": random.uniform(0.25, 0.42),             # low canopy
            "season": "Yala",
            "month": random.randint(1, 12),
            "year": 2026
        }
        df_drift = pd.DataFrame([{k: record[k] for k in FEATURES_ORDER}])
        prob = float(champion_model.predict_proba(df_drift)[0][1])
        log_prediction(record, int(prob >= 0.5), prob, 12.5)

    return {"status": "success", "message": f"Logged {num_records} drifted climate observations to logs/prediction_logs.jsonl"}


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the Agro-Climatic Intelligence Dashboard directly for rapid local testing and demo recordings."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Sri Lankan Tea Agro-Climatic Intelligence</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
      <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; background: #030712; color: #f3f4f6; }
      </style>
    </head>
    <body class="min-h-screen p-6 md:p-10">
      <div class="max-w-6xl mx-auto">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-gray-800 gap-4">
          <div>
            <div class="flex items-center gap-2">
              <span class="inline-block w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></span>
              <span class="text-xs font-semibold tracking-wider text-emerald-400 uppercase">MLOps Production System</span>
            </div>
            <h1 class="text-2xl md:text-3xl font-bold bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent mt-1">
              Agro-Climatic Intelligence System for Sri Lankan Tea
            </h1>
            <p class="text-sm text-gray-400 mt-1">CatBoost Harvest Readiness Classification, Yield Estimation & Explainable AI (SHAP)</p>
          </div>
          <div class="flex gap-2">
            <button onclick="applyPreset('nuwara_eliya')" class="px-3 py-1.5 text-xs bg-gray-900 border border-gray-700 hover:border-emerald-500 rounded-lg transition">Preset: Nuwara Eliya (High)</button>
            <button onclick="applyPreset('galle')" class="px-3 py-1.5 text-xs bg-gray-900 border border-gray-700 hover:border-emerald-500 rounded-lg transition">Preset: Galle (Low)</button>
            <button onclick="triggerDriftSim()" class="px-3 py-1.5 text-xs bg-rose-950/60 border border-rose-800/80 text-rose-300 hover:bg-rose-900 rounded-lg transition">Simulate Drift Drill</button>
          </div>
        </header>

        <!-- Main Layout -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 mt-8">
          <!-- Input Form Column -->
          <div class="lg:col-span-5 bg-gray-900/80 border border-gray-800 p-6 rounded-2xl shadow-xl space-y-4">
            <h2 class="text-base font-semibold text-emerald-400 flex items-center gap-2">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
              Agro-Climatic Sensor Telemetry
            </h2>

            <div>
              <label class="text-xs text-gray-400 block mb-1">Tea Growing Region</label>
              <select id="region" onchange="updateRegionDetails()" class="w-full bg-gray-950 border border-gray-700 rounded-lg p-2.5 text-sm text-gray-200 focus:border-emerald-500 outline-none">
                <option value="Nuwara Eliya">Nuwara Eliya (Up-Country)</option>
                <option value="Dimbula">Dimbula (Up-Country)</option>
                <option value="Haputale">Haputale (Uva High)</option>
                <option value="Badulla">Badulla (Uva Mid)</option>
                <option value="Kandy">Kandy (Mid-Country)</option>
                <option value="Uda Pussellawa">Uda Pussellawa</option>
                <option value="Ella">Ella</option>
                <option value="Galle">Galle (Low-Country)</option>
                <option value="Ratnapura">Ratnapura (Low-Country)</option>
                <option value="Matara">Matara (Low-Country)</option>
                <option value="Kegalle">Kegalle (Low-Country)</option>
              </select>
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="text-xs text-gray-400 block mb-1">Elevation (m)</label>
                <input id="elevation_m" type="number" value="1868" class="w-full bg-gray-950 border border-gray-700 rounded-lg p-2 text-sm text-gray-200">
              </div>
              <div>
                <label class="text-xs text-gray-400 block mb-1">Agro-Ecological Zone</label>
                <input id="zone" type="text" value="high-grown" readonly class="w-full bg-gray-950/60 border border-gray-800 rounded-lg p-2 text-sm text-gray-400">
              </div>
            </div>

            <div>
              <div class="flex justify-between text-xs text-gray-400 mb-1">
                <span>Monthly Rainfall</span>
                <span id="rain_val" class="text-emerald-400 font-mono">167.7 mm</span>
              </div>
              <input id="rainfall_mm" type="range" min="0" max="500" step="1" value="168" oninput="document.getElementById('rain_val').innerText = this.value + ' mm'" class="w-full accent-emerald-500">
            </div>

            <div>
              <div class="flex justify-between text-xs text-gray-400 mb-1">
                <span>NDVI Canopy Flush Index</span>
                <span id="ndvi_val" class="text-emerald-400 font-mono">0.68</span>
              </div>
              <input id="ndvi_index" type="range" min="0.2" max="0.95" step="0.01" value="0.68" oninput="document.getElementById('ndvi_val').innerText = this.value" class="w-full accent-emerald-500">
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <div class="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Temperature</span>
                  <span id="temp_val" class="text-emerald-400 font-mono">14.5°C</span>
                </div>
                <input id="temperature_c" type="range" min="8" max="38" step="0.5" value="14.5" oninput="document.getElementById('temp_val').innerText = this.value + '°C'" class="w-full accent-emerald-500">
              </div>
              <div>
                <div class="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Soil Moisture</span>
                  <span id="moist_val" class="text-emerald-400 font-mono">62%</span>
                </div>
                <input id="soil_moisture_percent" type="range" min="15" max="95" step="1" value="62" oninput="document.getElementById('moist_val').innerText = this.value + '%'" class="w-full accent-emerald-500">
              </div>
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="text-xs text-gray-400 block mb-1">Monsoon Season</label>
                <select id="season" class="w-full bg-gray-950 border border-gray-700 rounded-lg p-2 text-sm text-gray-200">
                  <option value="Maha">Maha (Northeast)</option>
                  <option value="Yala">Yala (Southwest)</option>
                </select>
              </div>
              <div>
                <label class="text-xs text-gray-400 block mb-1">Month</label>
                <input id="month" type="number" min="1" max="12" value="11" class="w-full bg-gray-950 border border-gray-700 rounded-lg p-2 text-sm text-gray-200">
              </div>
            </div>

            <button onclick="runInference()" id="predict_btn" class="w-full py-3 mt-2 bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 font-semibold rounded-xl text-white transition shadow-lg flex items-center justify-center gap-2">
              Classify Readiness & Estimate Yield
            </button>
          </div>

          <!-- Output & XAI Column -->
          <div class="lg:col-span-7 space-y-6">
            <!-- Results Badge Card -->
            <div id="result_card" class="bg-gray-900 border border-gray-800 p-6 rounded-2xl shadow-xl">
              <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <span class="text-xs uppercase tracking-wider text-gray-400">Harvest Readiness Status</span>
                  <h3 id="res_status" class="text-2xl font-bold text-emerald-400 mt-1">READY FOR HARVEST (Peak Flush)</h3>
                  <div class="flex items-center gap-3 mt-2 text-sm text-gray-400">
                    <span>Confidence: <strong id="res_prob" class="text-gray-200 font-mono">99.2%</strong></span>
                    <span>•</span>
                    <span id="res_badge" class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800">Harvest Stage</span>
                  </div>
                </div>
                <div class="bg-gray-950/80 border border-gray-800 p-4 rounded-xl text-right min-w-[160px]">
                  <span class="text-xs uppercase tracking-wider text-gray-400">Estimated Yield</span>
                  <div class="text-3xl font-mono font-bold text-teal-300 mt-1">
                    <span id="res_yield">2,480</span> <span class="text-xs font-sans text-gray-400">kg/ha</span>
                  </div>
                  <span class="text-[11px] text-gray-500">Green leaf projection</span>
                </div>
              </div>
            </div>

            <!-- Explainable AI Card -->
            <div class="bg-gray-900 border border-gray-800 p-6 rounded-2xl shadow-xl">
              <div class="flex justify-between items-center mb-4">
                <h4 class="text-sm font-bold text-gray-200 flex items-center gap-2">
                  <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                  Local Decision Explanation (SHAP Attribution)
                </h4>
                <span class="text-xs text-gray-400">TreeExplainer Shapley Values</span>
              </div>
              <p class="text-xs text-gray-400 mb-4">Green bars drive readiness positive; red bars indicate environmental inhibition.</p>
              
              <div id="shap_bars" class="space-y-3">
                <!-- Dynamically populated -->
              </div>
            </div>

            <!-- System Architecture & Quality Gate -->
            <div class="grid grid-cols-2 gap-4">
              <div class="bg-gray-900/60 border border-gray-800 p-4 rounded-xl">
                <span class="text-xs text-gray-500 uppercase">Champion Model</span>
                <p class="text-sm font-semibold text-gray-300 mt-1">CatBoost Classifier</p>
                <p class="text-xs text-emerald-400 mt-1">Test F1: 0.9903 | AUC: 0.9998</p>
              </div>
              <div class="bg-gray-900/60 border border-gray-800 p-4 rounded-xl">
                <span class="text-xs text-gray-500 uppercase">Monitoring Engine</span>
                <p class="text-sm font-semibold text-gray-300 mt-1">Evidently AI Drift Gate</p>
                <p class="text-xs text-teal-400 mt-1">Prediction logs actively ingested</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <script>
        const REGION_DATA = {
          "Nuwara Eliya": { elev: 1868, zone: "high-grown", temp: 14.5, rain: 168, moist: 62 },
          "Dimbula": { elev: 1450, zone: "high-grown", temp: 18.0, rain: 210, moist: 65 },
          "Haputale": { elev: 1420, zone: "high-grown", temp: 17.5, rain: 155, moist: 58 },
          "Badulla": { elev: 680, zone: "mid-grown", temp: 22.5, rain: 130, moist: 55 },
          "Kandy": { elev: 500, zone: "mid-grown", temp: 24.0, rain: 175, moist: 68 },
          "Uda Pussellawa": { elev: 1300, zone: "high-grown", temp: 18.2, rain: 145, moist: 60 },
          "Ella": { elev: 1041, zone: "mid-grown", temp: 20.5, rain: 140, moist: 59 },
          "Galle": { elev: 15, zone: "low-grown", temp: 27.5, rain: 240, moist: 75 },
          "Ratnapura": { elev: 130, zone: "low-grown", temp: 28.0, rain: 310, moist: 82 },
          "Matara": { elev: 20, zone: "low-grown", temp: 27.8, rain: 195, moist: 72 },
          "Kegalle": { elev: 116, zone: "low-grown", temp: 26.5, rain: 230, moist: 70 }
        };

        function updateRegionDetails() {
          const reg = document.getElementById("region").value;
          const info = REGION_DATA[reg];
          if (info) {
            document.getElementById("elevation_m").value = info.elev;
            document.getElementById("zone").value = info.zone;
          }
        }

        function applyPreset(name) {
          if (name === 'nuwara_eliya') {
            document.getElementById("region").value = "Nuwara Eliya";
            document.getElementById("elevation_m").value = 1868;
            document.getElementById("zone").value = "high-grown";
            document.getElementById("rainfall_mm").value = 168;
            document.getElementById("rain_val").innerText = "168 mm";
            document.getElementById("temperature_c").value = 14.5;
            document.getElementById("temp_val").innerText = "14.5°C";
            document.getElementById("soil_moisture_percent").value = 62;
            document.getElementById("moist_val").innerText = "62%";
            document.getElementById("ndvi_index").value = 0.68;
            document.getElementById("ndvi_val").innerText = "0.68";
            document.getElementById("season").value = "Maha";
          } else {
            document.getElementById("region").value = "Galle";
            document.getElementById("elevation_m").value = 15;
            document.getElementById("zone").value = "low-grown";
            document.getElementById("rainfall_mm").value = 240;
            document.getElementById("rain_val").innerText = "240 mm";
            document.getElementById("temperature_c").value = 27.5;
            document.getElementById("temp_val").innerText = "27.5°C";
            document.getElementById("soil_moisture_percent").value = 75;
            document.getElementById("moist_val").innerText = "75%";
            document.getElementById("ndvi_index").value = 0.58;
            document.getElementById("ndvi_val").innerText = "0.58";
            document.getElementById("season").value = "Yala";
          }
          runInference();
        }

        async function runInference() {
          const btn = document.getElementById("predict_btn");
          btn.innerText = "Analyzing Telemetry & SHAP...";
          btn.disabled = true;

          const payload = {
            region: document.getElementById("region").value,
            elevation_m: parseFloat(document.getElementById("elevation_m").value),
            zone: document.getElementById("zone").value,
            rainfall_mm: parseFloat(document.getElementById("rainfall_mm").value),
            temperature_c: parseFloat(document.getElementById("temperature_c").value),
            humidity_percent: 82.0,
            soil_moisture_percent: parseFloat(document.getElementById("soil_moisture_percent").value),
            ndvi_index: parseFloat(document.getElementById("ndvi_index").value),
            season: document.getElementById("season").value,
            month: parseInt(document.getElementById("month").value),
            year: 2026
          };

          try {
            const resp = await fetch("/predict", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload)
            });
            const data = await resp.json();

            // Update Prediction Display
            const isReady = data.harvest_ready === 1;
            document.getElementById("res_status").innerText = data.status_label;
            document.getElementById("res_status").className = `text-2xl font-bold mt-1 ${isReady ? 'text-emerald-400' : 'text-amber-400'}`;
            document.getElementById("res_prob").innerText = (data.readiness_probability * 100).toFixed(1) + "%";
            document.getElementById("res_yield").innerText = data.estimated_yield_kg_per_ha.toLocaleString();

            const badge = document.getElementById("res_badge");
            badge.innerText = isReady ? "Optimal Plucking Round" : "Vegetative Stage";
            badge.className = `px-2.5 py-0.5 rounded-full text-xs font-semibold ${isReady ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-amber-950 text-amber-300 border border-amber-800'}`;

            // Render SHAP Bars
            const container = document.getElementById("shap_bars");
            container.innerHTML = "";
            const sortedFeats = Object.entries(data.feature_contributions).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]));

            sortedFeats.forEach(([feat, val]) => {
              const isPositive = val >= 0;
              const widthPct = Math.min(Math.abs(val) * 75, 100);
              const barHtml = `
                <div class="text-xs">
                  <div class="flex justify-between text-gray-400 mb-1">
                    <span class="capitalize">${feat.replace('_', ' ')}</span>
                    <span class="font-mono ${isPositive ? 'text-emerald-400' : 'text-rose-400'}">
                      ${isPositive ? '+' : ''}${val.toFixed(4)}
                    </span>
                  </div>
                  <div class="w-full bg-gray-950 h-2 rounded-full overflow-hidden flex">
                    <div class="h-full ${isPositive ? 'bg-emerald-500 ml-auto' : 'bg-rose-500'}" style="width: ${widthPct}%;"></div>
                  </div>
                </div>
              `;
              container.innerHTML += barHtml;
            });
          } catch (e) {
            alert("Inference Error: " + e.message);
          } finally {
            btn.innerText = "Classify Readiness & Estimate Yield";
            btn.disabled = false;
          }
        }

        async function triggerDriftSim() {
          const res = await fetch("/simulate-drift?num_records=40", { method: "POST" });
          const json = await res.json();
          alert("Drift Drill: " + json.message);
        }

        // Initialize on load
        window.onload = runInference;
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

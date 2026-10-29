"use client";
import React, { useState, useEffect } from "react";
import ShapVisualizer from "../components/ShapVisualizer";
import {
  Sprout,
  Activity,
  AlertTriangle,
  RefreshCw,
  BarChart2,
  CheckCircle2,
  CloudRain,
  Sliders,
  ShieldCheck,
} from "lucide-react";

const REGIONS = [
  { name: "Nuwara Eliya", elev: 1868, zone: "high-grown", temp: 14.5, rain: 168, moist: 62 },
  { name: "Dimbula", elev: 1450, zone: "high-grown", temp: 18.0, rain: 210, moist: 65 },
  { name: "Haputale", elev: 1420, zone: "high-grown", temp: 17.5, rain: 155, moist: 58 },
  { name: "Badulla", elev: 680, zone: "mid-grown", temp: 22.5, rain: 130, moist: 55 },
  { name: "Kandy", elev: 500, zone: "mid-grown", temp: 24.0, rain: 175, moist: 68 },
  { name: "Uda Pussellawa", elev: 1300, zone: "high-grown", temp: 18.2, rain: 145, moist: 60 },
  { name: "Ella", elev: 1041, zone: "mid-grown", temp: 20.5, rain: 140, moist: 59 },
  { name: "Galle", elev: 15, zone: "low-grown", temp: 27.5, rain: 240, moist: 75 },
  { name: "Ratnapura", elev: 130, zone: "low-grown", temp: 28.0, rain: 310, moist: 82 },
  { name: "Matara", elev: 20, zone: "low-grown", temp: 27.8, rain: 195, moist: 72 },
  { name: "Kegalle", elev: 116, zone: "low-grown", temp: 26.5, rain: 230, moist: 70 },
];

export default function Home() {
  const [formData, setFormData] = useState({
    region: "Nuwara Eliya",
    elevation_m: 1868,
    zone: "high-grown",
    rainfall_mm: 168,
    temperature_c: 14.5,
    humidity_percent: 85.0,
    soil_moisture_percent: 62.0,
    ndvi_index: 0.68,
    season: "Maha",
    month: 11,
    year: 2026,
  });

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [driftMsg, setDriftMsg] = useState<string | null>(null);

  const handleRegionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const reg = REGIONS.find((r) => r.name === e.target.value);
    if (reg) {
      setFormData((prev) => ({
        ...prev,
        region: reg.name,
        elevation_m: reg.elev,
        zone: reg.zone,
        temperature_c: reg.temp,
        rainfall_mm: reg.rain,
        soil_moisture_percent: reg.moist,
      }));
    }
  };

  const setPreset = (presetName: "nuwara" | "galle") => {
    if (presetName === "nuwara") {
      setFormData({
        region: "Nuwara Eliya",
        elevation_m: 1868,
        zone: "high-grown",
        rainfall_mm: 168,
        temperature_c: 14.5,
        humidity_percent: 86.0,
        soil_moisture_percent: 62.0,
        ndvi_index: 0.68,
        season: "Maha",
        month: 11,
        year: 2026,
      });
    } else {
      setFormData({
        region: "Galle",
        elevation_m: 15,
        zone: "low-grown",
        rainfall_mm: 240,
        temperature_c: 27.5,
        humidity_percent: 82.0,
        soil_moisture_percent: 74.0,
        ndvi_index: 0.58,
        season: "Yala",
        month: 6,
        year: 2026,
      });
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      alert("Failed to connect to FastAPI backend: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSimulateDrift = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/simulate-drift?num_records=40`, {
        method: "POST",
      });
      const data = await res.json();
      setDriftMsg(data.message);
      setTimeout(() => setDriftMsg(null), 5000);
    } catch (err: any) {
      alert("Error triggering drift: " + err.message);
    }
  };

  useEffect(() => {
    handlePredict();
  }, []);

  return (
    <main className="min-h-screen p-6 md:p-12 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-semibold tracking-wider text-emerald-400 uppercase">
              MLOps Production Deployment
            </span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">
            Agro-Climatic Intelligence: Sri Lankan Tea Plantations
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            CatBoost Harvest Readiness Classifier · Yield Estimator · Real-Time SHAP XAI
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setPreset("nuwara")}
            className="px-3 py-1.5 text-xs bg-slate-900 border border-slate-700 hover:border-emerald-500 rounded-lg transition"
          >
            Preset: Nuwara Eliya (High)
          </button>
          <button
            onClick={() => setPreset("galle")}
            className="px-3 py-1.5 text-xs bg-slate-900 border border-slate-700 hover:border-emerald-500 rounded-lg transition"
          >
            Preset: Galle (Low)
          </button>
          <button
            onClick={handleSimulateDrift}
            className="px-3 py-1.5 text-xs bg-rose-950/60 border border-rose-800 text-rose-300 hover:bg-rose-900 rounded-lg transition flex items-center gap-1"
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            Simulate Drift Drill
          </button>
        </div>
      </header>

      {driftMsg && (
        <div className="p-3 bg-amber-950/70 border border-amber-600/60 text-amber-300 text-xs rounded-xl flex items-center gap-2 animate-bounce">
          <AlertTriangle className="w-4 h-4" />
          <span>{driftMsg}</span>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Input Form Controls */}
        <section className="lg:col-span-5 bg-slate-900/70 border border-slate-800 p-6 rounded-2xl space-y-5 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-emerald-400 flex items-center gap-2">
              <Sliders className="w-4 h-4" /> Agro-Meteorological Inputs
            </h2>
            <span className="text-[11px] text-slate-500">Live Telemetry</span>
          </div>

          <div>
            <label className="text-xs text-slate-400 block mb-1">Tea Region</label>
            <select
              value={formData.region}
              onChange={handleRegionChange}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl p-2.5 text-sm focus:border-emerald-500 outline-none"
            >
              {REGIONS.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name} ({r.zone})
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Elevation (m)</label>
              <input
                type="number"
                value={formData.elevation_m}
                onChange={(e) =>
                  setFormData({ ...formData, elevation_m: parseFloat(e.target.value) || 0 })
                }
                className="w-full bg-slate-950 border border-slate-700 rounded-xl p-2 text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Zone Category</label>
              <input
                type="text"
                value={formData.zone}
                readOnly
                className="w-full bg-slate-950/60 border border-slate-800 rounded-xl p-2 text-sm text-slate-400 cursor-not-allowed"
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Monthly Rainfall (mm)</span>
              <span className="text-emerald-400 font-mono font-semibold">
                {formData.rainfall_mm} mm
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="500"
              value={formData.rainfall_mm}
              onChange={(e) =>
                setFormData({ ...formData, rainfall_mm: parseFloat(e.target.value) })
              }
              className="w-full accent-emerald-500 cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>NDVI Canopy Flush Density</span>
              <span className="text-emerald-400 font-mono font-semibold">
                {formData.ndvi_index}
              </span>
            </div>
            <input
              type="range"
              min="0.2"
              max="0.95"
              step="0.01"
              value={formData.ndvi_index}
              onChange={(e) =>
                setFormData({ ...formData, ndvi_index: parseFloat(e.target.value) })
              }
              className="w-full accent-emerald-500 cursor-pointer"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Temperature</span>
                <span className="text-emerald-400 font-mono font-semibold">
                  {formData.temperature_c}°C
                </span>
              </div>
              <input
                type="range"
                min="8"
                max="38"
                step="0.5"
                value={formData.temperature_c}
                onChange={(e) =>
                  setFormData({ ...formData, temperature_c: parseFloat(e.target.value) })
                }
                className="w-full accent-emerald-500 cursor-pointer"
              />
            </div>
            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Soil Moisture</span>
                <span className="text-emerald-400 font-mono font-semibold">
                  {formData.soil_moisture_percent}%
                </span>
              </div>
              <input
                type="range"
                min="15"
                max="95"
                step="1"
                value={formData.soil_moisture_percent}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    soil_moisture_percent: parseFloat(e.target.value),
                  })
                }
                className="w-full accent-emerald-500 cursor-pointer"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Monsoon Season</label>
              <select
                value={formData.season}
                onChange={(e: any) => setFormData({ ...formData, season: e.target.value })}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl p-2 text-sm"
              >
                <option value="Maha">Maha (Northeast)</option>
                <option value="Yala">Yala (Southwest)</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Month</label>
              <input
                type="number"
                min="1"
                max="12"
                value={formData.month}
                onChange={(e) =>
                  setFormData({ ...formData, month: parseInt(e.target.value) || 1 })
                }
                className="w-full bg-slate-950 border border-slate-700 rounded-xl p-2 text-sm"
              />
            </div>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full py-3 bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 font-semibold rounded-xl text-white transition shadow-lg flex items-center justify-center gap-2"
          >
            {loading ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Sprout className="w-4 h-4" />
            )}
            {loading ? "Computing with Explainable AI..." : "Classify Readiness & Estimate Yield"}
          </button>
        </section>

        {/* Right: Results & XAI */}
        <section className="lg:col-span-7 space-y-6">
          {/* Prediction Status Card */}
          {result && (
            <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <span className="text-xs uppercase tracking-wider text-slate-400">
                  Harvest Readiness Classification
                </span>
                <h3
                  className={`text-2xl font-bold mt-1 ${
                    result.harvest_ready === 1 ? "text-emerald-400" : "text-amber-400"
                  }`}
                >
                  {result.status_label}
                </h3>
                <div className="flex items-center gap-3 mt-2 text-sm text-slate-400">
                  <span>
                    Confidence:{" "}
                    <strong className="text-slate-200 font-mono">
                      {(result.readiness_probability * 100).toFixed(1)}%
                    </strong>
                  </span>
                  <span>•</span>
                  <span
                    className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                      result.harvest_ready === 1
                        ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
                        : "bg-amber-950 text-amber-300 border border-amber-800"
                    }`}
                  >
                    {result.harvest_ready === 1 ? "Optimal Plucking Round" : "Vegetative Stage"}
                  </span>
                </div>
              </div>

              <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl text-right min-w-[170px]">
                <span className="text-xs uppercase tracking-wider text-slate-400">
                  Estimated Yield
                </span>
                <div className="text-3xl font-mono font-bold text-teal-300 mt-1">
                  {result.estimated_yield_kg_per_ha.toLocaleString()}{" "}
                  <span className="text-xs font-sans text-slate-400">kg/ha</span>
                </div>
                <span className="text-[11px] text-slate-500">Green leaf projection</span>
              </div>
            </div>
          )}

          {/* Explainable AI SHAP Card */}
          <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl">
            <div className="flex justify-between items-center mb-4">
              <h4 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-emerald-400" />
                Local Decision Explanation (SHAP Attribution)
              </h4>
              <span className="text-xs text-slate-400">TreeExplainer Shapley Values</span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Green bars drive the harvest readiness probability upwards; red bars indicate environmental inhibition.
            </p>
            {result && <ShapVisualizer contributions={result.feature_contributions} />}
          </div>

          {/* MLOps Governance & Drift Monitor Card */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
              <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold uppercase">
                <ShieldCheck className="w-4 h-4" /> Champion Algorithm
              </div>
              <p className="text-sm font-semibold text-slate-200 mt-1">
                CatBoost (Oblivious Trees)
              </p>
              <p className="text-xs text-slate-400 mt-1">Test F1: 0.9903 | Test AUC: 0.9998</p>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
              <div className="flex items-center gap-2 text-teal-400 text-xs font-semibold uppercase">
                <Activity className="w-4 h-4" /> Monitoring Engine
              </div>
              <p className="text-sm font-semibold text-slate-200 mt-1">
                Evidently AI Drift Gate
              </p>
              <p className="text-xs text-slate-400 mt-1">Prediction logs actively ingested</p>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

"use client";
import React from "react";

interface ShapProps {
  contributions: Record<string, number>;
}

export default function ShapVisualizer({ contributions }: ShapProps) {
  if (!contributions || Object.keys(contributions).length === 0) {
    return (
      <div className="text-slate-500 text-xs italic">
        Awaiting prediction to compute local SHAP Shapley values...
      </div>
    );
  }

  const sorted = Object.entries(contributions).sort(
    (a, b) => Math.abs(b[1]) - Math.abs(a[1])
  );

  return (
    <div className="space-y-3">
      {sorted.map(([feature, val]) => {
        const isPositive = val >= 0;
        const widthPct = Math.min(Math.abs(val) * 80, 100);

        return (
          <div key={feature} className="text-xs">
            <div className="flex justify-between text-slate-300 mb-1">
              <span className="capitalize font-medium">
                {feature.replace(/_/g, " ")}
              </span>
              <span
                className={`font-mono ${
                  isPositive ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {isPositive ? `+${val.toFixed(4)}` : val.toFixed(4)}
              </span>
            </div>
            <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden flex border border-slate-800">
              <div
                className={`h-full transition-all duration-500 ${
                  isPositive ? "bg-emerald-500 ml-auto" : "bg-rose-500"
                }`}
                style={{ width: `${widthPct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

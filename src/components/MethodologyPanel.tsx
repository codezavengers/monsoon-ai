import React from 'react';
import { BookOpen, ShieldAlert, Cpu, Database, CheckCircle2, GitBranch } from 'lucide-react';

export const MethodologyPanel: React.FC = () => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl space-y-6 text-slate-200">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="px-2 py-0.5 text-[10px] font-bold tracking-widest bg-sky-500/10 text-sky-400 border border-sky-500/30 rounded uppercase">
            MEGHDRISTI Neural Core
          </span>
          <span className="text-[10px] text-slate-400 font-mono">v2.4-JJAS</span>
        </div>
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-sky-400" />
          MEGHDRISTI Neural Architecture & Scientific Methodology
        </h3>
        <p className="text-xs text-slate-400 mt-1">
          Problem Statement ID: 26080 — Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts
        </p>
      </div>

      {/* Pipeline Diagram */}
      <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-3">
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-emerald-400" />
          Data & ML Processing Pipeline
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
            <span className="font-bold text-sky-400 block mb-1">1. Ingestion & QC</span>
            <p className="text-slate-400 text-[11px]">
              Raw NWP forecasts, physical boundary checks, non-negative clipping, and unit standardization.
            </p>
          </div>
          <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
            <span className="font-bold text-purple-400 block mb-1">2. Regime Identification</span>
            <p className="text-slate-400 text-[11px]">
              Supervised Random Forest classifies synoptic state into 8 regimes (Active, Break, Orographic, Depression, etc.).
            </p>
          </div>
          <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
            <span className="font-bold text-emerald-400 block mb-1">3. Regime-Aware ML</span>
            <p className="text-slate-400 text-[11px]">
              Samples routed to specialized regime models. Inverts log1p transforms and enforces non-negative constraints.
            </p>
          </div>
          <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
            <span className="font-bold text-amber-400 block mb-1">4. Verification & Alerts</span>
            <p className="text-slate-400 text-[11px]">
              Calibrated exceedance probabilities, district spatial aggregation, and continuous/categorical verification metrics.
            </p>
          </div>
        </div>
      </div>

      {/* Operational Thresholds */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
          <h4 className="font-semibold text-white mb-2 flex items-center gap-1.5">
            <Cpu className="w-4 h-4 text-sky-400" />
            Configurable Operational Thresholds (IMD Standards)
          </h4>
          <ul className="space-y-1.5 text-slate-300">
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span>Light Rainfall:</span>
              <span className="font-mono text-slate-400">2.5 – 15.5 mm/day</span>
            </li>
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span>Moderate Rainfall:</span>
              <span className="font-mono text-slate-400">15.6 – 64.4 mm/day</span>
            </li>
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span>Heavy Rainfall (Yellow Alert):</span>
              <span className="font-mono text-amber-400 font-bold">≥ 64.5 mm/day</span>
            </li>
            <li className="flex justify-between py-1 border-b border-slate-800/80">
              <span>Very Heavy Rainfall (Orange Alert):</span>
              <span className="font-mono text-rose-400 font-bold">≥ 115.6 mm/day</span>
            </li>
            <li className="flex justify-between py-1">
              <span>Extremely Heavy Rainfall (Red Alert):</span>
              <span className="font-mono text-purple-400 font-bold">≥ 204.5 mm/day</span>
            </li>
          </ul>
        </div>

        <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
          <h4 className="font-semibold text-white mb-2 flex items-center gap-1.5">
            <Database className="w-4 h-4 text-emerald-400" />
            Verification Formulation
          </h4>
          <ul className="space-y-1.5 text-slate-300 text-[11px] leading-relaxed">
            <li>
              <strong>RMSE:</strong> <code className="text-sky-300 font-mono text-[10px]">sqrt(mean((Forecast - Observation)²))</code> — penalizes large intensity misjudgments.
            </li>
            <li>
              <strong>CSI (Critical Success Index):</strong> <code className="text-emerald-300 font-mono text-[10px]">Hits / (Hits + Misses + False Alarms)</code> — evaluates extreme event detection skill.
            </li>
            <li>
              <strong>ETS (Equitable Threat Score):</strong> Deducts hits attributable to random climatological chance.
            </li>
            <li>
              <strong>FSS (Fractions Skill Score):</strong> Roberts & Lean (2008) spatial scale-dependent evaluation across 1x1, 3x3, 5x5, and 7x7 grid windows.
            </li>
          </ul>
        </div>
      </div>

      {/* Scientific Integrity Disclaimer */}
      <div className="bg-amber-950/20 border border-amber-800/40 p-4 rounded-xl text-xs text-amber-200/90 flex items-start gap-3">
        <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h5 className="font-bold text-amber-300">Scientific Integrity Notice</h5>
          <p className="text-[11px] leading-relaxed text-amber-200/80">
            This deployment is currently operating in <strong>Demonstration Mode</strong> using physically realistic synthetic meteorological time-series modeled after Indian monsoon synoptic climatology (2018–2024). Operational thresholds are aligned with official India Meteorological Department (IMD) standards. Explainability attributions reflect statistical model gradient weights and should not be construed as direct physical causality.
          </p>
        </div>
      </div>
    </div>
  );
};

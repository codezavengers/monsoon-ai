import React, { useState, useEffect } from 'react';
import { ModelMonitoringPayload, FeatureDriftMetric } from '../types';
import { 
  AlertTriangle, 
  CheckCircle2, 
  Activity, 
  RefreshCw, 
  TrendingUp, 
  TrendingDown, 
  ShieldAlert, 
  ShieldCheck, 
  Sliders, 
  Info, 
  Zap, 
  CloudRain, 
  Thermometer, 
  Droplets, 
  Gauge, 
  Wind, 
  ArrowUpRight,
  Database
} from 'lucide-react';

interface ModelMonitoringProps {
  currentDate?: string;
  leadTime?: string;
  nwpProvider?: string;
}

export function ModelMonitoring({
  currentDate = '2024-07-15',
  leadTime = '24',
  nwpProvider = 'GFS 0.25° (NOAA)'
}: ModelMonitoringProps) {
  const [data, setData] = useState<ModelMonitoringPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [simulationMode, setSimulationMode] = useState<'none' | 'moderate' | 'extreme'>('none');
  const [categoryFilter, setCategoryFilter] = useState<'all' | 'precipitation' | 'thermodynamic' | 'kinematic' | 'surface'>('all');
  const [activeDate, setActiveDate] = useState(currentDate);

  const fetchMonitoringData = async (dateVal = activeDate, simVal = simulationMode) => {
    try {
      setLoading(true);
      const cleanLead = leadTime.replace('+', '').replace('h', '');
      const url = `/api/model/monitoring?date=${encodeURIComponent(dateVal)}&lead_time=${encodeURIComponent(cleanLead)}&provider=${encodeURIComponent(nwpProvider)}&simulate_drift=${encodeURIComponent(simVal)}`;
      const res = await fetch(url);
      const json = await res.json();
      if (json.success && json.data) {
        setData(json.data);
      }
    } catch (e) {
      console.error('Failed to load model monitoring data:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMonitoringData(activeDate, simulationMode);
  }, [activeDate, simulationMode, leadTime, nwpProvider]);

  const summary = data?.summary;
  const features = data?.features || [];

  const filteredFeatures = categoryFilter === 'all' 
    ? features 
    : features.filter(f => f.category === categoryFilter);

  const isCriticalDrift = summary?.overall_status === 'SIGNIFICANT_DRIFT_ALERT';
  const isModerateDrift = summary?.overall_status === 'MODERATE_DRIFT';

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'precipitation':
        return <CloudRain className="w-4 h-4 text-sky-400" />;
      case 'thermodynamic':
        return <Thermometer className="w-4 h-4 text-amber-400" />;
      case 'kinematic':
        return <Wind className="w-4 h-4 text-indigo-400" />;
      case 'surface':
        return <Gauge className="w-4 h-4 text-emerald-400" />;
      default:
        return <Activity className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Controls & Simulation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900 border border-slate-800 p-4 rounded-xl">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl flex items-center justify-center border shadow-lg ${
            isCriticalDrift 
              ? 'bg-rose-500/10 border-rose-500/30 text-rose-400 shadow-rose-500/10' 
              : isModerateDrift 
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-400 shadow-amber-500/10' 
              : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 shadow-emerald-500/10'
          }`}>
            {isCriticalDrift ? (
              <ShieldAlert className="w-6 h-6 animate-pulse" />
            ) : isModerateDrift ? (
              <AlertTriangle className="w-6 h-6" />
            ) : (
              <ShieldCheck className="w-6 h-6" />
            )}
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <span>MEGHDRISHTI NWP Distribution Drift Monitor</span>
              <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded border text-cyan-400 border-cyan-800 bg-cyan-950/60">
                Baseline: 2018–2022 JJAS
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Continuously audits mean shift and variance ratios of incoming numerical weather prediction forecasts
            </p>
          </div>
        </div>

        {/* Date Selector & Interactive Drift Stress Test */}
        <div className="flex flex-wrap items-center gap-2.5 text-xs">
          <div className="flex items-center gap-1.5 bg-slate-950 px-2.5 py-1.5 rounded-lg border border-slate-800">
            <span className="text-slate-400">Cycle Date:</span>
            <select
              value={activeDate}
              onChange={(e) => setActiveDate(e.target.value)}
              className="bg-transparent text-white font-medium focus:outline-none cursor-pointer"
            >
              <option value="2024-07-15" className="bg-slate-900">15 July 2024 (Active Spell)</option>
              <option value="2024-08-03" className="bg-slate-900">03 August 2024 (Depression)</option>
              <option value="2024-08-20" className="bg-slate-900">20 August 2024 (Break Spell)</option>
            </select>
          </div>

          {/* Drift Simulation Stress Test */}
          <div className="flex items-center gap-1.5 bg-slate-950 px-2.5 py-1.5 rounded-lg border border-slate-800">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-slate-400">Stress Test Drift:</span>
            <select
              value={simulationMode}
              onChange={(e) => setSimulationMode(e.target.value as any)}
              className="bg-transparent text-white font-medium focus:outline-none cursor-pointer"
            >
              <option value="none" className="bg-slate-900">Normal Benchmark (In-Distribution)</option>
              <option value="moderate" className="bg-slate-900">Moderate Convective Shift (Warning)</option>
              <option value="extreme" className="bg-slate-900">Extreme Cyclonic Storm (Trigger Alert)</option>
            </select>
          </div>

          <button
            type="button"
            onClick={() => fetchMonitoringData(activeDate, simulationMode)}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold rounded-lg border border-slate-700 transition disabled:opacity-50 cursor-pointer"
            title="Re-run statistical drift audit"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Audit</span>
          </button>
        </div>
      </div>

      {/* Prominent High-Visibility Alert Banner When Drift Is Detected */}
      {isCriticalDrift && (
        <div className="bg-rose-950/80 border-2 border-rose-600/80 text-rose-100 rounded-xl p-4 shadow-xl backdrop-blur-md animate-in fade-in slide-in-from-top-2 duration-200">
          <div className="flex items-start gap-3.5">
            <div className="p-2 rounded-lg bg-rose-600 text-white shrink-0 mt-0.5 shadow-md">
              <ShieldAlert className="w-6 h-6 animate-pulse" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-bold uppercase tracking-wider text-rose-200 flex items-center gap-2">
                  <span>Input Data Distribution Shift Detected</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-900 border border-rose-500 text-rose-100">
                    CRITICAL DRIFT ALERT
                  </span>
                </h3>
                <span className="text-xs font-mono text-rose-300 font-semibold">
                  Max Z-Score: {summary?.max_z_score}σ · Max Variance Ratio: {summary?.max_variance_ratio}x
                </span>
              </div>
              <p className="text-xs text-rose-200/90 mt-1 leading-relaxed">
                {summary?.alert_summary}
              </p>
              
              <div className="mt-3 pt-2.5 border-t border-rose-800/80 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="font-semibold text-rose-300">Operational Safeguards Activated:</span>
                  <ul className="list-disc list-inside text-rose-200/90 text-[11px] mt-1 space-y-0.5">
                    {summary?.recommended_actions.map((act, idx) => (
                      <li key={idx}>{act}</li>
                    ))}
                  </ul>
                </div>
                <div className="bg-rose-900/40 p-2.5 rounded-lg border border-rose-800/60">
                  <span className="font-semibold text-rose-300 text-[11px] block">Meteorological Assessment:</span>
                  <p className="text-[11px] text-rose-200/80 mt-0.5 leading-tight">
                    Severe variance expansion ({summary?.max_variance_ratio}x baseline) indicates uncharacteristic synoptic forcing. Defaulting AI inference to soft mixture regime routing with widened prediction intervals.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {isModerateDrift && !isCriticalDrift && (
        <div className="bg-amber-950/70 border border-amber-600/80 text-amber-100 rounded-xl p-4 shadow-md backdrop-blur-md">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-amber-200">
                  Moderate Synoptic Distribution Shift
                </h3>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-900/60 border border-amber-600 text-amber-200">
                  MODERATE SHIFT
                </span>
              </div>
              <p className="text-xs text-amber-200/90 mt-0.5">
                {summary?.alert_summary}
              </p>
            </div>
          </div>
        </div>
      )}

      {!isCriticalDrift && !isModerateDrift && (
        <div className="bg-emerald-950/40 border border-emerald-800/80 text-emerald-200 rounded-xl p-3.5 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <div>
              <span className="font-bold text-xs text-emerald-100">NWP Ingest Statistics In-Distribution</span>
              <p className="text-[11px] text-emerald-300/80">
                All 7 primary meteorological input variables align with the 2018–2022 JJAS training bounds.
              </p>
            </div>
          </div>
          <span className="text-xs font-mono font-semibold text-emerald-300 bg-emerald-950 px-2 py-1 rounded border border-emerald-800">
            Health Score: 100/100
          </span>
        </div>
      )}

      {/* KPI Cards: Health, Z-Score, Variance Ratio, and OOD Count */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <div className="text-[11px] text-slate-400 font-medium">Input Health Score</div>
          <div className="flex items-baseline gap-2 mt-1">
            <span className={`text-2xl font-bold font-mono tabular-nums ${
              (summary?.overall_health_score ?? 100) < 60 
                ? 'text-rose-400' 
                : (summary?.overall_health_score ?? 100) < 85 
                ? 'text-amber-400' 
                : 'text-emerald-400'
            }`}>
              {summary?.overall_health_score ?? 100}
            </span>
            <span className="text-xs text-slate-400">/ 100</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            {summary?.drift_detected ? 'Deviations flagged' : 'Nominal alignment'}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <div className="text-[11px] text-slate-400 font-medium">Max Mean Shift (Z-Score)</div>
          <div className="flex items-baseline gap-1.5 mt-1">
            <span className={`text-2xl font-bold font-mono tabular-nums ${
              (summary?.max_z_score ?? 0) >= 2.5 
                ? 'text-rose-400' 
                : (summary?.max_z_score ?? 0) >= 1.4 
                ? 'text-amber-400' 
                : 'text-slate-200'
            }`}>
              {summary?.max_z_score ?? 0}
            </span>
            <span className="text-xs font-mono text-slate-400">σ</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Critical Threshold: $\ge 2.5\sigma$
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <div className="text-[11px] text-slate-400 font-medium">Max Variance Ratio (F)</div>
          <div className="flex items-baseline gap-1.5 mt-1">
            <span className={`text-2xl font-bold font-mono tabular-nums ${
              (summary?.max_variance_ratio ?? 1) > 3.0 || (summary?.max_variance_ratio ?? 1) < 0.3
                ? 'text-rose-400' 
                : (summary?.max_variance_ratio ?? 1) > 2.0 || (summary?.max_variance_ratio ?? 1) < 0.5 
                ? 'text-amber-400' 
                : 'text-slate-200'
            }`}>
              {summary?.max_variance_ratio ?? 1}x
            </span>
            <span className="text-xs text-slate-400">ratio</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Bounds: 0.5x – 2.0x baseline
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl">
          <div className="text-[11px] text-slate-400 font-medium">Distribution Status Count</div>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-emerald-400 font-bold font-mono text-lg">{summary?.stable_features_count ?? 7}</span>
            <span className="text-slate-400 text-xs">In</span>
            <span className="text-slate-600">·</span>
            <span className="text-amber-400 font-bold font-mono text-lg">{summary?.warning_features_count ?? 0}</span>
            <span className="text-slate-400 text-xs">Warn</span>
            <span className="text-slate-600">·</span>
            <span className="text-rose-400 font-bold font-mono text-lg">{summary?.critical_features_count ?? 0}</span>
            <span className="text-slate-400 text-xs">Crit</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-1">
            Sample: {summary?.sample_size_districts ?? 729} districts analyzed
          </div>
        </div>
      </div>

      {/* Category Filter Tabs */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-1.5 text-xs">
          <span className="text-slate-400 text-[11px] mr-1">Filter Domain:</span>
          {(['all', 'precipitation', 'thermodynamic', 'kinematic', 'surface'] as const).map((cat) => (
            <button
              key={cat}
              onClick={() => setCategoryFilter(cat)}
              className={`px-2.5 py-1 rounded-md font-medium transition cursor-pointer capitalize ${
                categoryFilter === cat
                  ? 'bg-sky-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
        <span className="text-[11px] text-slate-400 font-mono hidden sm:inline">
          NWP: {nwpProvider} · +{leadTime}h
        </span>
      </div>

      {/* Detailed Statistical Comparison Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredFeatures.map((feat) => {
          const isCritical = feat.drift_status === 'CRITICAL_DRIFT';
          const isWarning = feat.drift_status === 'MODERATE_SHIFT';

          return (
            <div
              key={feat.feature_key}
              className={`p-4 rounded-xl border transition-all ${
                isCritical
                  ? 'bg-rose-950/20 border-rose-800 shadow-md ring-1 ring-rose-500/20'
                  : isWarning
                  ? 'bg-amber-950/20 border-amber-800/80 shadow-sm'
                  : 'bg-slate-900 border-slate-800 hover:border-slate-700'
              }`}
            >
              {/* Feature Header */}
              <div className="flex items-start justify-between gap-2 border-b border-slate-800/80 pb-2.5 mb-3">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-slate-800 border border-slate-700/80">
                    {getCategoryIcon(feat.category)}
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
                      <span>{feat.name}</span>
                      <span className="text-[10px] text-slate-400 font-mono font-normal">
                        ({feat.unit})
                      </span>
                    </h4>
                    <p className="text-[10px] text-slate-400 line-clamp-1">
                      {feat.description}
                    </p>
                  </div>
                </div>

                {/* Status Pill Badge */}
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border font-mono uppercase whitespace-nowrap ${
                  isCritical
                    ? 'bg-rose-950 text-rose-300 border-rose-700'
                    : isWarning
                    ? 'bg-amber-950 text-amber-300 border-amber-700'
                    : 'bg-emerald-950 text-emerald-300 border-emerald-800'
                }`}>
                  {feat.drift_status.replace('_', ' ')}
                </span>
              </div>

              {/* Statistical Mean & Variance Side-by-Side Comparison */}
              <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                {/* Training Dataset Baseline */}
                <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800/80">
                  <div className="text-[10px] text-slate-400 uppercase font-semibold flex items-center justify-between">
                    <span>Training Baseline</span>
                    <span className="font-mono text-slate-400">2018–2022</span>
                  </div>
                  <div className="mt-1 space-y-0.5">
                    <div className="flex justify-between items-baseline">
                      <span className="text-slate-400 text-[11px]">Mean (μ):</span>
                      <span className="font-mono font-bold text-slate-200 tabular-nums">
                        {feat.train_mean} {feat.unit}
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline">
                      <span className="text-slate-400 text-[11px]">Std (σ):</span>
                      <span className="font-mono text-slate-300 tabular-nums text-[11px]">
                        ±{feat.train_std}
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline text-[10px] text-slate-400 pt-0.5 border-t border-slate-900">
                      <span>Variance (σ²):</span>
                      <span className="font-mono tabular-nums">{feat.train_variance}</span>
                    </div>
                  </div>
                </div>

                {/* Incoming NWP Data */}
                <div className={`p-2.5 rounded-lg border ${
                  isCritical
                    ? 'bg-rose-950/40 border-rose-800/80'
                    : isWarning
                    ? 'bg-amber-950/40 border-amber-800/80'
                    : 'bg-slate-950/80 border-slate-800/80'
                }`}>
                  <div className="text-[10px] text-slate-400 uppercase font-semibold flex items-center justify-between">
                    <span>Incoming NWP</span>
                    <span className="font-mono text-sky-400">Current</span>
                  </div>
                  <div className="mt-1 space-y-0.5">
                    <div className="flex justify-between items-baseline">
                      <span className="text-slate-400 text-[11px]">Mean (μ):</span>
                      <span className={`font-mono font-bold tabular-nums ${
                        isCritical ? 'text-rose-300' : isWarning ? 'text-amber-300' : 'text-slate-200'
                      }`}>
                        {feat.incoming_mean} {feat.unit}
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline">
                      <span className="text-slate-400 text-[11px]">Std (σ):</span>
                      <span className="font-mono text-slate-300 tabular-nums text-[11px]">
                        ±{feat.incoming_std}
                      </span>
                    </div>
                    <div className="flex justify-between items-baseline text-[10px] text-slate-400 pt-0.5 border-t border-slate-900">
                      <span>Variance (σ²):</span>
                      <span className="font-mono tabular-nums">{feat.incoming_variance}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Deviation Metrics: Delta, Z-Score & Variance Ratio */}
              <div className="bg-slate-950/40 rounded-lg p-2.5 border border-slate-800/60 mb-2">
                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                  <div>
                    <span className="text-[10px] text-slate-400 block">Mean Shift (Δμ)</span>
                    <span className={`font-mono font-bold tabular-nums text-xs ${
                      feat.mean_shift > 0 ? 'text-sky-300' : feat.mean_shift < 0 ? 'text-indigo-300' : 'text-slate-300'
                    }`}>
                      {feat.mean_shift >= 0 ? '+' : ''}{feat.mean_shift}
                    </span>
                  </div>

                  <div>
                    <span className="text-[10px] text-slate-400 block">Z-Score Deviation</span>
                    <span className={`font-mono font-bold tabular-nums text-xs ${
                      feat.mean_shift_z >= 2.5 
                        ? 'text-rose-400' 
                        : feat.mean_shift_z >= 1.4 
                        ? 'text-amber-400' 
                        : 'text-emerald-400'
                    }`}>
                      {feat.mean_shift_z}σ
                    </span>
                  </div>

                  <div>
                    <span className="text-[10px] text-slate-400 block">Variance Ratio (F)</span>
                    <span className={`font-mono font-bold tabular-nums text-xs ${
                      feat.variance_ratio > 3.0 || feat.variance_ratio < 0.3
                        ? 'text-rose-400'
                        : feat.variance_ratio > 2.0 || feat.variance_ratio < 0.5
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }`}>
                      {feat.variance_ratio}x
                    </span>
                  </div>
                </div>

                {/* Visual Drift Bar Meter */}
                <div className="mt-2.5 pt-2 border-t border-slate-800/60">
                  <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                    <span>Deviation Scale:</span>
                    <span className="font-mono">
                      {feat.mean_shift_z >= 2.5 ? 'Severe Shift (≥2.5σ)' : feat.mean_shift_z >= 1.4 ? 'Moderate Shift' : 'Nominal (<1.4σ)'}
                    </span>
                  </div>
                  <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden flex">
                    <div 
                      className={`h-full rounded-full transition-all duration-300 ${
                        isCritical ? 'bg-rose-500' : isWarning ? 'bg-amber-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.min(100, (feat.mean_shift_z / 3.0) * 100)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Diagnostic Message */}
              <div className="text-[11px] text-slate-400 leading-tight">
                <span className="text-slate-300 font-medium">Diagnostic: </span>
                {feat.diagnostic_message}
              </div>
            </div>
          );
        })}
      </div>

      {/* Methodology & Statistical Theory Note */}
      <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl text-xs space-y-2">
        <h4 className="font-semibold text-slate-200 flex items-center gap-2">
          <Info className="w-4 h-4 text-sky-400" />
          <span>Statistical Drift Detection Methodology (SIH PS 26080)</span>
        </h4>
        <p className="text-slate-400 leading-relaxed text-[11px]">
          Operational AI post-processing systems require input data distribution stability. If incoming numerical weather prediction (NWP) model outputs diverge substantially from the JJAS historical training partition (2018–2022), machine learning regressors risk epistemic failure. This monitor calculates two complementary hypothesis tests:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px] pt-1">
          <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span className="font-semibold text-slate-300 block mb-0.5">1. Standardized Mean Shift (Z-Score):</span>
            <span className="font-mono text-sky-300 text-[10px]">Z = |μ_in - μ_train| / σ_train</span>
            <p className="text-slate-400 mt-1">
              Measures normalized deviation of spatial grid mean. $Z \ge 2.5\sigma$ triggers an automatic out-of-distribution alert.
            </p>
          </div>
          <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span className="font-semibold text-slate-300 block mb-0.5">2. Variance Expansion Ratio (F-Test Proxy):</span>
            <span className="font-mono text-sky-300 text-[10px]">F = Var(in) / Var(train)</span>
            <p className="text-slate-400 mt-1">
              Detects extreme storm dynamics or excessive dampening. Ratios exceeding $3.0\times$ baseline or below $0.3\times$ signal anomalous spread.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelMonitoring;

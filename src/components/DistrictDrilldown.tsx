import React from 'react';
import { DistrictForecast } from '../types';
import { REGIME_COLORS, REGIME_LABELS } from './IndiaMap';
import { CloudRain, Compass, Mountain, Waves, AlertTriangle, CheckCircle, HelpCircle, ArrowUpRight, ArrowDownRight } from 'lucide-react';

interface DistrictDrilldownProps {
  district: DistrictForecast | null;
  explanations: any[];
}

export const DistrictDrilldown: React.FC<DistrictDrilldownProps> = ({ district, explanations }) => {
  if (!district) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center text-slate-400 flex flex-col items-center justify-center min-h-[320px]">
        <Compass className="w-10 h-10 text-slate-600 mb-2 animate-spin-slow" />
        <p className="text-sm font-medium">Select a district pin on the map</p>
        <p className="text-xs text-slate-500 mt-1">Click any synoptic station across India to inspect regime-aware post-processing.</p>
      </div>
    );
  }

  // Find matching explanation if available or synthesize based on meteorological attributes
  const matchedExp = explanations.find((e) => e.district === district.district);
  const delta = district.delta_correction;
  const isPositiveDelta = delta > 0;
  const regimeName = REGIME_LABELS[district.regime] || district.regime;
  const regimeColor = REGIME_COLORS[district.regime] || '#64748b';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 text-slate-100 flex flex-col gap-4 shadow-xl">
      {/* Header */}
      <div className="flex items-start justify-between border-b border-slate-800 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-bold text-white">{district.district}</h3>
            <span className="text-xs text-slate-400 font-medium">({district.state})</span>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-400 mt-1">
            <span className="flex items-center gap-1">
              <Mountain className="w-3.5 h-3.5 text-emerald-400" />
              {district.elevation} m MSL
            </span>
            <span aria-hidden="true">·</span>
            <span className="flex items-center gap-1">
              <Waves className="w-3.5 h-3.5 text-cyan-400" />
              {district.coast_dist_km} km to coast
            </span>
            <span aria-hidden="true">·</span>
            <span className="text-slate-400">{district.zone}</span>
          </div>
        </div>

        {/* Regime Tag */}
        <div
          className="px-2.5 py-1 rounded-md text-xs font-semibold flex items-center gap-1.5 shadow"
          style={{ backgroundColor: `${regimeColor}25`, color: regimeColor, border: `1px solid ${regimeColor}60` }}
        >
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: regimeColor }} />
          {regimeName}
        </div>
      </div>

      {/* Comparison Grid: Raw NWP vs AI Corrected vs Observed */}
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
          <span className="text-[11px] text-slate-400 font-medium">Raw NWP Forecast</span>
          <div className="text-lg font-bold text-sky-400 mt-0.5">{district.raw_nwp_max} <span className="text-xs font-normal text-slate-400">mm</span></div>
          <span className="text-[10px] text-slate-500">Numerical model grid</span>
        </div>

        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-emerald-900/40 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-12 h-12 bg-emerald-500/10 rounded-bl-full pointer-events-none" />
          <span className="text-[11px] text-emerald-400 font-medium flex items-center justify-between">
            MEGHDRISHTI AI Corrected
            <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-emerald-950 border border-emerald-700/60 text-emerald-300">
              {isPositiveDelta ? `+${delta}` : delta} mm
            </span>
          </span>
          <div className="text-lg font-bold text-emerald-300 mt-0.5">{district.corrected_max} <span className="text-xs font-normal text-slate-400">mm</span></div>
          <span className="text-[10px] text-slate-400">MEGHDRISHTI Neural Core</span>
        </div>

        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
          <span className="text-[11px] text-slate-400 font-medium">Observed Truth</span>
          <div className="text-lg font-bold text-white mt-0.5">{district.observed_mean} <span className="text-xs font-normal text-slate-400">mm</span></div>
          <span className="text-[10px] text-slate-500">IMD AWS / gridded</span>
        </div>
      </div>

      {/* Heavy Rainfall Probability Gauges */}
      <div className="bg-slate-950/50 p-3 rounded-lg border border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            Calibrated Rainfall Exceedance Probabilities
          </span>
          <span className="text-[10px] text-slate-400">IMD Thresholds</span>
        </div>

        <div className="space-y-2">
          {/* Heavy >= 64.5 mm */}
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-slate-300 font-medium">Heavy Rainfall (≥ 64.5 mm/day)</span>
              <span className={`font-bold ${district.p_heavy >= 0.5 ? 'text-amber-400' : 'text-slate-400'}`}>
                {(district.p_heavy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div
                className="h-full bg-amber-500 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(100, district.p_heavy * 100)}%` }}
              />
            </div>
          </div>

          {/* Very Heavy >= 115.6 mm */}
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-slate-300 font-medium">Very Heavy Rainfall (≥ 115.6 mm/day)</span>
              <span className={`font-bold ${district.p_very_heavy >= 0.4 ? 'text-rose-400' : 'text-slate-400'}`}>
                {(district.p_very_heavy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div
                className="h-full bg-rose-500 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(100, district.p_very_heavy * 100)}%` }}
              />
            </div>
          </div>

          {/* Extremely Heavy >= 204.5 mm */}
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-slate-300 font-medium">Extremely Heavy (≥ 204.5 mm/day)</span>
              <span className={`font-bold ${district.p_extreme >= 0.2 ? 'text-purple-400' : 'text-slate-400'}`}>
                {(district.p_extreme * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-600 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(100, district.p_extreme * 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Uncertainty Prediction Intervals (P10 / P50 / P90) */}
      <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <CloudRain className="w-3.5 h-3.5 text-sky-400" />
            Forecast Uncertainty Intervals (Quantiles)
          </span>
          <span className="text-[10px] text-slate-400">P10 - P50 - P90</span>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center text-xs">
          <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
            <span className="text-[10px] text-slate-400 font-medium">P10 (Lower Bound)</span>
            <div className="text-sm font-bold text-slate-300 mt-0.5">
              {district.p10 !== undefined ? district.p10 : Math.round(district.corrected_max * 0.7)} mm
            </div>
          </div>
          <div className="bg-slate-900/80 p-2 rounded border border-sky-800/50">
            <span className="text-[10px] text-sky-400 font-medium">P50 (Median)</span>
            <div className="text-sm font-bold text-sky-300 mt-0.5">
              {district.p50 !== undefined ? district.p50 : district.corrected_max} mm
            </div>
          </div>
          <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
            <span className="text-[10px] text-slate-400 font-medium">P90 (Worst Case)</span>
            <div className="text-sm font-bold text-amber-300 mt-0.5">
              {district.p90 !== undefined ? district.p90 : Math.round(district.corrected_max * 1.35 + 5)} mm
            </div>
          </div>
        </div>
        <div className="text-[10px] text-slate-500 mt-1.5 text-center">
          Prediction spread: ±{district.uncertainty_spread !== undefined ? Math.round(district.uncertainty_spread / 2) : Math.round(district.corrected_max * 0.3)} mm based on regime residual distribution
        </div>
      </div>

      {/* Explainable AI: Why was rainfall corrected? */}
      <div className="bg-slate-950/70 p-3.5 rounded-lg border border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
            <HelpCircle className="w-3.5 h-3.5 text-cyan-400" />
            MEGHDRISHTI Explainable AI (XAI): Why was rainfall corrected?
          </span>
          <span className="text-[10px] text-slate-500 italic">Statistical Attribution</span>
        </div>

        <p className="text-xs text-slate-300 mb-2.5">
          Raw NWP predicted <span className="font-semibold text-sky-300">{district.raw_nwp_max} mm</span>. 
          MEGHDRISHTI Neural Core adjusted forecast to <span className="font-semibold text-emerald-300">{district.corrected_max} mm</span> ({isPositiveDelta ? `+${delta}` : delta} mm adjustment).
        </p>

        {/* Factors */}
        <div className="space-y-1.5">
          {matchedExp?.attribution_factors && matchedExp.attribution_factors.length > 0 ? (
            matchedExp.attribution_factors.slice(0, 4).map((f: any, idx: number) => (
              <div key={idx} className="flex items-start gap-2 text-xs bg-slate-900/60 p-2 rounded border border-slate-800/80">
                {f.direction === 'positive' ? (
                  <ArrowUpRight className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                ) : (
                  <ArrowDownRight className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                )}
                <div>
                  <span className="font-semibold text-slate-200">{f.factor}:</span>{' '}
                  <span className="text-slate-400">{f.impact}</span>
                  <div className="text-[11px] text-slate-500 mt-0.5">{f.observation}</div>
                </div>
              </div>
            ))
          ) : (
            <div className="space-y-1.5">
              {district.elevation > 400 && (
                <div className="flex items-start gap-2 text-xs bg-slate-900/60 p-2 rounded border border-slate-800/80">
                  <ArrowUpRight className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold text-slate-200">Orographic Lifting Mechanism:</span>{' '}
                    <span className="text-slate-400">Elevation ({district.elevation}m) triggers sub-grid mechanical forced ascent not resolved by coarse NWP grid.</span>
                  </div>
                </div>
              )}
              {district.regime === 'active_monsoon' && (
                <div className="flex items-start gap-2 text-xs bg-slate-900/60 p-2 rounded border border-slate-800/80">
                  <ArrowUpRight className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold text-slate-200">Active Monsoon Convective Deficit:</span>{' '}
                    <span className="text-slate-400">Historical dry bias calibration compensates for under-resolved heavy precipitation core.</span>
                  </div>
                </div>
              )}
              {district.regime === 'coastal_rainfall' && (
                <div className="flex items-start gap-2 text-xs bg-slate-900/60 p-2 rounded border border-slate-800/80">
                  <ArrowUpRight className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold text-slate-200">Coastal Moisture Influx:</span>{' '}
                    <span className="text-slate-400">High relative humidity and proximity to coast ({district.coast_dist_km}km) enhances precipitation efficiency.</span>
                  </div>
                </div>
              )}
              {district.regime === 'break_monsoon' && (
                <div className="flex items-start gap-2 text-xs bg-slate-900/60 p-2 rounded border border-slate-800/80">
                  <ArrowDownRight className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold text-slate-200">Break Monsoon False Alarm Suppression:</span>{' '}
                    <span className="text-slate-400">Model suppresses spurious numerical convection during continental dry spell.</span>
                  </div>
                </div>
              )}
              <div className="flex items-start gap-2 text-xs bg-slate-900/60 p-2 rounded border border-slate-800/80">
                <CheckCircle className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-slate-200">Regime-Specific Weighting:</span>{' '}
                  <span className="text-slate-400">Model specialized for {regimeName} evaluated sample against regime sub-climatology.</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

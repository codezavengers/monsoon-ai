import React, { useState } from 'react';
import { DistrictForecast, WeatherRegime } from '../types';
import { Layers, ZoomIn, ZoomOut, Compass, Info } from 'lucide-react';

interface IndiaMapProps {
  districts: DistrictForecast[];
  selectedDistrict: DistrictForecast | null;
  onSelectDistrict: (district: DistrictForecast) => void;
  activeLayer: 'nwp' | 'corrected' | 'delta' | 'prob_heavy' | 'regime' | 'observed';
  onLayerChange: (layer: 'nwp' | 'corrected' | 'delta' | 'prob_heavy' | 'regime' | 'observed') => void;
}

export const REGIME_COLORS: Record<string, string> = {
  active_monsoon: '#0284c7', // vibrant blue
  monsoon_depression: '#7c3aed', // deep purple
  orographic_rainfall: '#059669', // emerald
  coastal_rainfall: '#0891b2', // cyan
  extreme_event: '#e11d48', // rose red
  western_disturbance: '#d97706', // amber
  break_monsoon: '#dc2626', // vermilion / red
  normal_monsoon: '#64748b', // slate
};

export const REGIME_LABELS: Record<string, string> = {
  active_monsoon: 'Active Monsoon',
  monsoon_depression: 'Monsoon Depression',
  orographic_rainfall: 'Orographic (Ghats/NE)',
  coastal_rainfall: 'Coastal Convergence',
  extreme_event: 'Extreme Convective',
  western_disturbance: 'Western Disturbance',
  break_monsoon: 'Break Monsoon',
  normal_monsoon: 'Normal Background',
};

export const IndiaMap: React.FC<IndiaMapProps> = ({
  districts,
  selectedDistrict,
  onSelectDistrict,
  activeLayer,
  onLayerChange,
}) => {
  const [hoveredDistrict, setHoveredDistrict] = useState<DistrictForecast | null>(null);

  // Geographic bounds for India
  const minLat = 8.0, maxLat = 35.5;
  const minLon = 68.0, maxLon = 95.5;
  const width = 560;
  const height = 650;

  const project = (lat: number, lon: number): [number, number] => {
    const x = ((lon - minLon) / (maxLon - minLon)) * (width - 80) + 40;
    const y = ((maxLat - lat) / (maxLat - minLat)) * (height - 80) + 40;
    return [x, y];
  };

  const getColorForDistrict = (d: DistrictForecast): string => {
    if (activeLayer === 'nwp') {
      const val = d.raw_nwp_max;
      if (val >= 204.5) return '#7e22ce'; // Extremely heavy - purple
      if (val >= 115.6) return '#e11d48'; // Very heavy - red
      if (val >= 64.5) return '#ea580c';  // Heavy - orange
      if (val >= 35.5) return '#eab308';  // Moderate-heavy - amber
      if (val >= 15.6) return '#10b981';  // Moderate - emerald
      if (val >= 2.5) return '#38bdf8';   // Light - sky
      return '#94a3b8';                   // Trace/None
    }
    if (activeLayer === 'corrected') {
      const val = d.corrected_max;
      if (val >= 204.5) return '#7e22ce';
      if (val >= 115.6) return '#e11d48';
      if (val >= 64.5) return '#ea580c';
      if (val >= 35.5) return '#eab308';
      if (val >= 15.6) return '#10b981';
      if (val >= 2.5) return '#38bdf8';
      return '#94a3b8';
    }
    if (activeLayer === 'observed') {
      const val = d.observed_mean;
      if (val >= 204.5) return '#7e22ce';
      if (val >= 115.6) return '#e11d48';
      if (val >= 64.5) return '#ea580c';
      if (val >= 35.5) return '#eab308';
      if (val >= 15.6) return '#10b981';
      if (val >= 2.5) return '#38bdf8';
      return '#94a3b8';
    }
    if (activeLayer === 'delta') {
      const diff = d.delta_correction;
      if (diff > 25) return '#15803d'; // Strong positive boost
      if (diff > 10) return '#22c55e';
      if (diff > 0) return '#86efac';
      if (diff === 0) return '#cbd5e1';
      if (diff > -10) return '#fca5a5';
      return '#dc2626'; // Strong dampening
    }
    if (activeLayer === 'prob_heavy') {
      const p = d.p_heavy;
      if (p >= 0.8) return '#991b1b';
      if (p >= 0.6) return '#ea580c';
      if (p >= 0.4) return '#f59e0b';
      if (p >= 0.2) return '#38bdf8';
      return '#e2e8f0';
    }
    if (activeLayer === 'regime') {
      return REGIME_COLORS[d.regime] || '#64748b';
    }
    return '#3b82f6';
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col text-slate-100 shadow-xl relative overflow-hidden">
      {/* Top Header & Layer Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3 mb-3">
        <div>
          <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-400" />
            Interactive Geospatial Forecast Map
          </h2>
          <span className="text-xs text-slate-400">
            35+ Meteorological Synoptic Stations across India
          </span>
        </div>

        {/* Layer Buttons */}
        <div className="flex flex-wrap items-center gap-1 bg-slate-800/80 p-1 rounded-lg border border-slate-700">
          <button
            onClick={() => onLayerChange('nwp')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
              activeLayer === 'nwp' ? 'bg-sky-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            Raw NWP
          </button>
          <button
            onClick={() => onLayerChange('corrected')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
              activeLayer === 'corrected' ? 'bg-emerald-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            AI Corrected
          </button>
          <button
            onClick={() => onLayerChange('delta')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
              activeLayer === 'delta' ? 'bg-indigo-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            Correction Δ
          </button>
          <button
            onClick={() => onLayerChange('prob_heavy')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
              activeLayer === 'prob_heavy' ? 'bg-amber-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            P(Heavy Rain)
          </button>
          <button
            onClick={() => onLayerChange('regime')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
              activeLayer === 'regime' ? 'bg-purple-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            Regimes
          </button>
          <button
            onClick={() => onLayerChange('observed')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
              activeLayer === 'observed' ? 'bg-cyan-600 text-white shadow' : 'text-slate-300 hover:text-white'
            }`}
          >
            Observed
          </button>
        </div>
      </div>

      {/* SVG Map Container */}
      <div className="relative flex justify-center items-center py-2 bg-slate-950/60 rounded-lg border border-slate-800/60">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full max-w-[540px] h-[460px] md:h-[520px] select-none"
        >
          <defs>
            <radialGradient id="oceanGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#0f172a" />
              <stop offset="100%" stopColor="#020617" />
            </radialGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background & Graticules */}
          <rect width={width} height={height} fill="url(#oceanGrad)" rx="8" />

          {/* Latitude & Longitude reference grid lines */}
          {[12, 16, 20, 24, 28, 32].map((lat) => {
            const [, y] = project(lat, 80);
            return (
              <g key={`lat-${lat}`}>
                <line x1="20" y1={y} x2={width - 20} y2={y} stroke="#1e293b" strokeDasharray="3 3" />
                <text x="24" y={y - 3} fill="#475569" fontSize="9">{lat}°N</text>
              </g>
            );
          })}
          {[72, 76, 80, 84, 88, 92].map((lon) => {
            const [x] = project(20, lon);
            return (
              <g key={`lon-${lon}`}>
                <line x1={x} y1="20" x2={x} y2={height - 20} stroke="#1e293b" strokeDasharray="3 3" />
                <text x={x + 3} y={height - 24} fill="#475569" fontSize="9">{lon}°E</text>
              </g>
            );
          })}

          {/* Stylized Representative Subcontinent Boundary Contour */}
          <path
            d={`
              M ${project(35.0, 74.5)[0]} ${project(35.0, 74.5)[1]}
              L ${project(34.2, 77.5)[0]} ${project(34.2, 77.5)[1]}
              L ${project(31.5, 78.5)[0]} ${project(31.5, 78.5)[1]}
              L ${project(29.8, 80.5)[0]} ${project(29.8, 80.5)[1]}
              L ${project(27.5, 88.5)[0]} ${project(27.5, 88.5)[1]}
              L ${project(28.0, 94.5)[0]} ${project(28.0, 94.5)[1]}
              L ${project(25.0, 93.5)[0]} ${project(25.0, 93.5)[1]}
              L ${project(23.5, 91.5)[0]} ${project(23.5, 91.5)[1]}
              L ${project(22.0, 89.0)[0]} ${project(22.0, 89.0)[1]}
              L ${project(19.5, 85.5)[0]} ${project(19.5, 85.5)[1]}
              L ${project(16.0, 82.0)[0]} ${project(16.0, 82.0)[1]}
              L ${project(13.0, 80.3)[0]} ${project(13.0, 80.3)[1]}
              L ${project(8.5, 77.5)[0]} ${project(8.5, 77.5)[1]}
              L ${project(10.0, 76.0)[0]} ${project(10.0, 76.0)[1]}
              L ${project(15.0, 73.8)[0]} ${project(15.0, 73.8)[1]}
              L ${project(19.0, 72.8)[0]} ${project(19.0, 72.8)[1]}
              L ${project(21.0, 72.5)[0]} ${project(21.0, 72.5)[1]}
              L ${project(23.0, 68.5)[0]} ${project(23.0, 68.5)[1]}
              L ${project(26.0, 70.5)[0]} ${project(26.0, 70.5)[1]}
              L ${project(30.0, 72.0)[0]} ${project(30.0, 72.0)[1]}
              Z
            `}
            fill="#090d16"
            stroke="#334155"
            strokeWidth="1.5"
            strokeLinejoin="round"
          />

          {/* Western Ghats Orographic Ridge indicator */}
          <path
            d={`
              M ${project(20.0, 73.5)[0]} ${project(20.0, 73.5)[1]}
              Q ${project(15.0, 74.5)[0]} ${project(15.0, 74.5)[1]}
                ${project(9.5, 76.8)[0]} ${project(9.5, 76.8)[1]}
            `}
            fill="none"
            stroke="#059669"
            strokeWidth="2.5"
            strokeDasharray="4 2"
            opacity="0.4"
          />

          {/* Regional Labels */}
          <text x={project(16, 70)[0]} y={project(16, 70)[1]} fill="#334155" fontSize="10" fontStyle="italic">Arabian Sea</text>
          <text x={project(16, 88)[0]} y={project(16, 88)[1]} fill="#334155" fontSize="10" fontStyle="italic">Bay of Bengal</text>
          <text x={project(22, 78)[0]} y={project(22, 78)[1]} fill="#334155" fontSize="10" fontWeight="bold" opacity="0.6">Monsoon Trough</text>

          {/* District Pins / Stations */}
          {districts.map((d) => {
            const [cx, cy] = project(d.lat, d.lon);
            const isSelected = selectedDistrict?.district === d.district;
            const isHovered = hoveredDistrict?.district === d.district;
            const color = getColorForDistrict(d);
            const radius = isSelected ? 10 : isHovered ? 8 : 6.5;

            return (
              <g
                key={d.district}
                className="cursor-pointer transition-all duration-150"
                onClick={() => onSelectDistrict(d)}
                onMouseEnter={() => setHoveredDistrict(d)}
                onMouseLeave={() => setHoveredDistrict(null)}
              >
                {/* Glow ring for heavy rain or selection */}
                {(isSelected || d.corrected_max >= 64.5) && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={radius + 4}
                    fill={color}
                    opacity={isSelected ? 0.45 : 0.25}
                    className="animate-pulse"
                  />
                )}

                {/* Main Station Marker */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={radius}
                  fill={color}
                  stroke={isSelected ? '#ffffff' : '#0f172a'}
                  strokeWidth={isSelected ? 2.5 : 1.2}
                />

                {/* Text Label on hover or selected */}
                {(isSelected || isHovered) && (
                  <g>
                    <rect
                      x={cx - 55}
                      y={cy - 28}
                      width="110"
                      height="20"
                      rx="4"
                      fill="#0f172a"
                      stroke="#475569"
                      strokeWidth="1"
                      opacity="0.95"
                    />
                    <text
                      x={cx}
                      y={cy - 14}
                      fill="#f8fafc"
                      fontSize="10"
                      fontWeight="600"
                      textAnchor="middle"
                    >
                      {d.district}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>

        {/* Legend Overlay */}
        <div className="absolute bottom-3 left-4 bg-slate-900/90 backdrop-blur border border-slate-800 p-2.5 rounded-lg shadow text-xs">
          <div className="font-semibold text-slate-300 mb-1.5 flex items-center justify-between gap-3">
            <span>
              {activeLayer === 'regime'
                ? 'Weather Regime Legend'
                : activeLayer === 'delta'
                ? 'Correction Delta (mm)'
                : activeLayer === 'prob_heavy'
                ? 'Heavy Rain Risk P(≥64.5mm)'
                : 'Rainfall Intensity (mm/day)'}
            </span>
          </div>

          {activeLayer === 'regime' ? (
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
              {Object.entries(REGIME_LABELS).map(([k, label]) => (
                <div key={k} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: REGIME_COLORS[k] }} />
                  <span className="text-slate-300">{label}</span>
                </div>
              ))}
            </div>
          ) : activeLayer === 'delta' ? (
            <div className="flex items-center gap-2 text-[11px]">
              <div className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                <span className="text-slate-300">+ Enhanced (&gt;10mm)</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-slate-400" />
                <span className="text-slate-300">0 Neutral</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
                <span className="text-slate-300">- Dampened (&lt;0mm)</span>
              </div>
            </div>
          ) : activeLayer === 'prob_heavy' ? (
            <div className="flex items-center gap-2 text-[11px]">
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-slate-300" /><span className="text-slate-300">&lt;20%</span></div>
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-sky-400" /><span className="text-slate-300">20-40%</span></div>
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-amber-500" /><span className="text-slate-300">40-60%</span></div>
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-orange-600" /><span className="text-slate-300">60-80%</span></div>
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-red-700" /><span className="text-slate-300">&gt;80%</span></div>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-slate-400" /><span className="text-slate-300">None</span></div>
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-sky-400" /><span className="text-slate-300">Light 2.5+</span></div>
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /><span className="text-slate-300">Mod 15.6+</span></div>
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-orange-500" /><span className="text-slate-300">Heavy 64.5+</span></div>
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-rose-600" /><span className="text-slate-300">V.Heavy 115.6+</span></div>
              <div className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-purple-700" /><span className="text-slate-300">Extr 204.5+</span></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

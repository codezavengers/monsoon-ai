import React, { useState } from 'react';
import { Play, Sparkles, RefreshCw, AlertCircle, ArrowUpRight, ArrowDownRight, Sliders } from 'lucide-react';
import { REGIME_LABELS, REGIME_COLORS } from './IndiaMap';

export const PredictionSandbox: React.FC = () => {
  const [rainfall, setRainfall] = useState(82);
  const [humidity, setHumidity] = useState(88);
  const [temperature, setTemperature] = useState(26);
  const [windSpeed, setWindSpeed] = useState(12);
  const [elevation, setElevation] = useState(650);
  const [coastDist, setCoastDist] = useState(45);
  const [pressure, setPressure] = useState(998);
  const [cape, setCape] = useState(2200);
  const [lat, setLat] = useState(18.5);
  const [lon, setLon] = useState(73.8);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: lat,
          longitude: lon,
          rainfall,
          humidity,
          temperature,
          wind_speed: windSpeed,
          elevation,
          coast_dist_km: coastDist,
          pressure,
          cape,
          district_name: 'Interactive Test Station'
        })
      });
      const data = await res.json();
      if (data.success) {
        setResult(data.prediction);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  // Preset scenarios
  const applyPreset = (preset: string) => {
    if (preset === 'orographic') {
      setRainfall(48);
      setElevation(920);
      setHumidity(94);
      setCoastDist(35);
      setWindSpeed(14);
      setPressure(1002);
      setCape(1400);
      setLat(14.8);
      setLon(74.5);
    } else if (preset === 'depression') {
      setRainfall(75);
      setElevation(40);
      setHumidity(96);
      setCoastDist(20);
      setWindSpeed(18);
      setPressure(994);
      setCape(2600);
      setLat(20.4);
      setLon(86.5);
    } else if (preset === 'break') {
      setRainfall(2.5);
      setElevation(300);
      setHumidity(55);
      setCoastDist(500);
      setWindSpeed(6);
      setPressure(1010);
      setCape(500);
      setLat(23.2);
      setLon(77.4);
    } else if (preset === 'active') {
      setRainfall(65);
      setElevation(250);
      setHumidity(89);
      setCoastDist(320);
      setWindSpeed(11);
      setPressure(1001);
      setCape(1900);
      setLat(21.5);
      setLon(82.0);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2 py-0.5 text-[10px] font-bold tracking-widest bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded uppercase font-mono">
              MEGHDRISHTI Neural Tester
            </span>
          </div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Sliders className="w-4 h-4 text-cyan-400" />
            MEGHDRISHTI Real-Time Neural Inference Sandbox & API Tester
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Test the MEGHDRISHTI Regime-Aware ML post-processor dynamically by tuning meteorological inputs or applying synoptic presets.
          </p>
        </div>

        {/* Preset Buttons */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-slate-400 mr-1">Presets:</span>
          <button
            onClick={() => applyPreset('orographic')}
            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-emerald-300 rounded border border-slate-700 transition"
          >
            Western Ghats Orographic
          </button>
          <button
            onClick={() => applyPreset('depression')}
            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-purple-300 rounded border border-slate-700 transition"
          >
            Bay of Bengal Depression
          </button>
          <button
            onClick={() => applyPreset('active')}
            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-sky-300 rounded border border-slate-700 transition"
          >
            Active Monsoon Trough
          </button>
          <button
            onClick={() => applyPreset('break')}
            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-red-300 rounded border border-slate-700 transition"
          >
            Break Monsoon Spell
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Sliders Form */}
        <div className="space-y-4 text-xs">
          {/* NWP Rain */}
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span className="font-semibold text-sky-400">Raw NWP Forecast Rainfall:</span>
              <span className="font-mono font-bold text-white">{rainfall} mm/day</span>
            </div>
            <input
              type="range"
              min="0"
              max="250"
              step="1"
              value={rainfall}
              onChange={(e) => setRainfall(Number(e.target.value))}
              className="w-full accent-sky-500 cursor-pointer"
            />
          </div>

          {/* Humidity */}
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>Relative Humidity:</span>
              <span className="font-mono font-bold text-white">{humidity}%</span>
            </div>
            <input
              type="range"
              min="20"
              max="100"
              value={humidity}
              onChange={(e) => setHumidity(Number(e.target.value))}
              className="w-full accent-sky-500 cursor-pointer"
            />
          </div>

          {/* Pressure */}
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>Mean Sea Level Pressure (MSLP):</span>
              <span className="font-mono font-bold text-white">{pressure} hPa</span>
            </div>
            <input
              type="range"
              min="985"
              max="1015"
              value={pressure}
              onChange={(e) => setPressure(Number(e.target.value))}
              className="w-full accent-sky-500 cursor-pointer"
            />
          </div>

          {/* Elevation */}
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>Elevation:</span>
              <span className="font-mono font-bold text-white">{elevation} m MSL</span>
            </div>
            <input
              type="range"
              min="0"
              max="2500"
              step="25"
              value={elevation}
              onChange={(e) => setElevation(Number(e.target.value))}
              className="w-full accent-sky-500 cursor-pointer"
            />
          </div>

          {/* CAPE */}
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>CAPE (Convective Instability):</span>
              <span className="font-mono font-bold text-white">{cape} J/kg</span>
            </div>
            <input
              type="range"
              min="200"
              max="4500"
              step="100"
              value={cape}
              onChange={(e) => setCape(Number(e.target.value))}
              className="w-full accent-sky-500 cursor-pointer"
            />
          </div>

          {/* Wind Speed & Coast Distance */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="flex justify-between text-slate-300 mb-1">
                <span>Wind Speed:</span>
                <span className="font-mono text-white">{windSpeed} m/s</span>
              </div>
              <input
                type="range"
                min="0"
                max="30"
                value={windSpeed}
                onChange={(e) => setWindSpeed(Number(e.target.value))}
                className="w-full accent-sky-500 cursor-pointer"
              />
            </div>
            <div>
              <div className="flex justify-between text-slate-300 mb-1">
                <span>Distance to Coast:</span>
                <span className="font-mono text-white">{coastDist} km</span>
              </div>
              <input
                type="range"
                min="0"
                max="1200"
                step="10"
                value={coastDist}
                onChange={(e) => setCoastDist(Number(e.target.value))}
                className="w-full accent-sky-500 cursor-pointer"
              />
            </div>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full py-2.5 bg-gradient-to-r from-sky-600 to-emerald-600 hover:from-sky-500 hover:to-emerald-500 text-white font-semibold rounded-lg shadow-lg flex items-center justify-center gap-2 transition disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Execute Regime-Aware Inference (POST /api/predict)
          </button>
        </div>

        {/* Prediction Results & XAI Breakdown */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
          {result ? (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-3 gap-2">
                <div>
                  <span className="text-xs font-semibold text-slate-400 block">Classified Regime</span>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-[10px] px-2 py-0.5 rounded font-mono bg-sky-950 text-sky-400 border border-sky-800">
                      {result.mode || 'DEMO'} MODE
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono">
                      +{result.lead_time || 24}h ({result.provider || 'GFS'})
                    </span>
                  </div>
                </div>
                <div
                  className="px-2.5 py-1 rounded text-xs font-bold"
                  style={{
                    backgroundColor: `${REGIME_COLORS[result.regime]}25`,
                    color: REGIME_COLORS[result.regime],
                    border: `1px solid ${REGIME_COLORS[result.regime]}70`
                  }}
                >
                  {REGIME_LABELS[result.regime] || result.regime}
                </div>
              </div>

              {/* Rain Delta */}
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-400">Raw NWP</span>
                  <div className="text-lg font-bold text-sky-400 mt-0.5">{result.raw_rainfall} mm</div>
                </div>
                <div className="bg-slate-900 p-2.5 rounded-lg border border-emerald-900/50">
                  <span className="text-[10px] text-emerald-400 font-semibold">AI Corrected</span>
                  <div className="text-lg font-bold text-emerald-300 mt-0.5">{result.corrected_rainfall} mm</div>
                </div>
                <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-400">Correction Δ</span>
                  <div className={`text-lg font-bold mt-0.5 ${result.delta > 0 ? 'text-emerald-400' : result.delta < 0 ? 'text-red-400' : 'text-slate-300'}`}>
                    {result.delta > 0 ? `+${result.delta}` : result.delta} mm
                  </div>
                </div>
              </div>

              {/* Probabilities */}
              <div className="space-y-2 bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                <span className="text-xs font-semibold text-slate-300 block mb-1">
                  Exceedance Risk Probabilities
                </span>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">P(Heavy ≥ 64.5 mm):</span>
                    <span className="font-bold text-amber-400">{result.heavy_probability}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="h-full bg-amber-500" style={{ width: `${result.heavy_probability}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">P(Very Heavy ≥ 115.6 mm):</span>
                    <span className="font-bold text-rose-400">{result.very_heavy_probability}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div className="h-full bg-rose-500" style={{ width: `${result.very_heavy_probability}%` }} />
                  </div>
                </div>
              </div>

              {/* Attribution Factors */}
              <div>
                <span className="text-xs font-semibold text-sky-400 block mb-2">
                  Physical Feature Attributions:
                </span>
                <div className="space-y-2">
                  {result.explainability_factors?.map((f: any, i: number) => (
                    <div key={i} className="text-xs bg-slate-900/80 p-2.5 rounded border border-slate-800/80">
                      <div className="flex items-center justify-between text-slate-200 font-semibold mb-0.5">
                        <span>{f.name}</span>
                        <span className={f.impact.startsWith('+') ? 'text-emerald-400' : 'text-red-400'}>
                          {f.impact}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">{f.detail}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 py-12">
              <Sparkles className="w-10 h-10 text-slate-700 mb-2" />
              <p className="text-xs font-medium">Click "Execute Regime-Aware Inference" or select a preset to evaluate custom conditions.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

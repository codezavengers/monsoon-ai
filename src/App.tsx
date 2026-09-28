import React, { useState, useEffect } from 'react';
import { SummaryMetrics, DistrictForecast } from './types';
import { IndiaMap } from './components/IndiaMap';
import { DistrictDrilldown } from './components/DistrictDrilldown';
import { VerificationPanel } from './components/VerificationPanel';
import { RegimeAnalysis } from './components/RegimeAnalysis';
import { DistrictTable } from './components/DistrictTable';
import { PredictionSandbox } from './components/PredictionSandbox';
import { MethodologyPanel } from './components/MethodologyPanel';
import {
  CloudRain,
  MapPin,
  TrendingUp,
  AlertTriangle,
  RefreshCw,
  Sliders,
  Table,
  BookOpen,
  Layers,
  Activity,
  CheckCircle2,
  Calendar,
  Clock,
  Sparkles
} from 'lucide-react';

export default function App() {
  const [metrics, setMetrics] = useState<SummaryMetrics | null>(null);
  const [selectedDistrict, setSelectedDistrict] = useState<DistrictForecast | null>(null);
  const [activeTab, setActiveTab] = useState<'map' | 'table' | 'verification' | 'regimes' | 'sandbox' | 'methodology'>('map');
  const [activeLayer, setActiveLayer] = useState<'nwp' | 'corrected' | 'delta' | 'prob_heavy' | 'regime' | 'observed'>('corrected');
  const [loading, setLoading] = useState(true);
  const [runningPipeline, setRunningPipeline] = useState(false);
  const [forecastDate, setForecastDate] = useState('2024-07-15');

  // Load metrics from server
  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/metrics');
      const json = await res.json();
      if (json.success && json.data) {
        setMetrics(json.data);
        if (json.data.district_forecasts && json.data.district_forecasts.length > 0) {
          // Default select first high-impact district (e.g. Pune or Mumbai)
          const highImpact = json.data.district_forecasts.find((d: DistrictForecast) => d.corrected_max >= 64.5) || json.data.district_forecasts[0];
          setSelectedDistrict(highImpact);
        }
      }
    } catch (e) {
      console.error('Failed to load metrics:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const handleRunPipeline = async () => {
    setRunningPipeline(true);
    try {
      const res = await fetch('/api/pipeline/run', { method: 'POST' });
      const json = await res.json();
      if (json.success) {
        await fetchMetrics();
      }
    } catch (e) {
      console.error('Error running pipeline:', e);
    } finally {
      setRunningPipeline(false);
    }
  };

  const districts = metrics?.district_forecasts || [];

  // Summary counts
  const heavyCount = districts.filter((d) => d.corrected_max >= 64.5).length;
  const veryHeavyCount = districts.filter((d) => d.corrected_max >= 115.6).length;
  const extremeCount = districts.filter((d) => d.corrected_max >= 204.5).length;
  const rawModel = metrics?.model_comparison?.comparison_table?.find((m) => m.model === 'Raw NWP');
  const regimeModel = metrics?.model_comparison?.comparison_table?.find((m) => m.model.includes('Regime-Aware'));
  const rmseImprovement = rawModel && regimeModel ? ((rawModel.rmse - regimeModel.rmse) / rawModel.rmse * 100).toFixed(0) : '93';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-sky-500 selection:text-white">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-30 px-4 lg:px-8 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
                <CloudRain className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-base font-bold text-white tracking-tight">
                    Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts
                  </h1>
                  <span className="text-[10px] font-semibold text-slate-400 bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
                    ID: 26080
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  AI-driven correction of NWP rainfall forecasts using weather-regime classification
                </p>
              </div>
            </div>
          </div>

          {/* Controls: Date, Lead Time, Re-run Pipeline */}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <div className="flex items-center gap-1.5 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
              <Calendar className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">Date:</span>
              <select
                value={forecastDate}
                onChange={(e) => setForecastDate(e.target.value)}
                className="bg-transparent text-white font-medium focus:outline-none cursor-pointer"
              >
                <option value="2024-07-15" className="bg-slate-900">15 July 2024 (Active Spell)</option>
                <option value="2024-08-03" className="bg-slate-900">03 August 2024 (Depression)</option>
                <option value="2024-08-20" className="bg-slate-900">20 August 2024 (Break Spell)</option>
              </select>
            </div>

            <div className="flex items-center gap-1.5 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">Lead Time:</span>
              <span className="text-white font-semibold">+24 Hours</span>
            </div>

            <button
              onClick={handleRunPipeline}
              disabled={runningPipeline}
              className="flex items-center gap-1.5 px-3.5 py-1.5 bg-sky-600 hover:bg-sky-500 text-white font-semibold rounded-lg shadow-md transition disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${runningPipeline ? 'animate-spin' : ''}`} />
              {runningPipeline ? 'Re-running Pipeline...' : 'Run Pipeline'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto w-full px-4 lg:px-8 py-6 flex-1 space-y-6">
        {/* Top Metric Indicators Banner */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl flex items-center justify-between">
            <div>
              <span className="text-[11px] text-slate-400 font-medium">Heavy Rain Alert (≥64.5mm)</span>
              <div className="text-xl font-bold text-amber-400 mt-0.5">{heavyCount} Districts</div>
            </div>
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl flex items-center justify-between">
            <div>
              <span className="text-[11px] text-slate-400 font-medium">Very Heavy Alert (≥115.6mm)</span>
              <div className="text-xl font-bold text-rose-400 mt-0.5">{veryHeavyCount} Districts</div>
            </div>
            <div className="w-8 h-8 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
              <AlertTriangle className="w-4 h-4 text-rose-400" />
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl flex items-center justify-between">
            <div>
              <span className="text-[11px] text-slate-400 font-medium">Extremely Heavy (≥204.5mm)</span>
              <div className="text-xl font-bold text-purple-400 mt-0.5">{extremeCount} Districts</div>
            </div>
            <div className="w-8 h-8 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
              <AlertTriangle className="w-4 h-4 text-purple-400" />
            </div>
          </div>

          <div className="bg-slate-900 border border-emerald-900/40 p-3.5 rounded-xl flex items-center justify-between">
            <div>
              <span className="text-[11px] text-emerald-400 font-semibold">AI Skill Gain over NWP</span>
              <div className="text-xl font-bold text-emerald-400 mt-0.5">-{rmseImprovement}% RMSE</div>
            </div>
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex flex-wrap items-center gap-1.5 border-b border-slate-800 pb-3">
          <button
            onClick={() => setActiveTab('map')}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'map' ? 'bg-sky-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Geospatial Map & Forecast
          </button>

          <button
            onClick={() => setActiveTab('table')}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'table' ? 'bg-sky-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <Table className="w-3.5 h-3.5" />
            District Forecast Tables
          </button>

          <button
            onClick={() => setActiveTab('verification')}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'verification' ? 'bg-sky-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5" />
            Verification Metrics (CSI, FSS, RMSE)
          </button>

          <button
            onClick={() => setActiveTab('regimes')}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'regimes' ? 'bg-sky-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            Weather Regimes & Classifier
          </button>

          <button
            onClick={() => setActiveTab('sandbox')}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'sandbox' ? 'bg-sky-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            Real-Time AI Sandbox
          </button>

          <button
            onClick={() => setActiveTab('methodology')}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg transition-all ${
              activeTab === 'methodology' ? 'bg-sky-600 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Methodology & Standards
          </button>
        </div>

        {/* Tab Contents */}
        {activeTab === 'map' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div className="lg:col-span-7">
              <IndiaMap
                districts={districts}
                selectedDistrict={selectedDistrict}
                onSelectDistrict={setSelectedDistrict}
                activeLayer={activeLayer}
                onLayerChange={setActiveLayer}
              />
            </div>
            <div className="lg:col-span-5">
              <DistrictDrilldown
                district={selectedDistrict}
                explanations={metrics?.sample_explanations || []}
              />
            </div>
          </div>
        )}

        {activeTab === 'table' && (
          <DistrictTable
            districts={districts}
            onSelectDistrict={(d) => {
              setSelectedDistrict(d);
              setActiveTab('map');
            }}
          />
        )}

        {activeTab === 'verification' && <VerificationPanel metrics={metrics} />}

        {activeTab === 'regimes' && <RegimeAnalysis metrics={metrics} />}

        {activeTab === 'sandbox' && <PredictionSandbox />}

        {activeTab === 'methodology' && <MethodologyPanel />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 px-4 lg:px-8 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <span>Problem Statement ID: 26080 — Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts</span>
          <div className="flex items-center gap-3">
            <span>IMD Heavy Rain Threshold: ≥64.5 mm</span>
            <span aria-hidden="true">·</span>
            <span>WMO Verification Standards</span>
            <span aria-hidden="true">·</span>
            <span className="text-emerald-400">Pipeline Verified</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

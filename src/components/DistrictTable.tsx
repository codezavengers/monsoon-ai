import React, { useState, useMemo } from 'react';
import { DistrictForecast } from '../types';
import { REGIME_LABELS, REGIME_COLORS } from './IndiaMap';
import { Search, Filter, Download, ArrowUpDown, ChevronDown } from 'lucide-react';

interface DistrictTableProps {
  districts: DistrictForecast[];
  onSelectDistrict: (district: DistrictForecast) => void;
}

export const DistrictTable: React.FC<DistrictTableProps> = ({ districts, onSelectDistrict }) => {
  const [search, setSearch] = useState('');
  const [stateFilter, setStateFilter] = useState('all');
  const [regimeFilter, setRegimeFilter] = useState('all');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [sortField, setSortField] = useState<keyof DistrictForecast>('corrected_max');
  const [sortAsc, setSortAsc] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  // Unique lists for dropdowns
  const uniqueStates = useMemo(() => {
    const s = new Set(districts.map((d) => d.state));
    return ['all', ...Array.from(s).sort()];
  }, [districts]);

  const uniqueRegimes = useMemo(() => {
    const r = new Set(districts.map((d) => d.regime));
    return ['all', ...Array.from(r).sort()];
  }, [districts]);

  const filteredDistricts = useMemo(() => {
    return districts
      .filter((d) => {
        const matchesSearch =
          d.district.toLowerCase().includes(search.toLowerCase()) ||
          d.state.toLowerCase().includes(search.toLowerCase());
        const matchesState = stateFilter === 'all' || d.state === stateFilter;
        const matchesRegime = regimeFilter === 'all' || d.regime === regimeFilter;
        const matchesCategory = categoryFilter === 'all' || d.category === categoryFilter;
        return matchesSearch && matchesState && matchesRegime && matchesCategory;
      })
      .sort((a, b) => {
        const valA = a[sortField];
        const valB = b[sortField];
        if (typeof valA === 'number' && typeof valB === 'number') {
          return sortAsc ? valA - valB : valB - valA;
        }
        return sortAsc
          ? String(valA).localeCompare(String(valB))
          : String(valB).localeCompare(String(valA));
      });
  }, [districts, search, stateFilter, regimeFilter, categoryFilter, sortField, sortAsc]);

  const handleSort = (field: keyof DistrictForecast) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const handleExportCSV = async () => {
    setIsExporting(true);
    try {
      // Fetch results from backend API endpoint
      const response = await fetch('/api/export/districts-csv');
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'meghdrishti_district_forecasts.csv';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        return;
      }
    } catch (err) {
      console.warn('Backend CSV download fetch failed, falling back to client generation:', err);
    } finally {
      setIsExporting(false);
    }

    // Client-side fallback if backend was unreachable
    if (filteredDistricts.length === 0) return;
    const headers = [
      'District',
      'State',
      'Zone',
      'Regime',
      'Raw NWP (mm)',
      'AI Corrected (mm)',
      'Delta (mm)',
      'Observed (mm)',
      'P(Heavy >=64.5mm)',
      'P(Very Heavy >=115.6mm)',
      'P(Extreme >=204.5mm)',
      'Rainfall Category'
    ];
    const rows = filteredDistricts.map((d) => [
      `"${d.district}"`,
      `"${d.state}"`,
      `"${d.zone}"`,
      `"${d.regime}"`,
      d.raw_nwp_max,
      d.corrected_max,
      d.delta_correction,
      d.observed_mean,
      (d.p_heavy * 100).toFixed(1) + '%',
      (d.p_very_heavy * 100).toFixed(1) + '%',
      (d.p_extreme * 100).toFixed(1) + '%',
      `"${d.category}"`
    ]);

    const csvContent = '\uFEFF' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'meghdrishti_district_forecasts.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl space-y-4 p-5">
      {/* Filters & Export Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-sm font-semibold text-white">District-Level Rainfall Forecast Products</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Operational forecasts, regime attribution, and exceedance risk probabilities across India.
          </p>
        </div>

        <button
          onClick={handleExportCSV}
          disabled={isExporting}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 active:bg-slate-900 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 transition disabled:opacity-50"
          title="Download complete District Forecasts CSV from backend"
        >
          <Download className={`w-3.5 h-3.5 ${isExporting ? 'animate-bounce text-sky-400' : ''}`} />
          {isExporting ? 'Downloading...' : 'Download CSV'}
        </button>
      </div>

      {/* Filter Controls Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        {/* Search */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
          <input
            type="text"
            placeholder="Search district or state..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500"
          />
        </div>

        {/* State Filter */}
        <div>
          <select
            value={stateFilter}
            onChange={(e) => setStateFilter(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500"
          >
            <option value="all">All States ({uniqueStates.length - 1})</option>
            {uniqueStates.filter((s) => s !== 'all').map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        {/* Regime Filter */}
        <div>
          <select
            value={regimeFilter}
            onChange={(e) => setRegimeFilter(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500"
          >
            <option value="all">All Weather Regimes</option>
            {uniqueRegimes.filter((r) => r !== 'all').map((r) => (
              <option key={r} value={r}>{REGIME_LABELS[r] || r}</option>
            ))}
          </select>
        </div>

        {/* Category Filter */}
        <div>
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500"
          >
            <option value="all">All Categories</option>
            <option value="Extremely Heavy">Extremely Heavy (≥ 204.5 mm)</option>
            <option value="Very Heavy">Very Heavy (≥ 115.6 mm)</option>
            <option value="Heavy">Heavy (≥ 64.5 mm)</option>
            <option value="Moderate">Moderate (15.6 - 64.4 mm)</option>
            <option value="Light">Light (2.5 - 15.5 mm)</option>
            <option value="No Rain">No Rain (&lt; 2.5 mm)</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="bg-slate-950 text-slate-300 border-b border-slate-800 font-semibold select-none">
              <th onClick={() => handleSort('district')} className="py-3 px-3 cursor-pointer hover:text-white">
                <div className="flex items-center gap-1">District <ArrowUpDown className="w-3 h-3 text-slate-500" /></div>
              </th>
              <th onClick={() => handleSort('state')} className="py-3 px-3 cursor-pointer hover:text-white">
                <div className="flex items-center gap-1">State <ArrowUpDown className="w-3 h-3 text-slate-500" /></div>
              </th>
              <th onClick={() => handleSort('regime')} className="py-3 px-3 cursor-pointer hover:text-white">
                <div className="flex items-center gap-1">Regime <ArrowUpDown className="w-3 h-3 text-slate-500" /></div>
              </th>
              <th onClick={() => handleSort('raw_nwp_max')} className="py-3 px-3 text-right cursor-pointer hover:text-white">
                <div className="flex items-center justify-end gap-1">Raw NWP <ArrowUpDown className="w-3 h-3 text-slate-500" /></div>
              </th>
              <th onClick={() => handleSort('corrected_max')} className="py-3 px-3 text-right cursor-pointer hover:text-white">
                <div className="flex items-center justify-end gap-1">AI Corrected <ArrowUpDown className="w-3 h-3 text-slate-500" /></div>
              </th>
              <th onClick={() => handleSort('delta_correction')} className="py-3 px-3 text-right cursor-pointer hover:text-white">
                <div className="flex items-center justify-end gap-1">Delta Δ <ArrowUpDown className="w-3 h-3 text-slate-500" /></div>
              </th>
              <th onClick={() => handleSort('p_heavy')} className="py-3 px-3 text-right cursor-pointer hover:text-white">
                <div className="flex items-center justify-end gap-1">P(Heavy) <ArrowUpDown className="w-3 h-3 text-slate-500" /></div>
              </th>
              <th onClick={() => handleSort('p_very_heavy')} className="py-3 px-3 text-right cursor-pointer hover:text-white">
                <div className="flex items-center justify-end gap-1">P(V.Heavy) <ArrowUpDown className="w-3 h-3 text-slate-500" /></div>
              </th>
              <th className="py-3 px-3 text-center">Category</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 bg-slate-900/60">
            {filteredDistricts.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-8 text-center text-slate-500">
                  No districts match the filter criteria.
                </td>
              </tr>
            ) : (
              filteredDistricts.map((d) => {
                const regColor = REGIME_COLORS[d.regime] || '#64748b';
                const regLabel = REGIME_LABELS[d.regime] || d.regime;
                const isHeavy = d.corrected_max >= 64.5;
                const isVeryHeavy = d.corrected_max >= 115.6;

                return (
                  <tr
                    key={d.district}
                    onClick={() => onSelectDistrict(d)}
                    className="hover:bg-slate-800/60 cursor-pointer transition-colors"
                  >
                    <td className="py-2.5 px-3 font-medium text-white flex items-center gap-1.5">
                      {isVeryHeavy && <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />}
                      {isHeavy && !isVeryHeavy && <span className="w-2 h-2 rounded-full bg-amber-500" />}
                      {d.district}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">{d.state}</td>
                    <td className="py-2.5 px-3">
                      <span
                        className="px-2 py-0.5 rounded text-[11px] font-semibold"
                        style={{ backgroundColor: `${regColor}20`, color: regColor }}
                      >
                        {regLabel}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-sky-400 font-semibold">{d.raw_nwp_max}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-emerald-300 font-bold">{d.corrected_max}</td>
                    <td className="py-2.5 px-3 text-right font-mono">
                      <span className={d.delta_correction > 0 ? 'text-emerald-400' : d.delta_correction < 0 ? 'text-red-400' : 'text-slate-400'}>
                        {d.delta_correction > 0 ? `+${d.delta_correction}` : d.delta_correction}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono">
                      <span className={d.p_heavy >= 0.5 ? 'text-amber-400 font-bold' : 'text-slate-400'}>
                        {(d.p_heavy * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono">
                      <span className={d.p_very_heavy >= 0.4 ? 'text-rose-400 font-bold' : 'text-slate-400'}>
                        {(d.p_very_heavy * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          d.category === 'Extremely Heavy'
                            ? 'bg-purple-950 text-purple-300 border border-purple-800'
                            : d.category === 'Very Heavy'
                            ? 'bg-rose-950 text-rose-300 border border-rose-800'
                            : d.category === 'Heavy'
                            ? 'bg-amber-950 text-amber-300 border border-amber-800'
                            : d.category === 'Moderate'
                            ? 'bg-emerald-950 text-emerald-300'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {d.category}
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
      <div className="text-[11px] text-slate-500">
        Showing {filteredDistricts.length} of {districts.length} meteorological districts. Click any row to view complete Explainable AI breakdown.
      </div>
    </div>
  );
};

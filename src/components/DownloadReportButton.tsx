import React, { useState, useRef, useEffect } from 'react';
import { Download, FileSpreadsheet, FileJson, ChevronDown, Check } from 'lucide-react';
import { SummaryMetrics, DistrictForecast } from '../types';

interface DownloadReportButtonProps {
  forecastDate: string;
  leadTime: string;
  nwpProvider: string;
  metrics?: SummaryMetrics | null;
  districts?: DistrictForecast[];
}

export function DownloadReportButton({
  forecastDate,
  leadTime,
  nwpProvider,
  metrics,
  districts
}: DownloadReportButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [downloading, setDownloading] = useState<'csv' | 'json' | null>(null);
  const [downloadSuccess, setDownloadSuccess] = useState<'csv' | 'json' | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const handleDownload = async (format: 'csv' | 'json') => {
    setDownloading(format);
    setIsOpen(false);

    try {
      const cleanLead = leadTime.replace('+', '').replace('h', '');
      const url = `/api/export/report?format=${format}&date=${encodeURIComponent(forecastDate)}&lead_time=${encodeURIComponent(cleanLead)}&nwp=${encodeURIComponent(nwpProvider)}`;
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch report: ${response.statusText}`);
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `meghdrishti_monsoon_forecast_report_${forecastDate}_${cleanLead}h.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);

      setDownloadSuccess(format);
      setTimeout(() => setDownloadSuccess(null), 3000);
    } catch (error) {
      console.error(`Error downloading ${format} report:`, error);
      
      // Fallback: If network or server fails, generate from current client state
      if (format === 'json') {
        const clientReport = {
          report_title: "MEGHDRISHTI - Regime-Aware AI Monsoon Rainfall Forecast Report",
          exported_at: new Date().toISOString(),
          dashboard_state: {
            forecast_date: forecastDate,
            lead_time_hours: parseInt(leadTime.replace('+', '').replace('h', ''), 10) || 24,
            nwp_provider: nwpProvider,
            total_districts: districts?.length ?? 0
          },
          verification_benchmarks: {
            model_comparison: metrics?.model_comparison?.comparison_table || [],
            regime_wise_verification: metrics?.regime_wise_verification?.regime_breakdown || [],
            fss_spatial_curve: metrics?.fss_spatial_curve || {}
          },
          district_forecasts: districts || []
        };

        const blob = new Blob([JSON.stringify(clientReport, null, 2)], { type: 'application/json' });
        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `meghdrishti_monsoon_forecast_report_${forecastDate}_${leadTime}h.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(downloadUrl);
        setDownloadSuccess('json');
        setTimeout(() => setDownloadSuccess(null), 3000);
      }
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div className="relative inline-block text-xs" ref={menuRef}>
      {/* Primary Split / Action Trigger */}
      <div className="flex items-center rounded-lg shadow-sm">
        <button
          type="button"
          onClick={() => handleDownload('csv')}
          disabled={downloading !== null}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-100 font-semibold rounded-l-lg border-y border-l border-slate-700 hover:border-slate-600 transition disabled:opacity-50 cursor-pointer"
          title="Download complete structured report for current dashboard state as CSV"
        >
          {downloadSuccess ? (
            <Check className="w-3.5 h-3.5 text-emerald-400" />
          ) : (
            <Download className={`w-3.5 h-3.5 text-sky-400 ${downloading ? 'animate-bounce' : ''}`} />
          )}
          <span>
            {downloading ? `Exporting ${downloading.toUpperCase()}...` : 'Download Report'}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          aria-haspopup="menu"
          aria-label="Select report format"
          className="px-2 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-r-lg border border-slate-700 hover:border-slate-600 transition cursor-pointer"
        >
          <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${isOpen ? 'rotate-180 text-sky-400' : ''}`} />
        </button>
      </div>

      {/* Format Selection Dropdown */}
      {isOpen && (
        <div
          role="menu"
          aria-orientation="vertical"
          className="absolute right-0 mt-2 w-64 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl z-50 p-2 text-slate-200 backdrop-blur-md animate-in fade-in zoom-in-95 duration-150"
        >
          <div className="px-2.5 py-1.5 border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-400 font-semibold">
            Select Export Format ({forecastDate} · +{leadTime.replace('+', '').replace('h', '')}h)
          </div>

          <div className="mt-1 space-y-1">
            <button
              type="button"
              role="menuitem"
              onClick={() => handleDownload('csv')}
              className="w-full flex items-start gap-2.5 px-2.5 py-2 rounded-lg hover:bg-slate-800/80 transition text-left cursor-pointer group"
            >
              <div className="p-1.5 rounded-md bg-emerald-500/10 text-emerald-400 group-hover:bg-emerald-500/20 shrink-0 mt-0.5">
                <FileSpreadsheet className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-semibold text-slate-100 flex items-center justify-between">
                  <span>Structured CSV</span>
                  <span className="text-[10px] text-slate-400 font-normal">Excel / Sheets</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-tight mt-0.5">
                  Multi-section spreadsheet with district forecasts, probability thresholds & verification benchmarks.
                </p>
              </div>
            </button>

            <button
              type="button"
              role="menuitem"
              onClick={() => handleDownload('json')}
              className="w-full flex items-start gap-2.5 px-2.5 py-2 rounded-lg hover:bg-slate-800/80 transition text-left cursor-pointer group"
            >
              <div className="p-1.5 rounded-md bg-sky-500/10 text-sky-400 group-hover:bg-sky-500/20 shrink-0 mt-0.5">
                <FileJson className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-semibold text-slate-100 flex items-center justify-between">
                  <span>Structured JSON</span>
                  <span className="text-[10px] text-slate-400 font-normal">API / GIS Ready</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-tight mt-0.5">
                  Complete hierarchical payload with active dashboard state, metrics, alerts, and model provenance.
                </p>
              </div>
            </button>
          </div>

          <div className="mt-2 pt-2 border-t border-slate-800/80 px-2.5 py-1 text-[10px] text-slate-400 flex items-center justify-between">
            <span>Provider: {nwpProvider}</span>
            <span>Districts: {districts?.length ?? 729}</span>
          </div>
        </div>
      )}
    </div>
  );
}
export default DownloadReportButton;

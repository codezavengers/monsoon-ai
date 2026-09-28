import React, { useState, useEffect, useRef } from 'react';
import { NwpFreshnessData } from '../types';
import { 
  Clock, 
  AlertTriangle, 
  CheckCircle2, 
  ChevronDown, 
  RefreshCw, 
  Radio, 
  Activity, 
  Server,
  Zap
} from 'lucide-react';

interface DataFreshnessIndicatorProps {
  nwpProvider: string;
}

export function DataFreshnessIndicator({ nwpProvider }: DataFreshnessIndicatorProps) {
  const [freshnessData, setFreshnessData] = useState<NwpFreshnessData | null>(null);
  const [loading, setLoading] = useState(true);
  const [isOpen, setIsOpen] = useState(false);
  const [simulateFresh, setSimulateFresh] = useState(false);
  const [currentClientTime, setCurrentClientTime] = useState<Date>(new Date());
  const popoverRef = useRef<HTMLDivElement>(null);

  // Fetch NWP cycle freshness data
  const fetchFreshness = async (simulate = simulateFresh) => {
    try {
      setLoading(true);
      const prov = nwpProvider.includes('ECMWF') ? 'ECMWF' : nwpProvider.includes('NCMRWF') ? 'NCMRWF' : 'GFS';
      const simParam = simulate ? '&simulate=fresh' : '';
      const res = await fetch(`/api/nwp/freshness?provider=${prov}${simParam}`);
      const json = await res.json();
      if (json.success && json.data) {
        setFreshnessData(json.data);
      }
    } catch (e) {
      console.error('Failed to load NWP freshness:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFreshness(simulateFresh);
  }, [nwpProvider, simulateFresh]);

  // Live timer: re-calculate difference against real-time system clock every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentClientTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Click outside listener
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
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

  // Compute live time difference in milliseconds between current system time and latest cycle
  const latestCycleTimestampStr = freshnessData?.latest_cycle_timestamp || '2024-07-15T00:00:00Z';
  const cycleDate = new Date(latestCycleTimestampStr);
  const diffMs = Math.max(0, currentClientTime.getTime() - cycleDate.getTime());
  const diffTotalSeconds = Math.floor(diffMs / 1000);
  const diffTotalMinutes = Math.floor(diffTotalSeconds / 60);
  const diffTotalHours = parseFloat((diffMs / (3600 * 1000)).toFixed(1));

  // Determine if older than 24 hours
  const isOlderThan24h = diffTotalHours > 24.0;

  // Breakdown formatting (days, hours, minutes, seconds)
  const formatTimeDifference = () => {
    const days = Math.floor(diffTotalSeconds / 86400);
    const hours = Math.floor((diffTotalSeconds % 86400) / 3600);
    const minutes = Math.floor((diffTotalSeconds % 3600) / 60);
    const seconds = diffTotalSeconds % 60;

    if (days > 0) {
      return `${days}d ${hours}h lag`;
    }
    if (hours > 0) {
      return `${hours}h ${minutes}m lag`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds}s lag`;
    }
    return `${seconds}s lag`;
  };

  // Human readable timestamp
  const formatIsoDate = (isoStr: string) => {
    try {
      const d = new Date(isoStr);
      return new Intl.DateTimeFormat('en-GB', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZoneName: 'short'
      }).format(d);
    } catch {
      return isoStr;
    }
  };

  return (
    <div className="relative inline-block text-xs" ref={popoverRef}>
      {/* Trigger Button: Turns RED if data is older than 24 hours, GREEN if fresh */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        title={`Latest NWP Cycle: ${latestCycleTimestampStr}. Difference: ${diffTotalHours}h. Click to view data freshness details.`}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all cursor-pointer text-left ${
          isOlderThan24h
            ? isOpen 
              ? 'bg-rose-950 border-rose-500 text-rose-100 shadow-md ring-1 ring-rose-500/30'
              : 'bg-rose-950/70 hover:bg-rose-900/80 border-rose-800 hover:border-rose-700 text-rose-200 shadow-sm'
            : isOpen
              ? 'bg-emerald-950 border-emerald-500 text-emerald-100 shadow-md ring-1 ring-emerald-500/30'
              : 'bg-emerald-950/70 hover:bg-emerald-900/80 border-emerald-800 hover:border-emerald-700 text-emerald-200 shadow-sm'
        }`}
      >
        {/* Pulsing Status Indicator Dot (Red when >24h, Green when <=24h) */}
        <span className="relative flex h-2 w-2">
          <span
            className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
              isOlderThan24h ? 'bg-rose-400' : 'bg-emerald-400'
            }`}
          />
          <span
            className={`relative inline-flex rounded-full h-2 w-2 ${
              isOlderThan24h ? 'bg-rose-500' : 'bg-emerald-500'
            }`}
          />
        </span>

        {/* Clean Unboxed Text Content */}
        <div className="flex items-center gap-1.5 whitespace-nowrap">
          <span className="font-semibold">
            {isOlderThan24h ? 'Data Stale' : 'Data Fresh'}
          </span>
          <span className={`${isOlderThan24h ? 'text-rose-400/60' : 'text-emerald-400/60'}`} aria-hidden="true">·</span>
          <span className="font-mono tabular-nums font-medium">
            {formatTimeDifference()}
          </span>
          {isOlderThan24h && (
            <span className="hidden sm:inline text-[10px] uppercase font-bold text-rose-300 bg-rose-900/60 px-1 py-0.5 rounded border border-rose-700/60">
              &gt;24h
            </span>
          )}
        </div>

        <ChevronDown
          className={`w-3.5 h-3.5 transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          } ${isOlderThan24h ? 'text-rose-400' : 'text-emerald-400'}`}
        />
      </button>

      {/* Popover Breakdown Dialog */}
      {isOpen && (
        <div
          role="dialog"
          aria-label="NWP Data Freshness and Initialization Cycle Monitor"
          className="absolute right-0 mt-2 w-80 sm:w-96 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl z-50 p-4 text-slate-200 backdrop-blur-md animate-in fade-in zoom-in-95 duration-150"
        >
          {/* Header */}
          <div className="flex items-start justify-between border-b border-slate-800 pb-3 mb-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-slate-100 text-sm">
                  NWP Data Freshness
                </span>
                <span
                  className={`text-[10px] font-bold px-1.5 py-0.5 rounded border font-mono ${
                    isOlderThan24h
                      ? 'bg-rose-950 text-rose-300 border-rose-800'
                      : 'bg-emerald-950 text-emerald-300 border-emerald-800'
                  }`}
                >
                  {isOlderThan24h ? 'STALE (>24H)' : 'FRESH (<=24H)'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Current system time vs. latest NWP cycle initialization
              </p>
            </div>

            <button
              type="button"
              onClick={() => fetchFreshness(simulateFresh)}
              disabled={loading}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer disabled:opacity-50"
              title="Refresh freshness check"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* Freshness Status Alert Card */}
          <div
            className={`p-3 rounded-lg border mb-3 flex items-start gap-2.5 ${
              isOlderThan24h
                ? 'bg-rose-950/40 border-rose-800/80 text-rose-200'
                : 'bg-emerald-950/40 border-emerald-800/80 text-emerald-200'
            }`}
          >
            {isOlderThan24h ? (
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            ) : (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            )}
            <div className="text-xs">
              <div className="font-semibold">
                {isOlderThan24h
                  ? 'Data Freshness SLA Exceeded'
                  : 'NWP Cycle Fresh & Up-to-Date'}
              </div>
              <p className="text-[11px] mt-0.5 opacity-90 leading-tight">
                {isOlderThan24h
                  ? `Cycle timestamp is ${diffTotalHours} hours old (> 24h threshold). In operational mode, upstream NWP runs should be ingested every 6 to 12 hours.`
                  : `Cycle timestamp is within the 24-hour latency SLA (${diffTotalHours} hours old). All forecast fields are physically current.`}
              </p>
            </div>
          </div>

          {/* Time Difference Calculations */}
          <div className="space-y-2 mb-3 bg-slate-950/60 p-3 rounded-lg border border-slate-800">
            {/* Live System Time */}
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-sky-400" />
                Current System Time
              </span>
              <span className="font-mono text-slate-200 tabular-nums font-medium text-[11px]">
                {formatIsoDate(currentClientTime.toISOString())}
              </span>
            </div>

            {/* Latest NWP Cycle Timestamp */}
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 flex items-center gap-1.5">
                <Radio className="w-3.5 h-3.5 text-indigo-400" />
                Latest NWP Cycle ({nwpProvider})
              </span>
              <span className="font-mono text-slate-200 tabular-nums font-medium text-[11px]">
                {formatIsoDate(latestCycleTimestampStr)}
              </span>
            </div>

            {/* Time Difference */}
            <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-xs">
              <span className="text-slate-300 font-semibold flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-amber-400" />
                Calculated Time Difference
              </span>
              <span
                className={`font-mono tabular-nums font-bold text-xs ${
                  isOlderThan24h ? 'text-rose-400' : 'text-emerald-400'
                }`}
              >
                {diffTotalHours.toLocaleString()} hrs ({formatTimeDifference()})
              </span>
            </div>

            <div className="flex items-center justify-between text-[10px] text-slate-400">
              <span>Threshold Limit:</span>
              <span className="font-mono font-medium text-slate-300">24.0 Hours Max</span>
            </div>
          </div>

          {/* Provider Status Details */}
          <div className="space-y-1.5 mb-3 border-t border-slate-800 pt-3">
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-medium">
              <span className="flex items-center gap-1.5 text-slate-300">
                <Server className="w-3.5 h-3.5 text-slate-400" />
                Upstream Provider Cycles
              </span>
              <span className="text-[10px] text-slate-400 font-mono">00Z / 03Z Active</span>
            </div>

            <div className="space-y-1 text-xs">
              {[
                { name: 'GFS 0.25° (NOAA)', cycle: '2024-07-15T00:00:00Z', latency: '1.2s' },
                { name: 'ECMWF HRES', cycle: '2024-07-15T00:00:00Z', latency: '2.1s' },
                { name: 'NCMRWF NCUM', cycle: '2024-07-15T00:00:00Z', latency: '1.8s' },
                { name: 'IMD Rain Gauges', cycle: '2024-07-15T03:00:00Z', latency: '0.9s' }
              ].map((p) => (
                <div
                  key={p.name}
                  className="flex items-center justify-between py-1 px-2 rounded bg-slate-950/40 border border-slate-800/40 text-[11px]"
                >
                  <span className="text-slate-300">{p.name}</span>
                  <div className="flex items-center gap-2 font-mono text-[10px] text-slate-400 tabular-nums">
                    <span>{simulateFresh ? 'Today 00Z' : '15 Jul 00Z'}</span>
                    <span className="text-emerald-400">Online</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Interactive Simulation Action to Test Both States */}
          <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
            <div className="text-[11px] text-slate-400">
              <span>Test SLA States:</span>
            </div>
            <button
              type="button"
              onClick={() => {
                const nextState = !simulateFresh;
                setSimulateFresh(nextState);
              }}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-semibold transition cursor-pointer border ${
                simulateFresh
                  ? 'bg-rose-950 text-rose-300 border-rose-800 hover:bg-rose-900'
                  : 'bg-emerald-950 text-emerald-300 border-emerald-800 hover:bg-emerald-900'
              }`}
            >
              <Zap className="w-3 h-3" />
              {simulateFresh ? 'Revert to Benchmark (>24h Stale)' : 'Simulate Fresh Cycle (<24h)'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default DataFreshnessIndicator;

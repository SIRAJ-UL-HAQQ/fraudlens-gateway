import React, { useState } from 'react';
import { Activity, ShieldCheck, Cpu, Clock, HelpCircle } from 'lucide-react';

export function OperationalMetricsCard({ metrics }) {
  const [showLatencyTooltip, setShowLatencyTooltip] = useState(false);

  if (!metrics) return null;

  const total = typeof metrics.total_transactions === 'number'
    ? metrics.total_transactions.toLocaleString()
    : '—';

  const stabilityPct = typeof metrics.explanation_stability_ratio === 'number'
    ? `${(metrics.explanation_stability_ratio * 100).toFixed(1)}%`
    : '—';

  const avgRisk = typeof metrics.average_risk_score === 'number'
    ? metrics.average_risk_score.toFixed(3)
    : '—';

  const p50 = typeof metrics.latency_p50_ms === 'number'
    ? `${metrics.latency_p50_ms.toFixed(1)} ms`
    : '—';

  const p95 = typeof metrics.latency_p95_ms === 'number'
    ? `${metrics.latency_p95_ms.toFixed(1)} ms`
    : '—';

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" data-testid="operational-metrics">
      {/* Total Scored */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Total Scored</span>
          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400">
            <Activity className="w-4 h-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold text-white tracking-tight">{total}</div>
          <p className="text-xs text-slate-400 mt-1">Live ingested transactions</p>
        </div>
      </div>

      {/* SHAP Explanation Stability */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">SHAP Stability</span>
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400">
            <ShieldCheck className="w-4 h-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold text-emerald-400 tracking-tight">{stabilityPct}</div>
          <p className="text-xs text-slate-400 mt-1">Consistent feature ranking across models</p>
        </div>
      </div>

      {/* Average Risk Score */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Mean Risk Score</span>
          <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center text-amber-400">
            <Cpu className="w-4 h-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold text-white tracking-tight">{avgRisk}</div>
          <p className="text-xs text-slate-400 mt-1">Gateway fraud probability mean</p>
        </div>
      </div>

      {/* Latency Percentiles (p50 / p95) with interactive tooltip */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between shadow-sm relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Latency (p50 / p95)</span>
            <div className="relative inline-block">
              <button
                type="button"
                onMouseEnter={() => setShowLatencyTooltip(true)}
                onMouseLeave={() => setShowLatencyTooltip(false)}
                onFocus={() => setShowLatencyTooltip(true)}
                onBlur={() => setShowLatencyTooltip(false)}
                aria-label="Latency calculation details"
                className="text-slate-400 hover:text-slate-200 focus:outline-none"
              >
                <HelpCircle className="w-3.5 h-3.5" />
              </button>
              {showLatencyTooltip && (
                <div
                  role="tooltip"
                  className="absolute z-20 left-1/2 -translate-x-1/2 bottom-full mb-2 w-52 p-2 bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-md shadow-lg pointer-events-none text-center"
                >
                  Server-side scoring + explanation time
                </div>
              )}
            </div>
          </div>
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 flex items-center justify-center text-cyan-400">
            <Clock className="w-4 h-4" />
          </div>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold text-white tracking-tight">
            {p50} <span className="text-sm font-normal text-slate-400">/</span> <span className="text-lg font-semibold text-slate-300">{p95}</span>
          </div>
          <p className="text-xs text-slate-400 mt-1">End-to-end inference latency</p>
        </div>
      </div>
    </div>
  );
}

export default OperationalMetricsCard;

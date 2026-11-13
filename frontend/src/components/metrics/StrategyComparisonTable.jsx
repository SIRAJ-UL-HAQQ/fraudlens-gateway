import React from 'react';
import { Layers } from 'lucide-react';
import { parseStrategyData } from './strategyUtils';

export function StrategyComparisonTable({ strategyComparison }) {
  if (!strategyComparison || Object.keys(strategyComparison).length === 0) {
    return null;
  }

  const order = ['none', 'class_weight', 'smote_enn'];
  const keys = [
    ...order.filter((k) => k in strategyComparison),
    ...Object.keys(strategyComparison).filter((k) => !order.includes(k)),
  ];

  const rows = keys
    .map((k) => parseStrategyData(k, strategyComparison[k]))
    .filter(Boolean);

  const formatScore = (val) =>
    typeof val === 'number' && !Number.isNaN(val) ? (val * 100).toFixed(1) + '%' : '—';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm" data-testid="strategy-comparison-table">
      <div className="flex items-center gap-2 pb-3 border-b border-slate-800">
        <Layers className="w-4 h-4 text-indigo-400" />
        <h2 className="text-sm font-semibold text-white">Imbalance Strategy Benchmark</h2>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400">
              <th className="py-2.5 px-3 font-semibold">Strategy</th>
              <th className="py-2.5 px-3 font-semibold text-right">Precision</th>
              <th className="py-2.5 px-3 font-semibold text-right">Recall</th>
              <th className="py-2.5 px-3 font-semibold text-right">F1 Score</th>
              <th className="py-2.5 px-3 font-semibold text-right">PR-AUC</th>
              <th className="py-2.5 px-3 font-semibold text-right">SHAP Stability</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {rows.map((row) => (
              <tr
                key={row.key}
                className={
                  row.isDeployed
                    ? 'bg-indigo-950/20 font-medium text-slate-200'
                    : 'text-slate-300 hover:bg-slate-800/30'
                }
                data-testid={`strategy-row-${row.key}`}
              >
                <td className="py-3 px-3 flex items-center gap-2">
                  <span>{row.label}</span>
                  {row.isDeployed && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                      Deployed
                    </span>
                  )}
                </td>
                <td className="py-3 px-3 text-right font-mono text-slate-200">{formatScore(row.precision)}</td>
                <td className="py-3 px-3 text-right font-mono text-slate-200">{formatScore(row.recall)}</td>
                <td className="py-3 px-3 text-right font-mono text-slate-200">{formatScore(row.f1)}</td>
                <td className="py-3 px-3 text-right font-mono text-slate-200">{formatScore(row.prAuc)}</td>
                <td className="py-3 px-3 text-right font-mono">
                  {typeof row.stability === 'number' && !Number.isNaN(row.stability) ? (
                    <span className={row.stability >= 0.85 ? 'text-emerald-400' : 'text-amber-400'}>
                      {formatScore(row.stability)}
                    </span>
                  ) : (
                    '—'
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default StrategyComparisonTable;

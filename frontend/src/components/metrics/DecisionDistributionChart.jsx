import React from 'react';
import { PieChart } from 'lucide-react';

export function DecisionDistributionChart({ distribution = {} }) {
  const lowRisk = distribution['Low Risk'] || distribution.low_risk || 0;
  const analystReview = distribution['Analyst Review'] || distribution.analyst_review || 0;
  const highRisk = distribution['High Risk'] || distribution.high_risk || 0;
  const total = lowRisk + analystReview + highRisk;

  const lowPct = total > 0 ? (lowRisk / total) * 100 : 0;
  const reviewPct = total > 0 ? (analystReview / total) * 100 : 0;
  const highPct = total > 0 ? (highRisk / total) * 100 : 0;

  // Chart SVG coordinates for stacked bar
  // SVG viewBox: 0 0 600 28
  const width = 600;
  const lowWidth = total > 0 ? (lowRisk / total) * width : 0;
  const reviewWidth = total > 0 ? (analystReview / total) * width : 0;
  const highWidth = total > 0 ? (highRisk / total) * width : 0;

  const descText = `Decision distribution: Low Risk ${lowRisk} (${lowPct.toFixed(1)}%), Analyst Review ${analystReview} (${reviewPct.toFixed(1)}%), High Risk ${highRisk} (${highPct.toFixed(1)}%). Total: ${total}.`;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm" data-testid="decision-distribution-card">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <PieChart className="w-4 h-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-white">Decision Routing Distribution</h2>
        </div>
        <span className="text-xs text-slate-400">Total: {total.toLocaleString()}</span>
      </div>

      <div className="mt-4">
        {/* Accessible SVG Stacked Bar */}
        <div className="w-full">
          <svg
            viewBox="0 0 600 28"
            className="w-full h-7 rounded-lg overflow-hidden bg-slate-800"
            role="img"
            aria-label="Decision distribution chart"
          >
            <title>Decision Distribution</title>
            <desc>{descText}</desc>
            {total === 0 ? (
              <rect x="0" y="0" width="600" height="28" fill="#334155" />
            ) : (
              <>
                {lowWidth > 0 && (
                  <rect
                    x="0"
                    y="0"
                    width={lowWidth}
                    height="28"
                    fill="#10b981"
                    data-testid="dist-bar-low"
                  />
                )}
                {reviewWidth > 0 && (
                  <rect
                    x={lowWidth}
                    y="0"
                    width={reviewWidth}
                    height="28"
                    fill="#f59e0b"
                    data-testid="dist-bar-review"
                  />
                )}
                {highWidth > 0 && (
                  <rect
                    x={lowWidth + reviewWidth}
                    y="0"
                    width={highWidth}
                    height="28"
                    fill="#f43f5e"
                    data-testid="dist-bar-high"
                  />
                )}
              </>
            )}
          </svg>
        </div>

        {/* Legend / Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
          {/* Low Risk */}
          <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shrink-0" />
              <span className="text-xs font-medium text-slate-300">Low Risk</span>
            </div>
            <div className="text-right">
              <div className="text-sm font-bold text-white">{lowRisk.toLocaleString()}</div>
              <div className="text-[11px] text-slate-400">{lowPct.toFixed(1)}%</div>
            </div>
          </div>

          {/* Analyst Review */}
          <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shrink-0" />
              <span className="text-xs font-medium text-slate-300">Analyst Review</span>
            </div>
            <div className="text-right">
              <div className="text-sm font-bold text-white">{analystReview.toLocaleString()}</div>
              <div className="text-[11px] text-slate-400">{reviewPct.toFixed(1)}%</div>
            </div>
          </div>

          {/* High Risk */}
          <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shrink-0" />
              <span className="text-xs font-medium text-slate-300">High Risk</span>
            </div>
            <div className="text-right">
              <div className="text-sm font-bold text-white">{highRisk.toLocaleString()}</div>
              <div className="text-[11px] text-slate-400">{highPct.toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DecisionDistributionChart;

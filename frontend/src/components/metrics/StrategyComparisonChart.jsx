import React from 'react';
import { BarChart3 } from 'lucide-react';
import { parseStrategyData } from './strategyUtils';

export function StrategyComparisonChart({ strategyComparison }) {
  if (!strategyComparison || Object.keys(strategyComparison).length === 0) {
    return null;
  }

  const order = ['none', 'class_weight', 'smote_enn'];
  const keys = [
    ...order.filter((k) => k in strategyComparison),
    ...Object.keys(strategyComparison).filter((k) => !order.includes(k)),
  ];

  const parsedStrategies = keys
    .map((k) => parseStrategyData(k, strategyComparison[k]))
    .filter(Boolean);

  if (parsedStrategies.length === 0) return null;

  // Metrics to compare on X-axis: Precision, Recall, F1, PR-AUC
  const metrics = [
    { key: 'precision', label: 'Precision' },
    { key: 'recall', label: 'Recall' },
    { key: 'f1', label: 'F1 Score' },
    { key: 'prAuc', label: 'PR-AUC' },
  ];

  const strategyColors = {
    none: '#64748b', // Slate 500
    class_weight: '#6366f1', // Indigo 500
    smote_enn: '#06b6d4', // Cyan 500
  };

  // Dimensions for SVG
  const width = 640;
  const height = 220;
  const paddingLeft = 45;
  const paddingRight = 20;
  const paddingTop = 20;
  const paddingBottom = 35;
  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;

  const groupWidth = chartWidth / metrics.length;
  const barWidth = Math.min(22, (groupWidth - 24) / parsedStrategies.length);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm" data-testid="strategy-comparison-chart">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800 gap-2">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-white">Strategy Metric Comparison</h2>
        </div>
        {/* Legend */}
        <div className="flex items-center gap-4 text-xs">
          {parsedStrategies.map((strat) => (
            <div key={strat.key} className="flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-sm shrink-0"
                style={{ backgroundColor: strategyColors[strat.key] || '#94a3b8' }}
              />
              <span className="text-slate-300 font-medium">
                {strat.key === 'class_weight' ? 'Class Weight (Deployed)' : strat.label.split(' ')[0]}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 w-full">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto overflow-visible"
          role="img"
          aria-label="Strategy performance comparison bar chart"
        >
          <title>Strategy Metric Comparison</title>
          <desc>
            Bar chart comparing Precision, Recall, F1, and PR-AUC across imbalance handling strategies: None, Class Weight, and SMOTE-ENN.
          </desc>

          {/* Grid lines (0%, 25%, 50%, 75%, 100%) */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((level) => {
            const y = paddingTop + chartHeight - level * chartHeight;
            return (
              <g key={level}>
                <line
                  x1={paddingLeft}
                  y1={y}
                  x2={width - paddingRight}
                  y2={y}
                  stroke="#334155"
                  strokeDasharray={level === 0 ? '0' : '2 2'}
                  strokeWidth="1"
                />
                <text
                  x={paddingLeft - 8}
                  y={y + 3}
                  textAnchor="end"
                  fill="#94a3b8"
                  fontSize="10"
                  fontFamily="sans-serif"
                >
                  {Math.round(level * 100)}%
                </text>
              </g>
            );
          })}

          {/* Metric Groups */}
          {metrics.map((metric, metricIdx) => {
            const groupX = paddingLeft + metricIdx * groupWidth;
            const groupCenterX = groupX + groupWidth / 2;
            const totalBarsWidth = parsedStrategies.length * barWidth;
            const startX = groupCenterX - totalBarsWidth / 2;

            return (
              <g key={metric.key}>
                {/* Metric X-axis Label */}
                <text
                  x={groupCenterX}
                  y={height - 12}
                  textAnchor="middle"
                  fill="#cbd5e1"
                  fontSize="11"
                  fontWeight="500"
                  fontFamily="sans-serif"
                >
                  {metric.label}
                </text>

                {/* Bars for each strategy */}
                {parsedStrategies.map((strat, stratIdx) => {
                  const val = strat[metric.key] ?? 0;
                  const barH = Math.max(2, val * chartHeight);
                  const barX = startX + stratIdx * barWidth;
                  const barY = paddingTop + chartHeight - barH;
                  const color = strategyColors[strat.key] || '#94a3b8';

                  return (
                    <rect
                      key={`${metric.key}-${strat.key}`}
                      x={barX + 1}
                      y={barY}
                      width={barWidth - 2}
                      height={barH}
                      fill={color}
                      rx="2"
                      data-testid={`bar-${metric.key}-${strat.key}`}
                    >
                      <title>{`${strat.label} ${metric.label}: ${(val * 100).toFixed(1)}%`}</title>
                    </rect>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export default StrategyComparisonChart;

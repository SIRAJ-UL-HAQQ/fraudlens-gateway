import React, { useState, useEffect } from 'react';
import { BarChart2, RefreshCw, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { getMetrics } from '../api/client';
import OperationalMetricsCard from '../components/metrics/OperationalMetricsCard';
import DecisionDistributionChart from '../components/metrics/DecisionDistributionChart';
import StrategyComparisonTable from '../components/metrics/StrategyComparisonTable';
import StrategyComparisonChart from '../components/metrics/StrategyComparisonChart';
import ConfusionMatrixGrid from '../components/metrics/ConfusionMatrixGrid';

export function MetricsView() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchMetricsData = async (isManual = false) => {
    if (isManual) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch (err) {
      setError(err.message || 'Failed to load pipeline metrics');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchMetricsData();
  }, []);

  const extractConfusionMatrix = (mlMetrics) => {
    if (!mlMetrics) return null;
    if (mlMetrics.confusion_matrix) {
      return mlMetrics.confusion_matrix;
    }
    if (mlMetrics.evaluation_metrics?.confusion_matrix) {
      return mlMetrics.evaluation_metrics.confusion_matrix;
    }
    const strat = mlMetrics.strategy_comparison?.class_weight || mlMetrics.strategy_comparison?.['class_weight (deployed)'];
    if (strat?.metrics_at_high?.confusion_matrix) {
      return strat.metrics_at_high.confusion_matrix;
    }
    return null;
  };

  const isMLAvailable = metrics?.ml_metrics_status === 'available' && metrics?.ml_metrics;
  const confusionMatrix = extractConfusionMatrix(metrics?.ml_metrics);

  return (
    <div className="space-y-6" data-testid="metrics-view">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-slate-800 gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <BarChart2 className="w-5 h-5 text-indigo-400" />
            Model & Pipeline Metrics
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Imbalance strategy benchmark, latency percentiles, and explanation stability.
          </p>
        </div>

        <button
          type="button"
          onClick={() => fetchMetricsData(true)}
          disabled={loading || refreshing}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-3">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading state */}
      {loading ? (
        <div className="py-20 text-center text-slate-400 text-sm flex flex-col items-center justify-center gap-3">
          <RefreshCw className="w-6 h-6 animate-spin text-indigo-400" />
          <span>Loading pipeline telemetry...</span>
        </div>
      ) : metrics ? (
        <>
          {/* Section 1: Operational Telemetry (Always rendered) */}
          <section className="space-y-4">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Gateway Operational Telemetry
            </h2>
            <OperationalMetricsCard metrics={metrics} />
            <DecisionDistributionChart distribution={metrics.decision_distribution} />
          </section>

          {/* Section 2: Machine Learning Evaluation */}
          <section className="space-y-4 pt-4 border-t border-slate-800/80">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Offline Research Benchmark (PaySim Test Split)
              </h2>
              {isMLAvailable && (
                <div className="flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Benchmark Loaded</span>
                </div>
              )}
            </div>

            {isMLAvailable ? (
              <div className="space-y-5" data-testid="ml-metrics-section">
                {/* Imbalance Strategy Benchmark Table */}
                <StrategyComparisonTable strategyComparison={metrics.ml_metrics.strategy_comparison} />

                {/* Strategy Metric Comparison SVG Chart */}
                <StrategyComparisonChart strategyComparison={metrics.ml_metrics.strategy_comparison} />

                {/* Confusion Matrix (High Risk Threshold) */}
                <ConfusionMatrixGrid matrix={confusionMatrix} />
              </div>
            ) : (
              /* Informative Unavailable Notice Banner */
              <div
                className="p-5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-200 text-xs space-y-1"
                data-testid="ml-metrics-unavailable-banner"
              >
                <div className="flex items-center gap-2 font-semibold text-amber-300">
                  <AlertTriangle className="w-4 h-4" />
                  <span>ML Benchmark Evaluation Unavailable</span>
                </div>
                <p className="text-slate-300 pl-6">
                  {metrics.ml_metrics_reason || 'ML benchmark report not found at ml/reports/metrics.json.'}
                </p>
                <p className="text-slate-400 pl-6 text-[11px]">
                  Offline research evaluation metrics and confusion matrices require training reports to be generated. Operational telemetry remains active.
                </p>
              </div>
            )}
          </section>
        </>
      ) : null}
    </div>
  );
}

export default MetricsView;

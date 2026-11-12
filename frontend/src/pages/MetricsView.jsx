import React from 'react';
import { BarChart2 } from 'lucide-react';

export function MetricsView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between pb-5 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <BarChart2 className="w-5 h-5 text-indigo-400" />
            Model & Pipeline Metrics
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Imbalance strategy benchmark, latency percentiles, and explanation stability.
          </p>
        </div>
      </div>

      <div className="rounded-xl border border-dashed border-slate-800 bg-slate-900/50 p-12 text-center">
        <p className="text-base font-medium text-slate-300">
          Metrics View: Not built yet (F2/F3)
        </p>
        <p className="text-xs text-slate-500 mt-2 max-w-md mx-auto">
          Latency percentiles, routing distribution, SHAP stability metrics, and class imbalance comparison charts will be implemented in Milestone F3.
        </p>
      </div>
    </div>
  );
}

export default MetricsView;

import React from 'react';
import { PlayCircle } from 'lucide-react';

export function TestTransactionView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between pb-5 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <PlayCircle className="w-5 h-5 text-indigo-400" />
            Test Single Transaction
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Simulate PaySim features and evaluate gateway routing in real-time.
          </p>
        </div>
      </div>

      <div className="rounded-xl border border-dashed border-slate-800 bg-slate-900/50 p-12 text-center">
        <p className="text-base font-medium text-slate-300">
          Test Transaction View: Not built yet (F2/F3)
        </p>
        <p className="text-xs text-slate-500 mt-2 max-w-md mx-auto">
          Interactive single transaction evaluation and SHAP waterfall inspection will be implemented in Milestone F2.
        </p>
      </div>
    </div>
  );
}

export default TestTransactionView;

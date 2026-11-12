import React from 'react';
import { AlertTriangle } from 'lucide-react';

export function FlaggedQueue() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between pb-5 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            Flagged Transaction Queue
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Analyst review queue for High Risk and Unstable explanation routings.
          </p>
        </div>
      </div>

      <div className="rounded-xl border border-dashed border-slate-800 bg-slate-900/50 p-12 text-center">
        <p className="text-base font-medium text-slate-300">
          Flagged Queue: Not built yet (F2/F3)
        </p>
        <p className="text-xs text-slate-500 mt-2 max-w-md mx-auto">
          Analyst review workflow, routing verification, and verdict submission will be implemented in Milestone F2.
        </p>
      </div>
    </div>
  );
}

export default FlaggedQueue;

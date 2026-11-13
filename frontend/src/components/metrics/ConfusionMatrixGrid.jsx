import React from 'react';
import { Grid, Info } from 'lucide-react';

export function ConfusionMatrixGrid({ matrix }) {
  if (!matrix) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm" data-testid="confusion-matrix-fallback">
        <div className="flex items-center gap-2 pb-3 border-b border-slate-800">
          <Grid className="w-4 h-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-white">Confusion Matrix (High Risk Threshold)</h2>
        </div>
        <div className="py-8 text-center text-slate-400 text-xs flex items-center justify-center gap-2">
          <Info className="w-4 h-4 text-slate-400" />
          <span>Confusion matrix is not included in the current ML benchmark payload.</span>
        </div>
      </div>
    );
  }

  const tp = Number(matrix.true_positives ?? matrix.tp ?? 0);
  const fp = Number(matrix.false_positives ?? matrix.fp ?? 0);
  const tn = Number(matrix.true_negatives ?? matrix.tn ?? 0);
  const fn = Number(matrix.false_negatives ?? matrix.fn ?? 0);

  const totalActualPositive = tp + fn;
  const totalActualNegative = tn + fp;
  const total = tp + fp + tn + fn;

  const precision = (tp + fp) > 0 ? (tp / (tp + fp)) * 100 : 0;
  const recall = totalActualPositive > 0 ? (tp / totalActualPositive) * 100 : 0;
  const specificity = totalActualNegative > 0 ? (tn / totalActualNegative) * 100 : 0;
  const fpr = totalActualNegative > 0 ? (fp / totalActualNegative) * 100 : 0;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm" data-testid="confusion-matrix-grid">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Grid className="w-4 h-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-white">Confusion Matrix (High Risk Threshold)</h2>
        </div>
        <span className="text-xs text-slate-400">Total Evaluated: {total.toLocaleString()}</span>
      </div>

      <div className="mt-5 overflow-x-auto">
        <div className="min-w-[480px]">
          {/* 2x2 Matrix Structure */}
          <div className="grid grid-cols-[140px_1fr_1fr] gap-2 text-center text-xs">
            {/* Header row */}
            <div className="p-2 font-medium text-slate-400 flex items-center justify-center">
              <span>Actual \ Predicted</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-800/60 font-semibold text-slate-300">
              Predicted Legitimate (Low Risk)
            </div>
            <div className="p-2.5 rounded-lg bg-slate-800/60 font-semibold text-slate-300">
              Predicted Fraud (High Risk)
            </div>

            {/* Row 1: Actual Legitimate */}
            <div className="p-2.5 rounded-lg bg-slate-800/60 font-semibold text-slate-300 flex items-center justify-center text-left">
              Actual Legitimate (Class 0)
            </div>
            {/* True Negative */}
            <div className="p-4 rounded-xl bg-slate-800/40 border border-emerald-500/20 flex flex-col justify-center items-center" data-testid="matrix-tn">
              <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">True Negative (TN)</span>
              <span className="text-2xl font-bold text-white mt-1">{tn.toLocaleString()}</span>
              <span className="text-[11px] text-slate-400 mt-0.5">
                Specificity: {specificity.toFixed(2)}%
              </span>
            </div>
            {/* False Positive */}
            <div className="p-4 rounded-xl bg-slate-800/40 border border-amber-500/20 flex flex-col justify-center items-center" data-testid="matrix-fp">
              <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider">False Positive (FP)</span>
              <span className="text-2xl font-bold text-amber-300 mt-1">{fp.toLocaleString()}</span>
              <span className="text-[11px] text-slate-400 mt-0.5">
                FPR: {fpr.toFixed(3)}%
              </span>
            </div>

            {/* Row 2: Actual Fraud */}
            <div className="p-2.5 rounded-lg bg-slate-800/60 font-semibold text-slate-300 flex items-center justify-center text-left">
              Actual Fraud (Class 1)
            </div>
            {/* False Negative */}
            <div className="p-4 rounded-xl bg-slate-800/40 border border-rose-500/20 flex flex-col justify-center items-center" data-testid="matrix-fn">
              <span className="text-xs font-semibold text-rose-400 uppercase tracking-wider">False Negative (FN)</span>
              <span className="text-2xl font-bold text-rose-300 mt-1">{fn.toLocaleString()}</span>
              <span className="text-[11px] text-slate-400 mt-0.5">
                Missed Fraud Alert
              </span>
            </div>
            {/* True Positive */}
            <div className="p-4 rounded-xl bg-slate-800/40 border border-indigo-500/20 flex flex-col justify-center items-center" data-testid="matrix-tp">
              <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">True Positive (TP)</span>
              <span className="text-2xl font-bold text-indigo-300 mt-1">{tp.toLocaleString()}</span>
              <span className="text-[11px] text-slate-400 mt-0.5">
                Recall: {recall.toFixed(2)}% | Precision: {precision.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ConfusionMatrixGrid;

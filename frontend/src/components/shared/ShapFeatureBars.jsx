import React from 'react';
import { TrendingUp, TrendingDown, HelpCircle } from 'lucide-react';

export function ShapFeatureBars({ features = [], className = '' }) {
  if (!features || !Array.isArray(features) || features.length === 0) {
    return (
      <div className={`space-y-2 ${className}`}>
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Top Feature Attributions (SHAP)
        </h4>
        <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 text-slate-500 text-xs italic text-center">
          No feature explanations available for this transaction.
        </div>
        <p className="text-[10px] text-slate-500 italic">
          SHAP values measure local feature attribution for this model prediction, not causal proof.
        </p>
      </div>
    );
  }

  // Guard against division-by-zero (Condition 3)
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.shap_value || 0)), 0);

  return (
    <div className={`space-y-3 ${className}`} data-testid="shap-feature-bars">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Top Feature Attributions (SHAP)
        </h4>
        <span className="text-[10px] font-mono text-slate-500">
          Relative to max |SHAP| = {maxAbs.toFixed(4)}
        </span>
      </div>

      <div className="space-y-2.5">
        {features.map((f, idx) => {
          const val = typeof f.shap_value === 'number' && !isNaN(f.shap_value) ? f.shap_value : 0;
          const isIncrease = f.direction === 'increases_risk' || val > 0;
          const signedValue = val > 0 ? `+${val.toFixed(4)}` : val.toFixed(4);
          const widthPct = maxAbs > 0 ? Math.min(100, Math.round((Math.abs(val) / maxAbs) * 100)) : 0;

          return (
            <div
              key={`${f.feature}-${idx}`}
              className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 space-y-1.5"
            >
              {/* Header row: feature name, signed value, text badge (Condition 3: never colour alone) */}
              <div className="flex items-center justify-between text-xs gap-2">
                <span className="font-mono font-medium text-slate-200 truncate" title={f.feature}>
                  {f.feature}
                </span>

                <div className="flex items-center gap-2 shrink-0">
                  <span className="font-mono text-xs font-semibold text-slate-300">
                    {signedValue}
                  </span>

                  {isIncrease ? (
                    <span
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-rose-500/10 text-rose-400 border border-rose-500/30"
                      title="Positive SHAP value: feature value pushes model score toward Fraud"
                    >
                      <TrendingUp className="w-3 h-3 shrink-0 text-rose-400" aria-hidden="true" />
                      <span>Pushes toward Fraud</span>
                    </span>
                  ) : (
                    <span
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                      title="Negative SHAP value: feature value pushes model score toward Legitimate"
                    >
                      <TrendingDown className="w-3 h-3 shrink-0 text-emerald-400" aria-hidden="true" />
                      <span>Pushes toward Legitimate</span>
                    </span>
                  )}
                </div>
              </div>

              {/* Bar visualization (decorative, aria-hidden; text labels above are accessible) */}
              <div
                className="h-2 w-full bg-slate-950 rounded-full overflow-hidden flex"
                aria-hidden="true"
                data-testid="shap-bar-track"
              >
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    isIncrease ? 'bg-rose-500' : 'bg-emerald-500'
                  }`}
                  style={{ width: `${widthPct}%` }}
                  data-testid="shap-bar-fill"
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Mandatory Causality Disclaimer */}
      <p className="text-[10px] text-slate-500 leading-normal italic">
        SHAP values measure local feature attribution for this model prediction, not causal proof.
      </p>
    </div>
  );
}

export default ShapFeatureBars;

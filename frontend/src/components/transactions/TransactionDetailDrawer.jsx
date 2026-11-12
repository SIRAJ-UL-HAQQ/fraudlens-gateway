import React, { useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { X, AlertTriangle, Clock, Loader2 } from 'lucide-react';
import { getTransaction } from '../../api/client';
import DecisionBadge from '../shared/DecisionBadge';
import ReliabilityBadge from '../shared/ReliabilityBadge';
import RiskScoreBar from '../shared/RiskScoreBar';
import RuleFlagList from '../shared/RuleFlagList';
import ShapFeatureBars from '../shared/ShapFeatureBars';
import ReviewHistoryTimeline from './ReviewHistoryTimeline';
import AnalystReviewForm from './AnalystReviewForm';

export function TransactionDetailDrawer({
  transactionId,
  isOpen,
  onClose,
  initialData = null,
}) {
  const closeButtonRef = useRef(null);

  // Fetch full transaction details (including review audit log)
  const {
    data: transaction,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['transaction', transactionId],
    queryFn: () => getTransaction(transactionId),
    enabled: Boolean(isOpen && transactionId),
    initialData: initialData && initialData.transaction_id === transactionId ? initialData : undefined,
  });

  // Handle ESC key press to close dialog
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
      // Auto-focus close button for keyboard accessibility
      setTimeout(() => {
        if (closeButtonRef.current) {
          closeButtonRef.current.focus();
        }
      }, 50);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  // Seamlessly adapt to both flat fixture shape and nested backend TransactionDetail shape
  const txData = transaction?.transaction_data || transaction || {};
  const predData = transaction?.prediction || transaction || {};

  const txId = transaction?.transaction_id || transactionId;
  const step = txData.step ?? transaction?.step;
  const type = txData.type ?? transaction?.type;
  const amount = txData.amount ?? transaction?.amount;
  const nameOrig = txData.name_orig ?? transaction?.name_orig;
  const oldbalanceOrg = txData.oldbalance_org ?? transaction?.oldbalance_org;
  const newbalanceOrig = txData.newbalance_orig ?? transaction?.newbalance_orig;
  const nameDest = txData.name_dest ?? transaction?.name_dest;
  const oldbalanceDest = txData.oldbalance_dest ?? transaction?.oldbalance_dest;
  const newbalanceDest = txData.newbalance_dest ?? transaction?.newbalance_dest;
  const isFlaggedFraud = txData.is_flagged_fraud ?? transaction?.is_flagged_fraud;

  const riskScore = predData.risk_score ?? transaction?.risk_score;
  const gatewayDecision = predData.gateway_decision ?? transaction?.gateway_decision;
  const explanationStatus = predData.explanation_status ?? transaction?.explanation_status;
  const ruleFlags = predData.rule_flags ?? transaction?.rule_flags ?? [];
  const shapFeatures = predData.top_features ?? transaction?.shap_features ?? [];
  const latencyMs = predData.latency_ms ?? transaction?.latency_ms;
  const reviews = transaction?.reviews ?? [];
  const createdAt = transaction?.created_at;

  const formatCurrency = (val) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 2,
    }).format(val);
  };

  return (
    <div
      className="fixed inset-0 z-50 overflow-hidden"
      role="dialog"
      aria-modal="true"
      aria-labelledby="drawer-title"
      data-testid="transaction-detail-drawer"
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm transition-opacity"
        onClick={onClose}
        data-testid="drawer-backdrop"
        aria-hidden="true"
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-2xl bg-slate-950 border-l border-slate-800 shadow-2xl flex flex-col">
          {/* Header */}
          <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60 shrink-0">
            <div>
              <span className="text-[11px] font-mono uppercase text-slate-500 tracking-wider">
                Transaction Details & Audit Record
              </span>
              <h2 id="drawer-title" className="text-base font-bold text-slate-100 font-mono flex items-center gap-2">
                {txId}
              </h2>
            </div>
            <button
              ref={closeButtonRef}
              type="button"
              onClick={onClose}
              aria-label="Close transaction details"
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors cursor-pointer"
              data-testid="drawer-close-btn"
            >
              <X className="w-5 h-5" aria-hidden="true" />
            </button>
          </div>

          {/* Body Content */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
            {isLoading && !transaction && (
              <div className="flex flex-col items-center justify-center py-20 text-slate-400 gap-3">
                <Loader2 className="w-8 h-8 animate-spin text-emerald-400" aria-hidden="true" />
                <span className="text-xs">Loading transaction record...</span>
              </div>
            )}

            {error && !transaction && (
              <div
                role="alert"
                className="p-4 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs space-y-2"
              >
                <div className="font-semibold flex items-center gap-1.5">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>Failed to load transaction details</span>
                </div>
                <p>{error.message || 'Transaction record could not be retrieved.'}</p>
              </div>
            )}

            {/* Model Prediction & Gateway Routing Banner */}
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-lg bg-slate-900/50 border border-slate-800">
                <div className="space-y-1">
                  <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
                    Gateway Routing Decision
                  </span>
                  <div className="flex items-center gap-2">
                    <DecisionBadge decision={gatewayDecision} />
                    <ReliabilityBadge status={explanationStatus} />
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
                    Inference Latency
                  </span>
                  <span className="text-xs font-mono text-slate-300 flex items-center justify-end gap-1">
                    <Clock className="w-3 h-3 text-slate-500" aria-hidden="true" />
                    {latencyMs != null ? `${Number(latencyMs).toFixed(1)} ms` : 'N/A'}
                  </span>
                </div>
              </div>

              {/* Prominent Warning Banner if Unstable (Condition 3 / Variant B) */}
              {explanationStatus === 'Unstable' && (
                <div
                  role="alert"
                  className="p-3.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs space-y-1.5"
                  data-testid="unstable-warning-banner"
                >
                  <div className="flex items-center gap-2 font-semibold text-amber-200">
                    <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" aria-hidden="true" />
                    <span>Unstable Explanation Detected (SRS Variant B Routing)</span>
                  </div>
                  <p className="text-[11px] text-amber-300/90 leading-relaxed">
                    Local SHAP feature attribution ranking fluctuates across resampling strategies
                    (no resampling vs class weighting vs SMOTE/SMOTE-ENN). High-risk predictions with
                    unstable explanations are routed to Analyst Review for human adjudication rather than
                    automated blocking.
                  </p>
                </div>
              )}

              {/* Risk Score */}
              <div className="p-4 rounded-lg bg-slate-900/30 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300">Calibrated Risk Score</span>
                  <span className="text-xs font-mono text-slate-400">
                    {riskScore != null ? Number(riskScore).toFixed(4) : '—'}
                  </span>
                </div>
                <RiskScoreBar score={riskScore || 0} />
              </div>

              {/* Rule Flags (Rendered unconditionally for all transactions, even Low Risk - Variant B) */}
              <div className="p-4 rounded-lg bg-slate-900/30 border border-slate-800">
                <RuleFlagList flags={ruleFlags} />
              </div>
            </div>

            {/* PaySim Raw Transaction Parameters */}
            <div className="space-y-3">
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                PaySim Financial Parameters
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 p-4 rounded-lg bg-slate-900/40 border border-slate-800 text-xs">
                <div>
                  <span className="text-slate-500 text-[11px] block">Step (Hour)</span>
                  <span className="font-mono text-slate-200">{step ?? '—'}</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[11px] block">Transaction Type</span>
                  <span className="font-mono font-medium text-emerald-400">{type ?? '—'}</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[11px] block">Transfer Amount</span>
                  <span className="font-mono font-bold text-slate-100">{formatCurrency(amount)}</span>
                </div>

                <div className="sm:col-span-3 border-t border-slate-800/80 my-1" />

                <div>
                  <span className="text-slate-500 text-[11px] block">Origin Account</span>
                  <span className="font-mono text-slate-300 break-all">{nameOrig ?? '—'}</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[11px] block">Old Balance (Orig)</span>
                  <span className="font-mono text-slate-400">{formatCurrency(oldbalanceOrg)}</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[11px] block">New Balance (Orig)</span>
                  <span className="font-mono text-slate-400">{formatCurrency(newbalanceOrig)}</span>
                </div>

                <div className="sm:col-span-3 border-t border-slate-800/80 my-1" />

                <div>
                  <span className="text-slate-500 text-[11px] block">Destination Account</span>
                  <span className="font-mono text-slate-300 break-all">{nameDest ?? '—'}</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[11px] block">Old Balance (Dest)</span>
                  <span className="font-mono text-slate-400">{formatCurrency(oldbalanceDest)}</span>
                </div>
                <div>
                  <span className="text-slate-500 text-[11px] block">New Balance (Dest)</span>
                  <span className="font-mono text-slate-400">{formatCurrency(newbalanceDest)}</span>
                </div>

                <div className="sm:col-span-3 border-t border-slate-800/80 my-1" />

                <div>
                  <span className="text-slate-500 text-[11px] block">Heuristic Flagged</span>
                  <span className="font-mono text-slate-300">
                    {isFlaggedFraud === 1 ? 'Yes (Flagged)' : 'No (0)'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 text-[11px] block">Logged At</span>
                  <span className="font-mono text-slate-400 text-[11px]">
                    {createdAt ? new Date(createdAt).toLocaleString() : '—'}
                  </span>
                </div>
              </div>
            </div>

            {/* SHAP Explanations Section */}
            <div className="p-4 rounded-lg bg-slate-900/40 border border-slate-800 space-y-3">
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Feature Attribution (SHAP)
              </h3>
              <ShapFeatureBars features={shapFeatures} />
            </div>

            {/* Analyst Review & Audit History Section */}
            <div className="space-y-4">
              <ReviewHistoryTimeline reviews={reviews} />
              <AnalystReviewForm transactionId={txId} onSuccess={() => refetch()} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TransactionDetailDrawer;

import React from 'react';
import { ArrowRight, FileText, Clock } from 'lucide-react';
import DecisionBadge from '../shared/DecisionBadge';
import ReliabilityBadge from '../shared/ReliabilityBadge';
import RiskScoreBar from '../shared/RiskScoreBar';
import Skeleton from '../shared/Skeleton';

export function TransactionTable({
  transactions = [],
  isLoading = false,
  onRowClick = null,
  className = '',
}) {
  const formatCurrency = (val) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 2,
    }).format(val);
  };

  const formatDate = (isoStr) => {
    if (!isoStr) return '—';
    try {
      const d = new Date(isoStr);
      return d.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return isoStr;
    }
  };

  return (
    <div
      className={`overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60 shadow-md ${className}`}
      data-testid="transaction-table"
    >
      <table className="w-full text-left text-xs border-collapse">
        <thead className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold text-[10px]">
          <tr>
            <th scope="col" className="py-2.5 px-3">Transaction / Route</th>
            <th scope="col" className="py-2.5 px-3">Step & Ingested</th>
            <th scope="col" className="py-2.5 px-3">Type & Amount</th>
            <th scope="col" className="py-2.5 px-3">Decision</th>
            <th scope="col" className="py-2.5 px-3 min-w-[120px]">Risk Score</th>
            <th scope="col" className="py-2.5 px-3">Reliability</th>
            <th scope="col" className="py-2.5 px-3 max-w-[160px]">Rule Flags</th>
            <th scope="col" className="py-2.5 px-3">Audit Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60 text-slate-200">
          {isLoading && transactions.length === 0 ? (
            Array.from({ length: 5 }).map((_, rIdx) => (
              <tr key={`loading-row-${rIdx}`} data-testid="table-loading-row">
                <td className="py-3 px-3"><Skeleton className="h-4 w-28" /></td>
                <td className="py-3 px-3"><Skeleton className="h-4 w-20" /></td>
                <td className="py-3 px-3"><Skeleton className="h-4 w-24" /></td>
                <td className="py-3 px-3"><Skeleton className="h-4 w-20" /></td>
                <td className="py-3 px-3"><Skeleton className="h-4 w-28" /></td>
                <td className="py-3 px-3"><Skeleton className="h-4 w-20" /></td>
                <td className="py-3 px-3"><Skeleton className="h-4 w-24" /></td>
                <td className="py-3 px-3"><Skeleton className="h-4 w-16" /></td>
              </tr>
            ))
          ) : transactions.length === 0 ? (
            <tr>
              <td
                colSpan={8}
                className="py-12 px-3 text-center text-slate-500 italic"
                data-testid="table-empty-message"
              >
                No transactions found matching the filter criteria.
              </td>
            </tr>
          ) : (
            transactions.map((tx) => {
              const ruleFlags = Array.isArray(tx.rule_flags) ? tx.rule_flags : [];
              const isReviewed = Boolean(
                tx.reviewed || (Array.isArray(tx.reviews) && tx.reviews.length > 0)
              );

              return (
                <tr
                  key={tx.transaction_id}
                  onClick={() => onRowClick && onRowClick(tx.transaction_id)}
                  className={`transition-colors ${
                    onRowClick
                      ? 'hover:bg-slate-800/60 cursor-pointer focus:bg-slate-800/60'
                      : 'hover:bg-slate-800/30'
                  }`}
                  data-testid={`transaction-row-${tx.transaction_id}`}
                  tabIndex={onRowClick ? 0 : undefined}
                  onKeyDown={(e) => {
                    if (onRowClick && (e.key === 'Enter' || e.key === ' ')) {
                      e.preventDefault();
                      onRowClick(tx.transaction_id);
                    }
                  }}
                >
                  {/* Transaction ID & Accounts */}
                  <td className="py-2.5 px-3">
                    <div className="font-mono font-medium text-slate-200 text-xs">
                      {tx.transaction_id}
                    </div>
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-400 font-mono mt-0.5">
                      <span>{tx.name_orig || '—'}</span>
                      <ArrowRight className="w-3 h-3 text-slate-500 shrink-0" aria-hidden="true" />
                      <span>{tx.name_dest || '—'}</span>
                    </div>
                  </td>

                  {/* Step & Ingested Timestamp (Combined - Bug 5 Fix) */}
                  <td className="py-2.5 px-3">
                    <div className="font-mono text-slate-200 text-xs font-semibold">
                      Step {tx.step ?? '—'}
                    </div>
                    <div className="font-mono text-[10px] text-slate-400 mt-0.5 whitespace-nowrap">
                      {formatDate(tx.created_at)}
                    </div>
                  </td>

                  {/* Type & Amount */}
                  <td className="py-2.5 px-3">
                    <span className="font-mono text-[11px] font-semibold text-emerald-400 block">
                      {tx.type || '—'}
                    </span>
                    <span className="font-mono text-xs font-bold text-slate-100">
                      {formatCurrency(tx.amount)}
                    </span>
                  </td>

                  {/* Decision */}
                  <td className="py-2.5 px-3">
                    <DecisionBadge decision={tx.gateway_decision} />
                  </td>

                  {/* Risk Score */}
                  <td className="py-2.5 px-3">
                    <div className="space-y-1">
                      <div className="flex justify-between text-[11px] font-mono text-slate-400">
                        <span>Score:</span>
                        <span>{tx.risk_score != null ? Number(tx.risk_score).toFixed(4) : '—'}</span>
                      </div>
                      <RiskScoreBar score={tx.risk_score || 0} />
                    </div>
                  </td>

                  {/* Reliability */}
                  <td className="py-2.5 px-3">
                    <ReliabilityBadge status={tx.explanation_status} />
                  </td>

                  {/* Rule Flags (Unconditionally rendered on all transactions - Variant B requirement) */}
                  <td className="py-2.5 px-3">
                    {ruleFlags.length === 0 ? (
                      <span className="text-[11px] text-slate-500 font-mono italic">None</span>
                    ) : (
                      <div className="flex flex-wrap gap-1 max-w-[150px]">
                        {ruleFlags.map((flag) => (
                          <span
                            key={flag}
                            className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700 truncate"
                          >
                            {flag}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>

                  {/* Audit Status (Neutral FileText icon - Bug 8 Fix, no check marks) */}
                  <td className="py-2.5 px-3">
                    {isReviewed ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                        <FileText className="w-3 h-3 shrink-0 text-emerald-400" aria-hidden="true" />
                        <span>Reviewed</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/30">
                        <Clock className="w-3 h-3 shrink-0 text-amber-400" aria-hidden="true" />
                        <span>Pending</span>
                      </span>
                    )}
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}

export default TransactionTable;

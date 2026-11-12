import React from 'react';
import { UserCheck, AlertOctagon, Shield } from 'lucide-react';

export function ReviewHistoryTimeline({ reviews = [], className = '' }) {
  const reviewList = Array.isArray(reviews) ? reviews : [];

  if (reviewList.length === 0) {
    return (
      <div
        className={`p-4 rounded-lg border border-slate-800 bg-slate-900/50 text-center text-xs text-slate-400 ${className}`}
        data-testid="empty-review-history"
      >
        No analyst reviews recorded yet.
      </div>
    );
  }

  // Display newest first
  const sortedReviews = [...reviewList].sort((a, b) => {
    const timeA = new Date(a.reviewed_at || a.created_at || 0).getTime();
    const timeB = new Date(b.reviewed_at || b.created_at || 0).getTime();
    return timeB - timeA;
  });

  return (
    <div className={`space-y-3 ${className}`} data-testid="review-history-timeline">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Review Audit Trail ({reviewList.length})
        </h4>
        <span className="text-[11px] text-slate-500 font-mono">Append-only</span>
      </div>

      <ol className="space-y-3 border-l-2 border-slate-800 pl-4 ml-1">
        {sortedReviews.map((rev, idx) => {
          const outcome = rev.outcome || rev.adjudication;
          const isFraud = outcome === 'Confirmed Fraud';
          const badgeStyle = isFraud
            ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
            : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
          const Icon = isFraud ? AlertOctagon : Shield;

          const rawDate = rev.reviewed_at || rev.created_at;
          let formattedDate = rawDate;
          try {
            formattedDate = new Date(rawDate).toLocaleString();
          } catch {
            formattedDate = rawDate;
          }

          return (
            <li
              key={rev.review_id || `${idx}-${rawDate}`}
              className="relative text-xs space-y-1.5"
              data-testid="review-history-item"
            >
              {/* Dot indicator */}
              <div
                className={`absolute -left-[21px] top-1 w-2.5 h-2.5 rounded-full border-2 border-slate-950 ${
                  isFraud ? 'bg-rose-400' : 'bg-emerald-400'
                }`}
                aria-hidden="true"
              />

              <div className="flex flex-wrap items-center justify-between gap-2">
                <span
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border ${badgeStyle}`}
                >
                  <Icon className="w-3 h-3 shrink-0" aria-hidden="true" />
                  <span>{outcome}</span>
                </span>
                <span className="text-[11px] text-slate-400 font-mono">{formattedDate}</span>
              </div>

              <div className="flex items-center gap-1.5 text-slate-300">
                <UserCheck className="w-3.5 h-3.5 text-slate-400 shrink-0" aria-hidden="true" />
                <span className="font-medium text-slate-200">
                  {rev.reviewed_by || 'Unknown Analyst'}
                </span>
              </div>

              {rev.note ? (
                <div className="p-2.5 rounded bg-slate-900 border border-slate-800/80 text-slate-300 font-mono text-[11px] whitespace-pre-wrap">
                  {rev.note}
                </div>
              ) : (
                <div className="text-[11px] text-slate-500 italic">(No note provided)</div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export default ReviewHistoryTimeline;

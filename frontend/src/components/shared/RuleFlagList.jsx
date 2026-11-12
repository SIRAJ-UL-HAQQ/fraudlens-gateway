import React from 'react';

export function RuleFlagList({ flags = [], className = '' }) {
  const flagList = Array.isArray(flags) ? flags : [];

  return (
    <div className={`space-y-1.5 ${className}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-300">
          Observed Rule Flags
        </span>
      </div>

      {flagList.length === 0 ? (
        <span className="text-xs font-mono text-slate-500 italic">None</span>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {flagList.map((flag) => (
            <span
              key={flag}
              className="px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-slate-800 text-slate-300 border border-slate-700/80"
            >
              {flag}
            </span>
          ))}
        </div>
      )}

      <p className="text-[10px] text-slate-500 italic">
        Observed rule flags do not imply causality in gateway routing.
      </p>
    </div>
  );
}

export default RuleFlagList;

import React from 'react';

export function RiskScoreBar({ score = 0, className = '' }) {
  const numScore = typeof score === 'number' && !isNaN(score) ? score : 0;
  const clampedScore = Math.max(0, Math.min(1, numScore));
  const percent = Math.round(clampedScore * 100);

  let fillColor = 'bg-emerald-500';
  if (clampedScore >= 0.7) {
    fillColor = 'bg-rose-500';
  } else if (clampedScore >= 0.3) {
    fillColor = 'bg-amber-500';
  }

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div
        role="meter"
        aria-valuenow={clampedScore}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuetext={`Risk score ${clampedScore.toFixed(2)} on a 0 to 1 scale`}
        className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden relative"
      >
        <div
          className={`h-full ${fillColor} rounded-full transition-all duration-300`}
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="text-xs font-mono font-medium text-slate-300 w-9 text-right shrink-0">
        {clampedScore.toFixed(2)}
      </span>
    </div>
  );
}

export default RiskScoreBar;

import React from 'react';
import { Shield, AlertTriangle, Clock, HelpCircle } from 'lucide-react';

export function DecisionBadge({ decision, className = '' }) {
  let badgeStyle = 'bg-slate-800/80 text-slate-300 border-slate-700';
  let Icon = HelpCircle;
  const label = decision || 'Unknown';

  if (label === 'Low Risk') {
    // Neutral icon (Shield), NO check mark anywhere near gateway badges
    badgeStyle = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    Icon = Shield;
  } else if (label === 'High Risk') {
    badgeStyle = 'bg-rose-500/10 text-rose-400 border-rose-500/30';
    Icon = AlertTriangle;
  } else if (label === 'Analyst Review') {
    badgeStyle = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    Icon = Clock;
  }

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${badgeStyle} ${className}`}
      data-testid="decision-badge"
    >
      <Icon className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}

export default DecisionBadge;

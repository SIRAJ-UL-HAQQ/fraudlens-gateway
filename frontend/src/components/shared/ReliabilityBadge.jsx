import React, { useState, useEffect, useRef, useId } from 'react';
import { Shield, AlertOctagon, HelpCircle, Info } from 'lucide-react';

export function ReliabilityBadge({ status, className = '' }) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);
  const tooltipId = useId();

  const label = status || 'Unknown';
  let badgeStyle = 'bg-slate-800/80 text-slate-300 border-slate-700';
  let Icon = HelpCircle;

  if (label === 'Stable') {
    // Neutral icon (Shield), NO check mark
    badgeStyle = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    Icon = Shield;
  } else if (label === 'Unstable') {
    badgeStyle = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    Icon = AlertOctagon;
  }

  // Handle escape key and click outside
  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };

    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [open]);

  return (
    <div
      ref={containerRef}
      className={`relative inline-flex items-center gap-1.5 ${className}`}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${badgeStyle}`}
        data-testid="reliability-badge"
      >
        <Icon className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
        <span>{label}</span>
      </span>

      {/* Accessible Tooltip Trigger Button */}
      <button
        type="button"
        aria-label="Explanation reliability details"
        aria-describedby={open ? tooltipId : undefined}
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        onFocus={() => setOpen(true)}
        className="p-0.5 rounded-full text-slate-400 hover:text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-400 transition-colors"
      >
        <Info className="w-3.5 h-3.5" aria-hidden="true" />
      </button>

      {/* Tooltip Content */}
      {open && (
        <div
          id={tooltipId}
          role="tooltip"
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-3 rounded-lg bg-slate-900 border border-slate-700 shadow-2xl text-[11px] leading-relaxed text-slate-300 z-50 pointer-events-auto"
        >
          <p className="font-semibold text-slate-200 mb-1">
            SHAP Reliability Disclaimer
          </p>
          <p>
            SHAP is an attribution, not proof of cause. Stable is not proof of correctness.
            Unstable means the explanation varies across replicate models and is not authoritative.
          </p>
          <div className="text-[10px] text-slate-500 mt-2">
            Press Esc or click outside to close
          </div>
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px border-4 border-transparent border-t-slate-700" />
        </div>
      )}
    </div>
  );
}

export default ReliabilityBadge;

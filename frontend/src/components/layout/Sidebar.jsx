import React from 'react';
import { NavLink } from 'react-router-dom';
import { Activity, AlertTriangle, BarChart2, PlayCircle, Download } from 'lucide-react';

const navItems = [
  {
    to: '/',
    label: 'Transaction Feed',
    icon: Activity,
    end: true,
  },
  {
    to: '/flagged',
    label: 'Flagged Queue',
    icon: AlertTriangle,
    end: false,
  },
  {
    to: '/metrics',
    label: 'Pipeline Metrics',
    icon: BarChart2,
    end: false,
  },
  {
    to: '/test',
    label: 'Test Transaction',
    icon: PlayCircle,
    end: false,
  },
  {
    to: '/export',
    label: 'Export Audit Logs',
    icon: Download,
    end: false,
  },
];

export function Sidebar() {
  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 min-h-[calc(100vh-61px)]">
      <nav className="p-4 space-y-1.5 flex-1">
        <div className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Gateway Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-indigo-600/15 text-indigo-400 border border-indigo-500/20 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Footer System Info */}
      <div className="p-4 border-t border-slate-800 text-[11px] text-slate-500 space-y-1">
        <div className="font-mono text-slate-400">PaySim Imbalance Study</div>
        <div>Split: Chronological (step)</div>
        <div>Target: High Risk Triage</div>
      </div>
    </aside>
  );
}

export default Sidebar;

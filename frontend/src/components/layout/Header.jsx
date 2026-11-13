import React, { useState, useEffect } from 'react';
import { Shield, LogOut, Radio, Activity } from 'lucide-react';
import { useAuth } from '../../auth/useAuth';
import { useHealthStatus } from '../../hooks/useHealthStatus';

export function Header() {
  const { user, logout } = useAuth();
  const { isStub, isUnloaded, isDbDisconnected, isUnreachable } = useHealthStatus();
  const [mockActive, setMockActive] = useState(false);

  useEffect(() => {
    const checkMock = () => {
      try {
        const raw = localStorage.getItem('fraudlens_mock_mode');
        const cfg = raw ? JSON.parse(raw) : null;
        setMockActive(Boolean(cfg && cfg.enabled));
      } catch {
        setMockActive(false);
      }
    };
    checkMock();
    window.addEventListener('storage', checkMock);
    return () => window.removeEventListener('storage', checkMock);
  }, []);

  let healthColor = 'bg-emerald-500';
  let healthLabel = 'Gateway Healthy';
  if (isUnreachable) {
    healthColor = 'bg-rose-500';
    healthLabel = 'Offline';
  } else if (isUnloaded) {
    healthColor = 'bg-rose-500';
    healthLabel = 'Model Unloaded';
  } else if (isDbDisconnected) {
    healthColor = 'bg-rose-500';
    healthLabel = 'DB Degraded';
  } else if (isStub) {
    healthColor = 'bg-amber-500';
    healthLabel = 'Stub Mode';
  }

  return (
    <header className="bg-slate-900 border-b border-slate-800 text-slate-100 px-6 py-3.5 flex items-center justify-between sticky top-0 z-30 backdrop-blur bg-slate-900/90">
      {/* Brand & Subtitle */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-indigo-600/20 border border-indigo-500/30 text-indigo-400">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white">
                FraudLens Gateway
              </span>
              {mockActive && (
                <span
                  title="Mock mode active: fixture data and simulated endpoints enabled"
                  className="px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider bg-purple-500/20 text-purple-300 border border-purple-500/30 flex items-center gap-1"
                >
                  <Radio className="w-3 h-3 text-purple-400 animate-pulse" />
                  Mock Mode
                </span>
              )}
            </div>
            <p className="text-[11px] text-slate-400 font-normal">
              Explanation-Reliable Financial Risk Triage
            </p>
          </div>
        </div>
      </div>

      {/* Health status & User controls */}
      <div className="flex items-center gap-6">
        {/* Gateway Health Indicator */}
        <div
          className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-xs text-slate-300"
          title={`Status: ${healthLabel}`}
        >
          <span className="relative flex h-2 w-2">
            <span
              className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${healthColor}`}
            />
            <span className={`relative inline-flex rounded-full h-2 w-2 ${healthColor}`} />
          </span>
          <span className="font-medium text-[11px]">{healthLabel}</span>
        </div>

        {/* Analyst Identity */}
        <div className="flex items-center gap-3 pl-2 border-l border-slate-800">
          <div className="text-right">
            <div className="text-xs font-semibold text-slate-200">
              {user?.username || 'analyst'}
            </div>
            <div className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              {user?.role || 'analyst'}
            </div>
          </div>

          <button
            onClick={logout}
            title="Sign out of Gateway"
            aria-label="Sign out"
            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
}

export default Header;

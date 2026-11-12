import React from 'react';
import { AlertTriangle, AlertOctagon, Database, WifiOff } from 'lucide-react';
import { useHealthStatus } from '../../hooks/useHealthStatus';

export function StatusBanner() {
  const { isStub, isUnloaded, isDbDisconnected, isUnreachable } = useHealthStatus();

  // If gateway is fully unreachable
  if (isUnreachable) {
    return (
      <div
        role="alert"
        aria-live="assertive"
        className="bg-rose-950/90 border-b border-rose-800/80 text-rose-200 px-4 py-2.5 text-xs font-medium flex items-center justify-between shadow-inner"
      >
        <div className="flex items-center gap-2 max-w-7xl mx-auto w-full">
          <WifiOff className="w-4 h-4 text-rose-400 shrink-0" />
          <span>
            <strong>Gateway unreachable.</strong> Check backend connection at http://localhost:8000.
          </span>
        </div>
      </div>
    );
  }

  // If model is unloaded (503 or model.mode === "unloaded")
  if (isUnloaded) {
    return (
      <div
        role="alert"
        aria-live="assertive"
        className="bg-rose-950/90 border-b border-rose-800/80 text-rose-200 px-4 py-2.5 text-xs font-medium flex items-center justify-between shadow-inner"
      >
        <div className="flex items-center gap-2 max-w-7xl mx-auto w-full">
          <AlertOctagon className="w-4 h-4 text-rose-400 shrink-0" />
          <span>
            <strong>Model unloaded / Gateway not ready.</strong> High Risk routing and explanation generation unavailable.
          </span>
        </div>
      </div>
    );
  }

  // If database is disconnected
  if (isDbDisconnected) {
    return (
      <div
        role="alert"
        aria-live="assertive"
        className="bg-rose-950/90 border-b border-rose-800/80 text-rose-200 px-4 py-2.5 text-xs font-medium flex items-center justify-between shadow-inner"
      >
        <div className="flex items-center gap-2 max-w-7xl mx-auto w-full">
          <Database className="w-4 h-4 text-rose-400 shrink-0" />
          <span>
            <strong>Database disconnected / Gateway degraded.</strong> Transaction persistence and review submission disabled.
          </span>
        </div>
      </div>
    );
  }

  // If stub mode
  if (isStub) {
    return (
      <div
        role="status"
        aria-live="polite"
        className="bg-amber-950/90 border-b border-amber-800/80 text-amber-200 px-4 py-2 text-xs font-medium flex items-center justify-between shadow-inner"
      >
        <div className="flex items-center gap-2 max-w-7xl mx-auto w-full">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
          <span>
            <strong>Gateway running in STUB MODE.</strong> Model artifacts not loaded; predictions are non-authoritative heuristic mocks.
          </span>
        </div>
      </div>
    );
  }

  return null;
}

export default StatusBanner;

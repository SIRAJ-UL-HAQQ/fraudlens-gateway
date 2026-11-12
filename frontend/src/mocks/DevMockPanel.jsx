import React, { useState, useEffect } from 'react';
import { getMockConfig, setMockConfig } from './mockService';

function DevMockPanelInner() {
  const [config, setConfig] = useState(getMockConfig());
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    setMockConfig(config);
  }, [config]);

  const toggleEnabled = () => {
    const updated = { ...config, enabled: !config.enabled };
    setConfig(updated);
    setMockConfig(updated);
  };

  const setState = (state) => {
    const updated = { ...config, state };
    setConfig(updated);
    setMockConfig(updated);
  };

  return (
    <div className="fixed bottom-4 left-4 z-50 text-xs font-mono">
      {!isOpen ? (
        <button
          type="button"
          onClick={() => setIsOpen(true)}
          className={`px-3 py-1.5 rounded-full border shadow-lg flex items-center gap-1.5 ${
            config.enabled
              ? 'bg-amber-950 text-amber-200 border-amber-600'
              : 'bg-slate-800 text-slate-300 border-slate-700'
          }`}
        >
          <span className={`w-2 h-2 rounded-full ${config.enabled ? 'bg-amber-400 animate-pulse' : 'bg-slate-500'}`} />
          <span>{config.enabled ? `Mock: ${config.state}` : 'Mock: OFF'}</span>
        </button>
      ) : (
        <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-2xl text-slate-200 w-64">
          <div className="flex justify-between items-center mb-2 pb-1 border-b border-slate-800">
            <span className="font-semibold text-slate-100">Dev Mock Controls</span>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="text-slate-400 hover:text-slate-100"
            >
              ✕
            </button>
          </div>

          <div className="mb-3 flex items-center justify-between">
            <span>Mock Mode</span>
            <button
              type="button"
              onClick={toggleEnabled}
              className={`px-2 py-0.5 rounded text-xs font-bold ${
                config.enabled ? 'bg-amber-600 text-white' : 'bg-slate-700 text-slate-400'
              }`}
            >
              {config.enabled ? 'ENABLED' : 'DISABLED'}
            </button>
          </div>

          <div className="space-y-1">
            <div className="text-[10px] uppercase text-slate-500">Gateway State</div>
            {['normal', 'stub', 'unloaded', 'disconnected', '503', 'network_error'].map((st) => (
              <button
                key={st}
                type="button"
                onClick={() => setState(st)}
                disabled={!config.enabled}
                className={`w-full text-left px-2 py-1 rounded text-xs transition ${
                  config.state === st && config.enabled
                    ? 'bg-blue-600 text-white font-bold'
                    : 'hover:bg-slate-800 text-slate-400'
                }`}
              >
                {st}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function DevMockPanel() {
  if (!import.meta.env.DEV) {
    return null;
  }
  return <DevMockPanelInner />;
}

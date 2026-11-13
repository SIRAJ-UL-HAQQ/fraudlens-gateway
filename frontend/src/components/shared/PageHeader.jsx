import React from 'react';

export function PageHeader({ title, subtitle = '', actions = null, className = '' }) {
  return (
    <div
      className={`flex flex-col sm:flex-row sm:items-center sm:justify-between pb-5 border-b border-slate-800 gap-4 ${className}`}
    >
      <div>
        <h1 className="text-xl font-bold tracking-tight text-white">{title}</h1>
        {subtitle && <p className="text-xs text-slate-400 mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-3">{actions}</div>}
    </div>
  );
}

export default PageHeader;

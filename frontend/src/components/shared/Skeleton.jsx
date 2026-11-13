import React from 'react';

export function Skeleton({ className = '' }) {
  return (
    <div
      className={`animate-pulse bg-slate-800/80 rounded ${className}`}
      aria-hidden="true"
    />
  );
}

export default Skeleton;

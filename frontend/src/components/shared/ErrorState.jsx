import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

export function ErrorState({
  error,
  onRetry = null,
  title = 'Failed to load data',
  className = '',
}) {
  const errorMessage =
    typeof error === 'string'
      ? error
      : error?.message || 'An unexpected error occurred while processing your request.';

  return (
    <div
      role="alert"
      className={`p-6 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-200 flex flex-col items-center text-center ${className}`}
    >
      <div className="w-10 h-10 rounded-full bg-rose-500/20 flex items-center justify-center text-rose-400 mb-3">
        <AlertCircle className="w-5 h-5" />
      </div>
      <h3 className="text-sm font-semibold text-rose-200">{title}</h3>
      <p className="text-xs text-rose-300/80 max-w-md mt-1 mb-4">{errorMessage}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-medium transition-colors shadow-sm"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Try Again</span>
        </button>
      )}
    </div>
  );
}

export default ErrorState;

import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Keep error logging accessible for debugging
    if (process.env.NODE_ENV !== 'test') {
      console.error('ErrorBoundary caught an unhandled runtime error:', error, errorInfo);
    }
  }

  handleReload = () => {
    this.setState({ hasError: false, error: null });
    if (typeof this.props.onReset === 'function') {
      this.props.onReset();
    } else if (typeof window !== 'undefined' && window.location?.reload) {
      window.location.reload();
    }
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div
          role="alert"
          data-testid="error-boundary-fallback"
          className="min-h-[300px] flex items-center justify-center p-6"
        >
          <div className="max-w-md w-full bg-slate-900 border border-rose-500/30 rounded-xl p-6 text-center shadow-lg space-y-4">
            <div className="w-12 h-12 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center mx-auto">
              <AlertTriangle className="w-6 h-6" />
            </div>

            <div>
              <h3 className="text-base font-semibold text-white">Application Error Encountered</h3>
              <p className="text-xs text-slate-400 mt-1">
                An unexpected runtime error interrupted the view. You can reload the view to recover application state.
              </p>
            </div>

            {this.state.error?.message && (
              <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-rose-300 font-mono text-[11px] text-left break-all max-h-32 overflow-y-auto">
                {this.state.error.message}
              </div>
            )}

            <button
              type="button"
              data-testid="reload-view-btn"
              onClick={this.handleReload}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-colors shadow-sm"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reload View</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

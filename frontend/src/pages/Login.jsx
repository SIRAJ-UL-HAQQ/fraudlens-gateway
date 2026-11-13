import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Shield, AlertCircle, Clock, LogIn } from 'lucide-react';
import { useAuth } from '../auth/useAuth';

export function Login() {
  const { login, isAuthenticated, sessionExpired, resetSessionExpired } = useAuth();
  const routerNavigate = useNavigate();
  const location = useLocation();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const from = location.state?.from?.pathname || '/';
  const showSessionExpiredNotice = sessionExpired || Boolean(location.state?.sessionExpired);

  useEffect(() => {
    if (isAuthenticated) {
      routerNavigate(from, { replace: true });
    }
  }, [isAuthenticated, routerNavigate, from]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError('Please enter both username and password.');
      return;
    }

    setError(null);
    setLoading(true);
    resetSessionExpired();

    try {
      await login({ username: username.trim(), password });
      routerNavigate(from, { replace: true });
    } catch (err) {
      setError(err.message || 'Failed to sign in. Please verify your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickFill = (e) => {
    e.preventDefault();
    setUsername('analyst_1');
    setPassword('password123');
    setError(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-center items-center px-4 py-12">
      <div className="w-full max-w-md">
        {/* Brand Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 mb-4 shadow-lg shadow-indigo-950/50">
            <Shield className="w-8 h-8" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">FraudLens Gateway</h1>
          <p className="text-sm text-slate-400 mt-1">
            Explanation-Reliable Financial Risk Triage
          </p>
        </div>

        {/* Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl backdrop-blur">
          <h2 className="text-lg font-semibold text-slate-200 mb-6">Analyst Sign In</h2>

          {/* Session Expired Notice */}
          {showSessionExpiredNotice && (
            <div
              role="alert"
              className="mb-5 p-4 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-sm flex items-start gap-3"
            >
              <Clock className="w-5 h-5 shrink-0 mt-0.5 text-amber-400" />
              <div>
                <p className="font-medium">Session Expired</p>
                <p className="text-xs text-amber-300/80 mt-0.5">
                  Session expired. Please log in again.
                </p>
              </div>
            </div>
          )}

          {/* Error Notice */}
          {error && (
            <div
              role="alert"
              className="mb-5 p-4 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-start gap-3"
            >
              <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
              <div>
                <p className="font-medium">Authentication Error</p>
                <p className="text-xs text-rose-300/80 mt-0.5">{error}</p>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label
                htmlFor="username"
                className="block text-xs font-medium uppercase tracking-wider text-slate-400 mb-2"
              >
                Username
              </label>
              <input
                id="username"
                type="text"
                autoComplete="username"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="analyst_1"
                className="w-full px-4 py-2.5 rounded-lg bg-slate-950 border border-slate-700/80 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-colors"
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className="block text-xs font-medium uppercase tracking-wider text-slate-400 mb-2"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full px-4 py-2.5 rounded-lg bg-slate-950 border border-slate-700/80 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-colors"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white font-medium text-sm transition-colors shadow-lg shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Signing In...</span>
                </>
              ) : (
                <>
                  <LogIn className="w-4 h-4" />
                  <span>Sign In</span>
                </>
              )}
            </button>
          </form>

          {/* Quick-fill helper for development */}
          {import.meta.env.DEV && (
            <div className="mt-6 pt-4 border-t border-slate-800 text-center">
              <button
                type="button"
                onClick={handleQuickFill}
                className="text-xs text-indigo-400 hover:text-indigo-300 underline underline-offset-4 transition-colors"
              >
                Use dev analyst credentials (analyst_1)
              </button>
            </div>
          )}
        </div>

        {/* Security & Imbalance Research Footer Notice */}
        <p className="text-center text-xs text-slate-500 mt-6">
          Gateway authentication required. All session activity is logged.
        </p>
      </div>
    </div>
  );
}

export default Login;

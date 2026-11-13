import React, { useState } from 'react';
import {
  Download,
  FileSpreadsheet,
  FileText,
  AlertTriangle,
  CheckCircle2,
  Filter,
  RotateCcw,
} from 'lucide-react';
import { exportTransactions } from '../api/client';

export function ExportView() {
  const [format, setFormat] = useState('csv');
  const [decision, setDecision] = useState('');
  const [explanationStatus, setExplanationStatus] = useState('');
  const [stepMin, setStepMin] = useState('');
  const [stepMax, setStepMax] = useState('');
  const [createdFrom, setCreatedFrom] = useState('');
  const [createdTo, setCreatedTo] = useState('');

  const [validationErrors, setValidationErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [successInfo, setSuccessInfo] = useState(null);

  const handleReset = () => {
    setFormat('csv');
    setDecision('');
    setExplanationStatus('');
    setStepMin('');
    setStepMax('');
    setCreatedFrom('');
    setCreatedTo('');
    setValidationErrors({});
    setApiError(null);
    setSuccessInfo(null);
  };

  const validate = () => {
    const errors = {};

    if (stepMin !== '') {
      const minNum = Number(stepMin);
      if (isNaN(minNum) || !Number.isInteger(minNum) || minNum < 1) {
        errors.stepMin = 'Step min must be an integer >= 1';
      }
    }

    if (stepMax !== '') {
      const maxNum = Number(stepMax);
      if (isNaN(maxNum) || !Number.isInteger(maxNum) || maxNum < 1) {
        errors.stepMax = 'Step max must be an integer >= 1';
      }
    }

    if (stepMin !== '' && stepMax !== '') {
      const minNum = Number(stepMin);
      const maxNum = Number(stepMax);
      if (!isNaN(minNum) && !isNaN(maxNum) && minNum > maxNum) {
        errors.stepRange = 'Step min cannot exceed Step max';
      }
    }

    if (createdFrom && createdTo) {
      const fromTime = new Date(createdFrom).getTime();
      const toTime = new Date(createdTo).getTime();
      if (!isNaN(fromTime) && !isNaN(toTime) && fromTime > toTime) {
        errors.dateRange = 'Date from cannot be later than Date to';
      }
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleExport = async (e) => {
    e.preventDefault();
    if (!validate()) return;

    setLoading(true);
    setApiError(null);
    setSuccessInfo(null);

    const params = { format };
    if (decision) params.decision = decision;
    if (explanationStatus) params.explanation_status = explanationStatus;
    if (stepMin !== '') params.step_min = parseInt(stepMin, 10);
    if (stepMax !== '') params.step_max = parseInt(stepMax, 10);
    if (createdFrom) params.created_from = new Date(createdFrom).toISOString();
    if (createdTo) params.created_to = new Date(createdTo).toISOString();

    try {
      const result = await exportTransactions(params);
      const blob = result.blob || result;
      const filename = result.filename || `transactions_export.${format}`;

      // Trigger browser download
      if (typeof window !== 'undefined' && window.URL && document.createElement) {
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = downloadUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
          if (window.URL.revokeObjectURL) {
            window.URL.revokeObjectURL(downloadUrl);
          }
          if (a.parentNode) {
            a.parentNode.removeChild(a);
          }
        }, 100);
      }

      setSuccessInfo({ filename, format });
    } catch (err) {
      setApiError(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6" data-testid="export-view">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-slate-800 gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <Download className="w-5 h-5 text-indigo-400" />
            Export Audit Logs
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Download triage logs, gateway decisions, and SHAP explanation records in authenticated CSV or PDF formats.
          </p>
        </div>

        <button
          type="button"
          onClick={handleReset}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Filters</span>
        </button>
      </div>

      {/* Main Export Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm space-y-6 max-w-4xl">
        <form onSubmit={handleExport} className="space-y-6" noValidate>
          {/* Format Selector */}
          <div>
            <span className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
              Export File Format
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <label
                data-testid="format-option-csv"
                className={`p-4 rounded-xl border cursor-pointer flex items-start gap-3 transition-all ${
                  format === 'csv'
                    ? 'bg-indigo-950/40 border-indigo-500 text-indigo-200 ring-1 ring-indigo-500/30'
                    : 'bg-slate-800/40 border-slate-800 text-slate-300 hover:bg-slate-800 hover:border-slate-700'
                }`}
              >
                <input
                  type="radio"
                  name="format"
                  value="csv"
                  checked={format === 'csv'}
                  onChange={() => setFormat('csv')}
                  className="sr-only"
                />
                <div className="w-9 h-9 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400 shrink-0">
                  <FileSpreadsheet className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-sm font-semibold flex items-center gap-2">
                    <span>CSV (Spreadsheet)</span>
                    {format === 'csv' && (
                      <span className="text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-1.5 py-0.5 rounded">
                        Selected
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Structured tabular data containing all PaySim transaction features, decisions, and feature attributions.
                  </p>
                </div>
              </label>

              <label
                data-testid="format-option-pdf"
                className={`p-4 rounded-xl border cursor-pointer flex items-start gap-3 transition-all ${
                  format === 'pdf'
                    ? 'bg-indigo-950/40 border-indigo-500 text-indigo-200 ring-1 ring-indigo-500/30'
                    : 'bg-slate-800/40 border-slate-800 text-slate-300 hover:bg-slate-800 hover:border-slate-700'
                }`}
              >
                <input
                  type="radio"
                  name="format"
                  value="pdf"
                  checked={format === 'pdf'}
                  onChange={() => setFormat('pdf')}
                  className="sr-only"
                />
                <div className="w-9 h-9 rounded-lg bg-rose-500/10 flex items-center justify-center text-rose-400 shrink-0">
                  <FileText className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-sm font-semibold flex items-center gap-2">
                    <span>PDF (Audit Report)</span>
                    {format === 'pdf' && (
                      <span className="text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-1.5 py-0.5 rounded">
                        Selected
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Executive summary and compliance documentation formatted for regulatory audit filings.
                  </p>
                </div>
              </label>
            </div>
          </div>

          {/* Query Filters */}
          <div className="pt-4 border-t border-slate-800/80 space-y-4">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-indigo-400" />
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Filter Audit Records (Optional)
              </h2>
            </div>

            {/* Decision & Status Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label htmlFor="filter-decision" className="block text-xs font-medium text-slate-300 mb-1">
                  Gateway Decision
                </label>
                <select
                  id="filter-decision"
                  value={decision}
                  onChange={(e) => setDecision(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="">All Decisions</option>
                  <option value="Low Risk">Low Risk</option>
                  <option value="Analyst Review">Analyst Review</option>
                  <option value="High Risk">High Risk</option>
                </select>
              </div>

              <div>
                <label htmlFor="filter-status" className="block text-xs font-medium text-slate-300 mb-1">
                  Explanation Reliability Status
                </label>
                <select
                  id="filter-status"
                  value={explanationStatus}
                  onChange={(e) => setExplanationStatus(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="">All Statuses</option>
                  <option value="Stable">Stable</option>
                  <option value="Unstable">Unstable</option>
                </select>
              </div>
            </div>

            {/* Step Range */}
            <div>
              <span className="block text-xs font-medium text-slate-300 mb-1">
                Simulation Step Range (Hours &gt;= 1)
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <input
                    type="number"
                    min="1"
                    placeholder="Min Step (e.g. 1)"
                    value={stepMin}
                    onChange={(e) => setStepMin(e.target.value)}
                    aria-label="Minimum step filter"
                    className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                      validationErrors.stepMin || validationErrors.stepRange
                        ? 'border-rose-500/80 focus:ring-rose-500'
                        : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                    }`}
                  />
                  {validationErrors.stepMin && (
                    <p className="text-[11px] text-rose-400 mt-1">{validationErrors.stepMin}</p>
                  )}
                </div>

                <div>
                  <input
                    type="number"
                    min="1"
                    placeholder="Max Step (e.g. 100)"
                    value={stepMax}
                    onChange={(e) => setStepMax(e.target.value)}
                    aria-label="Maximum step filter"
                    className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                      validationErrors.stepMax || validationErrors.stepRange
                        ? 'border-rose-500/80 focus:ring-rose-500'
                        : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                    }`}
                  />
                  {validationErrors.stepMax && (
                    <p className="text-[11px] text-rose-400 mt-1">{validationErrors.stepMax}</p>
                  )}
                </div>
              </div>
              {validationErrors.stepRange && (
                <p className="text-[11px] text-rose-400 mt-1">{validationErrors.stepRange}</p>
              )}
            </div>

            {/* Date Range */}
            <div>
              <span className="block text-xs font-medium text-slate-300 mb-1">
                Scoring Timestamp Range
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <input
                    type="datetime-local"
                    value={createdFrom}
                    onChange={(e) => setCreatedFrom(e.target.value)}
                    aria-label="Scoring time from"
                    className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
                <div>
                  <input
                    type="datetime-local"
                    value={createdTo}
                    onChange={(e) => setCreatedTo(e.target.value)}
                    aria-label="Scoring time to"
                    className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
              </div>
              {validationErrors.dateRange && (
                <p className="text-[11px] text-rose-400 mt-1">{validationErrors.dateRange}</p>
              )}
            </div>
          </div>

          {/* Success Banner */}
          {successInfo && (
            <div
              className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs flex items-center gap-3"
              data-testid="export-success-banner"
            >
              <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
              <div>
                <span className="font-semibold">Export Download Started:</span>{' '}
                <span className="font-mono">{successInfo.filename}</span>
              </div>
            </div>
          )}

          {/* API Error / Safety Cap Banner */}
          {apiError && (
            <div
              className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-start gap-3"
              data-testid="export-error-banner"
            >
              <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
              <div>
                <div className="font-semibold text-rose-200">
                  {apiError.status === 400
                    ? 'Export Row Limit Exceeded'
                    : apiError.status === 503
                    ? 'Database Service Unavailable'
                    : 'Export Request Failed'}
                </div>
                <p className="mt-0.5 text-rose-300">
                  {apiError.message || 'An error occurred while generating the audit log export.'}
                </p>
                {apiError.status === 503 && (
                  <p className="mt-1 text-slate-400 text-[11px]">
                    The database is currently unreachable. Please verify your connection or retry in a few moments.
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Action Row */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={loading}
              data-testid="download-export-button"
              className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs rounded-lg transition-colors flex items-center justify-center gap-2 shadow-sm disabled:opacity-50"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  <span>Generating {format.toUpperCase()} Export...</span>
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  <span>Download {format.toUpperCase()} Audit Log</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default ExportView;

import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Filter, X, AlertCircle } from 'lucide-react';

export function TransactionFilterBar({
  hideDecision = false,
  className = '',
}) {
  const [searchParams, setSearchParams] = useSearchParams();

  // Local form state initialized from URL search params
  const [decision, setDecision] = useState(searchParams.get('decision') || '');
  const [explanationStatus, setExplanationStatus] = useState(
    searchParams.get('explanation_status') || ''
  );
  const [stepMin, setStepMin] = useState(searchParams.get('step_min') || '');
  const [stepMax, setStepMax] = useState(searchParams.get('step_max') || '');
  const [createdFrom, setCreatedFrom] = useState(searchParams.get('created_from') || '');
  const [createdTo, setCreatedTo] = useState(searchParams.get('created_to') || '');
  const [validationError, setValidationError] = useState(null);

  // Keep local form in sync with URL search params changes
  useEffect(() => {
    if (!hideDecision) {
      setDecision(searchParams.get('decision') || '');
    }
    setExplanationStatus(searchParams.get('explanation_status') || '');
    setStepMin(searchParams.get('step_min') || '');
    setStepMax(searchParams.get('step_max') || '');
    setCreatedFrom(searchParams.get('created_from') || '');
    setCreatedTo(searchParams.get('created_to') || '');
  }, [searchParams, hideDecision]);

  const applyFilters = (newParams) => {
    // Validate step range
    const minVal = newParams.step_min !== '' && newParams.step_min !== undefined ? Number(newParams.step_min) : null;
    const maxVal = newParams.step_max !== '' && newParams.step_max !== undefined ? Number(newParams.step_max) : null;

    if (minVal !== null && maxVal !== null && minVal > maxVal) {
      setValidationError('Min step cannot exceed Max step.');
      return;
    }
    setValidationError(null);

    const updated = new URLSearchParams(searchParams);

    // Apply or delete filters
    Object.entries(newParams).forEach(([key, val]) => {
      if (val !== undefined && val !== null && String(val).trim() !== '') {
        updated.set(key, String(val).trim());
      } else {
        updated.delete(key);
      }
    });

    // Condition 4: ANY change in filter inputs resets page to 1
    updated.set('page', '1');

    setSearchParams(updated);
  };

  const handleApply = (e) => {
    if (e) e.preventDefault();
    const params = {
      explanation_status: explanationStatus,
      step_min: stepMin,
      step_max: stepMax,
      created_from: createdFrom,
      created_to: createdTo,
    };
    if (!hideDecision) {
      params.decision = decision;
    }
    applyFilters(params);
  };

  const handleReset = () => {
    setDecision('');
    setExplanationStatus('');
    setStepMin('');
    setStepMax('');
    setCreatedFrom('');
    setCreatedTo('');
    setValidationError(null);

    const updated = new URLSearchParams();
    // Preserve existing decision if locked/hidden (e.g. Flagged Queue tab)
    if (hideDecision && searchParams.get('decision')) {
      updated.set('decision', searchParams.get('decision'));
    }
    const currentSize = searchParams.get('page_size');
    if (currentSize) {
      updated.set('page_size', currentSize);
    }
    updated.set('page', '1');
    setSearchParams(updated);
  };

  const hasActiveFilters = Boolean(
    (!hideDecision && decision) ||
      explanationStatus ||
      stepMin ||
      stepMax ||
      createdFrom ||
      createdTo
  );

  return (
    <form
      onSubmit={handleApply}
      className={`p-4 rounded-xl border border-slate-800 bg-slate-900/70 space-y-3 ${className}`}
      data-testid="transaction-filter-bar"
    >
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-emerald-400" aria-hidden="true" />
          <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Filter Transactions
          </h3>
        </div>
        {hasActiveFilters && (
          <button
            type="button"
            onClick={handleReset}
            className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-rose-400 transition-colors cursor-pointer"
            data-testid="clear-filters-btn"
          >
            <X className="w-3.5 h-3.5" aria-hidden="true" />
            <span>Clear Filters</span>
          </button>
        )}
      </div>

      {validationError && (
        <div
          role="alert"
          className="p-2 rounded bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2"
          data-testid="filter-validation-error"
        >
          <AlertCircle className="w-4 h-4 shrink-0" aria-hidden="true" />
          <span>{validationError}</span>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {/* Gateway Decision (Hidden in Flagged Queue tabs) */}
        {!hideDecision && (
          <div>
            <label htmlFor="filter-decision" className="block text-[11px] font-medium text-slate-400 mb-1">
              Gateway Decision
            </label>
            <select
              id="filter-decision"
              value={decision}
              onChange={(e) => setDecision(e.target.value)}
              className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500"
              data-testid="filter-decision-select"
            >
              <option value="">All Decisions</option>
              <option value="Low Risk">Low Risk</option>
              <option value="High Risk">High Risk</option>
              <option value="Analyst Review">Analyst Review</option>
            </select>
          </div>
        )}

        {/* Explanation Status */}
        <div>
          <label htmlFor="filter-explanation" className="block text-[11px] font-medium text-slate-400 mb-1">
            Explanation Status
          </label>
          <select
            id="filter-explanation"
            value={explanationStatus}
            onChange={(e) => setExplanationStatus(e.target.value)}
            className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500"
            data-testid="filter-explanation-select"
          >
            <option value="">All Statuses</option>
            <option value="Stable">Stable</option>
            <option value="Unstable">Unstable</option>
          </select>
        </div>

        {/* Step Min */}
        <div>
          <label htmlFor="filter-step-min" className="block text-[11px] font-medium text-slate-400 mb-1">
            Min Step
          </label>
          <input
            id="filter-step-min"
            type="number"
            min="0"
            placeholder="e.g. 1"
            value={stepMin}
            onChange={(e) => setStepMin(e.target.value)}
            className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500 font-mono"
            data-testid="filter-step-min-input"
          />
        </div>

        {/* Step Max */}
        <div>
          <label htmlFor="filter-step-max" className="block text-[11px] font-medium text-slate-400 mb-1">
            Max Step
          </label>
          <input
            id="filter-step-max"
            type="number"
            min="0"
            placeholder="e.g. 743"
            value={stepMax}
            onChange={(e) => setStepMax(e.target.value)}
            className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500 font-mono"
            data-testid="filter-step-max-input"
          />
        </div>

        {/* Created From */}
        <div>
          <label htmlFor="filter-created-from" className="block text-[11px] font-medium text-slate-400 mb-1">
            Created From
          </label>
          <input
            id="filter-created-from"
            type="date"
            value={createdFrom}
            onChange={(e) => setCreatedFrom(e.target.value)}
            className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500"
            data-testid="filter-created-from-input"
          />
        </div>

        {/* Created To */}
        <div>
          <label htmlFor="filter-created-to" className="block text-[11px] font-medium text-slate-400 mb-1">
            Created To
          </label>
          <input
            id="filter-created-to"
            type="date"
            value={createdTo}
            onChange={(e) => setCreatedTo(e.target.value)}
            className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500"
            data-testid="filter-created-to-input"
          />
        </div>
      </div>

      <div className="flex justify-end pt-1">
        <button
          type="submit"
          className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-semibold text-xs transition-colors shadow-sm cursor-pointer"
          data-testid="apply-filters-btn"
        >
          <span>Apply Filters</span>
        </button>
      </div>
    </form>
  );
}

export default TransactionFilterBar;

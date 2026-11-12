import React, { useState } from 'react';
import {
  PlayCircle,
  Sparkles,
  RotateCcw,
  AlertTriangle,
  AlertOctagon,
  HelpCircle,
  Cpu,
  Layers,
  Clock,
  CheckCircle2,
} from 'lucide-react';
import { scoreTransaction } from '../api/client';
import { DecisionBadge } from '../components/shared/DecisionBadge';
import { ReliabilityBadge } from '../components/shared/ReliabilityBadge';
import { RiskScoreBar } from '../components/shared/RiskScoreBar';
import { RuleFlagList } from '../components/shared/RuleFlagList';
import { ShapFeatureBars } from '../components/shared/ShapFeatureBars';

const PRESETS = [
  {
    id: 'fraud',
    label: 'Obvious Fraud',
    desc: 'Large transfer draining origin balance',
    data: {
      step: '1',
      type: 'TRANSFER',
      amount: '181000.00',
      name_orig: 'C1231006815',
      old_balance_orig: '181000.00',
      new_balance_orig: '0.00',
      name_dest: 'C553264065',
      old_balance_dest: '0.00',
      new_balance_dest: '0.00',
    },
  },
  {
    id: 'legit',
    label: 'Obvious Legitimate',
    desc: 'Small merchant payment with modest balance delta',
    data: {
      step: '1',
      type: 'PAYMENT',
      amount: '25.50',
      name_orig: 'C1000000001',
      old_balance_orig: '500.00',
      new_balance_orig: '474.50',
      name_dest: 'M987654321',
      old_balance_dest: '0.00',
      new_balance_dest: '0.00',
    },
  },
  {
    id: 'borderline',
    label: 'Borderline / Ambiguous',
    desc: 'Moderate transfer triggering Unstable SHAP explanation',
    data: {
      step: '1',
      type: 'TRANSFER',
      amount: '65000.00',
      name_orig: 'C3000000003',
      old_balance_orig: '80000.00',
      new_balance_orig: '15000.00',
      name_dest: 'C4000000004',
      old_balance_dest: '20000.00',
      new_balance_dest: '85000.00',
    },
  },
  {
    id: 'zero',
    label: 'Zero Amount',
    desc: 'Boundary edge case with zero monetary transaction',
    data: {
      step: '1',
      type: 'PAYMENT',
      amount: '0.00',
      name_orig: 'C5000000005',
      old_balance_orig: '100.00',
      new_balance_orig: '100.00',
      name_dest: 'M6000000006',
      old_balance_dest: '0.00',
      new_balance_dest: '0.00',
    },
  },
];

const INITIAL_FORM = {
  step: '1',
  type: 'TRANSFER',
  amount: '181000.00',
  name_orig: 'C1231006815',
  old_balance_orig: '181000.00',
  new_balance_orig: '0.00',
  name_dest: 'C553264065',
  old_balance_dest: '0.00',
  new_balance_dest: '0.00',
};

export function TestTransactionView() {
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [result, setResult] = useState(null);
  const [showLatencyTooltip, setShowLatencyTooltip] = useState(false);
  const [activePreset, setActivePreset] = useState('fraud');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setActivePreset(null);
    if (errors[name]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
    }
  };

  const handleApplyPreset = (preset) => {
    setFormData(preset.data);
    setActivePreset(preset.id);
    setErrors({});
    setApiError(null);
  };

  const handleReset = () => {
    setFormData(INITIAL_FORM);
    setActivePreset('fraud');
    setErrors({});
    setApiError(null);
    setResult(null);
  };

  const validate = () => {
    const nextErrors = {};

    const stepNum = Number(formData.step);
    if (!formData.step || isNaN(stepNum) || !Number.isInteger(stepNum) || stepNum < 1) {
      nextErrors.step = 'Step must be an integer >= 1';
    }

    if (!formData.type) {
      nextErrors.type = 'Transaction type is required';
    }

    const amountNum = Number(formData.amount);
    if (formData.amount === '' || isNaN(amountNum) || amountNum < 0) {
      nextErrors.amount = 'Amount must be >= 0';
    }

    if (!formData.name_orig || !formData.name_orig.trim()) {
      nextErrors.name_orig = 'Origin account ID is required';
    }

    const oldBalOrig = Number(formData.old_balance_orig);
    if (formData.old_balance_orig === '' || isNaN(oldBalOrig) || oldBalOrig < 0) {
      nextErrors.old_balance_orig = 'Old balance origin must be >= 0';
    }

    const newBalOrig = Number(formData.new_balance_orig);
    if (formData.new_balance_orig === '' || isNaN(newBalOrig) || newBalOrig < 0) {
      nextErrors.new_balance_orig = 'New balance origin must be >= 0';
    }

    if (!formData.name_dest || !formData.name_dest.trim()) {
      nextErrors.name_dest = 'Destination account ID is required';
    }

    const oldBalDest = Number(formData.old_balance_dest);
    if (formData.old_balance_dest === '' || isNaN(oldBalDest) || oldBalDest < 0) {
      nextErrors.old_balance_dest = 'Old balance destination must be >= 0';
    }

    const newBalDest = Number(formData.new_balance_dest);
    if (formData.new_balance_dest === '' || isNaN(newBalDest) || newBalDest < 0) {
      nextErrors.new_balance_dest = 'New balance destination must be >= 0';
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validate()) return;

    setLoading(true);
    setApiError(null);

    const payload = {
      step: parseInt(formData.step, 10),
      type: formData.type,
      amount: parseFloat(formData.amount),
      name_orig: formData.name_orig.trim(),
      old_balance_orig: parseFloat(formData.old_balance_orig),
      new_balance_orig: parseFloat(formData.new_balance_orig),
      name_dest: formData.name_dest.trim(),
      old_balance_dest: parseFloat(formData.old_balance_dest),
      new_balance_dest: parseFloat(formData.new_balance_dest),
    };

    try {
      const pred = await scoreTransaction(payload);
      setResult(pred);
    } catch (err) {
      setApiError(err.message || 'Scoring pipeline evaluation failed');
    } finally {
      setLoading(false);
    }
  };

  const isStubMode = result?.model_version === 'stub' || result?.model_version?.toLowerCase().includes('stub');
  const isUnstable = result?.explanation_status === 'Unstable';

  return (
    <div className="space-y-6" data-testid="test-transaction-view">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-slate-800 gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <PlayCircle className="w-5 h-5 text-indigo-400" />
            Test Single Transaction
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Simulate PaySim transaction features and evaluate gateway routing, risk scoring, and SHAP explanations in real-time.
          </p>
        </div>

        <button
          type="button"
          onClick={handleReset}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Form</span>
        </button>
      </div>

      {/* Preset Selector */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm space-y-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-amber-400" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Quick Simulation Presets
          </h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
          {PRESETS.map((p) => {
            const isSelected = activePreset === p.id;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => handleApplyPreset(p)}
                data-testid={`preset-button-${p.id}`}
                className={`text-left p-3 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-indigo-950/40 border-indigo-500/50 text-indigo-200 ring-1 ring-indigo-500/30'
                    : 'bg-slate-800/40 border-slate-800 text-slate-300 hover:bg-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="text-xs font-semibold flex items-center justify-between">
                  <span>{p.label}</span>
                  {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400" />}
                </div>
                <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">{p.desc}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Grid: Input Form (Left) & Result Card (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Input Form (7 cols on lg) */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-white pb-3 border-b border-slate-800 mb-4">
            PaySim Transaction Attributes
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            {/* Row 1: Step & Type */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label htmlFor="step" className="block text-xs font-medium text-slate-300 mb-1">
                  Simulation Step (Hour &gt;= 1) <span className="text-rose-400">*</span>
                </label>
                <input
                  id="step"
                  name="step"
                  type="number"
                  min="1"
                  step="1"
                  value={formData.step}
                  onChange={handleChange}
                  className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                    errors.step
                      ? 'border-rose-500/80 focus:ring-rose-500'
                      : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                  }`}
                />
                {errors.step && <p className="text-[11px] text-rose-400 mt-1">{errors.step}</p>}
              </div>

              <div>
                <label htmlFor="type" className="block text-xs font-medium text-slate-300 mb-1">
                  Transaction Type <span className="text-rose-400">*</span>
                </label>
                <select
                  id="type"
                  name="type"
                  value={formData.type}
                  onChange={handleChange}
                  className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white focus:outline-none focus:ring-1 ${
                    errors.type
                      ? 'border-rose-500/80 focus:ring-rose-500'
                      : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                  }`}
                >
                  <option value="PAYMENT">PAYMENT</option>
                  <option value="TRANSFER">TRANSFER</option>
                  <option value="CASH_OUT">CASH_OUT</option>
                  <option value="DEBIT">DEBIT</option>
                  <option value="CASH_IN">CASH_IN</option>
                </select>
                {errors.type && <p className="text-[11px] text-rose-400 mt-1">{errors.type}</p>}
              </div>
            </div>

            {/* Row 2: Amount */}
            <div>
              <label htmlFor="amount" className="block text-xs font-medium text-slate-300 mb-1">
                Amount (Currency Units) <span className="text-rose-400">*</span>
              </label>
              <input
                id="amount"
                name="amount"
                type="number"
                min="0"
                step="any"
                value={formData.amount}
                onChange={handleChange}
                className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                  errors.amount
                    ? 'border-rose-500/80 focus:ring-rose-500'
                    : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                }`}
              />
              {errors.amount && <p className="text-[11px] text-rose-400 mt-1">{errors.amount}</p>}
            </div>

            {/* Row 3: Origin Details */}
            <div className="pt-2 border-t border-slate-800/80 space-y-3">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Origin Account (Sender)
              </span>

              <div>
                <label htmlFor="name_orig" className="block text-xs font-medium text-slate-300 mb-1">
                  Origin Account ID <span className="text-rose-400">*</span>
                </label>
                <input
                  id="name_orig"
                  name="name_orig"
                  type="text"
                  value={formData.name_orig}
                  onChange={handleChange}
                  placeholder="e.g. C123456789"
                  className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                    errors.name_orig
                      ? 'border-rose-500/80 focus:ring-rose-500'
                      : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                  }`}
                />
                {errors.name_orig && <p className="text-[11px] text-rose-400 mt-1">{errors.name_orig}</p>}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="old_balance_orig" className="block text-xs font-medium text-slate-300 mb-1">
                    Old Balance Origin <span className="text-rose-400">*</span>
                  </label>
                  <input
                    id="old_balance_orig"
                    name="old_balance_orig"
                    type="number"
                    min="0"
                    step="any"
                    value={formData.old_balance_orig}
                    onChange={handleChange}
                    className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                      errors.old_balance_orig
                        ? 'border-rose-500/80 focus:ring-rose-500'
                        : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                    }`}
                  />
                  {errors.old_balance_orig && (
                    <p className="text-[11px] text-rose-400 mt-1">{errors.old_balance_orig}</p>
                  )}
                </div>

                <div>
                  <label htmlFor="new_balance_orig" className="block text-xs font-medium text-slate-300 mb-1">
                    New Balance Origin <span className="text-rose-400">*</span>
                  </label>
                  <input
                    id="new_balance_orig"
                    name="new_balance_orig"
                    type="number"
                    min="0"
                    step="any"
                    value={formData.new_balance_orig}
                    onChange={handleChange}
                    className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                      errors.new_balance_orig
                        ? 'border-rose-500/80 focus:ring-rose-500'
                        : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                    }`}
                  />
                  {errors.new_balance_orig && (
                    <p className="text-[11px] text-rose-400 mt-1">{errors.new_balance_orig}</p>
                  )}
                </div>
              </div>
            </div>

            {/* Row 4: Destination Details */}
            <div className="pt-2 border-t border-slate-800/80 space-y-3">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Destination Account (Recipient)
              </span>

              <div>
                <label htmlFor="name_dest" className="block text-xs font-medium text-slate-300 mb-1">
                  Destination Account ID <span className="text-rose-400">*</span>
                </label>
                <input
                  id="name_dest"
                  name="name_dest"
                  type="text"
                  value={formData.name_dest}
                  onChange={handleChange}
                  placeholder="e.g. M123456789 or C987654321"
                  className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                    errors.name_dest
                      ? 'border-rose-500/80 focus:ring-rose-500'
                      : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                  }`}
                />
                {errors.name_dest && <p className="text-[11px] text-rose-400 mt-1">{errors.name_dest}</p>}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="old_balance_dest" className="block text-xs font-medium text-slate-300 mb-1">
                    Old Balance Destination <span className="text-rose-400">*</span>
                  </label>
                  <input
                    id="old_balance_dest"
                    name="old_balance_dest"
                    type="number"
                    min="0"
                    step="any"
                    value={formData.old_balance_dest}
                    onChange={handleChange}
                    className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                      errors.old_balance_dest
                        ? 'border-rose-500/80 focus:ring-rose-500'
                        : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                    }`}
                  />
                  {errors.old_balance_dest && (
                    <p className="text-[11px] text-rose-400 mt-1">{errors.old_balance_dest}</p>
                  )}
                </div>

                <div>
                  <label htmlFor="new_balance_dest" className="block text-xs font-medium text-slate-300 mb-1">
                    New Balance Destination <span className="text-rose-400">*</span>
                  </label>
                  <input
                    id="new_balance_dest"
                    name="new_balance_dest"
                    type="number"
                    min="0"
                    step="any"
                    value={formData.new_balance_dest}
                    onChange={handleChange}
                    className={`w-full px-3 py-2 bg-slate-800/80 border rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 ${
                      errors.new_balance_dest
                        ? 'border-rose-500/80 focus:ring-rose-500'
                        : 'border-slate-700 focus:ring-indigo-500 focus:border-indigo-500'
                    }`}
                  />
                  {errors.new_balance_dest && (
                    <p className="text-[11px] text-rose-400 mt-1">{errors.new_balance_dest}</p>
                  )}
                </div>
              </div>
            </div>

            {/* Error display */}
            {apiError && (
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{apiError}</span>
              </div>
            )}

            {/* Submit Button */}
            <div className="pt-2">
              <button
                type="submit"
                disabled={loading}
                data-testid="submit-transaction-button"
                className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs rounded-lg transition-colors flex items-center justify-center gap-2 shadow-sm disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                    <span>Evaluating Pipeline...</span>
                  </>
                ) : (
                  <>
                    <PlayCircle className="w-4 h-4" />
                    <span>Score Transaction</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Prediction Output Card (5 cols on lg) */}
        <div className="lg:col-span-5 space-y-4">
          {result ? (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-5" data-testid="prediction-result-card">
              {/* Header / Badges */}
              <div className="flex items-start justify-between pb-3 border-b border-slate-800 gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-white">Scoring Evaluation Result</h3>
                  <p className="text-[11px] font-mono text-slate-400 mt-0.5 break-all">
                    ID: {result.transaction_id || 'Simulated'}
                  </p>
                </div>
                <div className="flex flex-col items-end gap-1.5">
                  <DecisionBadge decision={result.gateway_decision} />
                  <ReliabilityBadge status={result.explanation_status} />
                </div>
              </div>

              {/* Stub mode banner */}
              {isStubMode && (
                <div
                  className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-200 text-xs flex items-start gap-2"
                  data-testid="stub-mode-banner"
                >
                  <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
                  <div>
                    <span className="font-semibold text-amber-300">Stub Mode Active:</span> Model weights are not loaded. Scoring returned fallback heuristic outputs.
                  </div>
                </div>
              )}

              {/* Unstable explanation banner */}
              {isUnstable && (
                <div
                  className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-200 text-xs flex items-start gap-2"
                  data-testid="unstable-explanation-banner"
                >
                  <AlertOctagon className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
                  <div>
                    <span className="font-semibold text-amber-300">Unstable Explanation:</span> SHAP feature attributions shift across imbalance resampling strategies. Routed to Analyst Review.
                  </div>
                </div>
              )}

              {/* Risk Score */}
              <div className="space-y-1.5">
                <span className="text-xs font-semibold text-slate-300">Calibrated Risk Score</span>
                <RiskScoreBar score={result.risk_score} />
              </div>

              {/* SHAP Feature Waterfall / Bars */}
              <div className="pt-2 border-t border-slate-800/80">
                <ShapFeatureBars features={result.top_features} />
              </div>

              {/* Rule Flags */}
              <div className="pt-2 border-t border-slate-800/80">
                <RuleFlagList flags={result.rule_flags} />
              </div>

              {/* Pipeline Telemetry Metadata */}
              <div className="pt-3 border-t border-slate-800/80 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="p-2 rounded bg-slate-800/40 border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-medium uppercase tracking-wider flex items-center justify-center gap-1">
                    <Cpu className="w-3 h-3 text-slate-400" />
                    <span>Model</span>
                  </div>
                  <div className="text-xs font-mono font-semibold text-slate-200 mt-1 truncate">
                    {result.model_version || 'v1.0.0'}
                  </div>
                </div>

                <div className="p-2 rounded bg-slate-800/40 border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-medium uppercase tracking-wider flex items-center justify-center gap-1">
                    <Layers className="w-3 h-3 text-slate-400" />
                    <span>Strategy</span>
                  </div>
                  <div className="text-xs font-mono font-semibold text-slate-200 mt-1 truncate">
                    {result.imbalance_strategy || 'class_weight'}
                  </div>
                </div>

                <div className="p-2 rounded bg-slate-800/40 border border-slate-800 relative">
                  <div className="text-[10px] text-slate-400 font-medium uppercase tracking-wider flex items-center justify-center gap-1">
                    <Clock className="w-3 h-3 text-slate-400" />
                    <span>Latency</span>
                    <button
                      type="button"
                      onMouseEnter={() => setShowLatencyTooltip(true)}
                      onMouseLeave={() => setShowLatencyTooltip(false)}
                      onFocus={() => setShowLatencyTooltip(true)}
                      onBlur={() => setShowLatencyTooltip(false)}
                      aria-label="Latency calculation details"
                      className="text-slate-400 hover:text-slate-200 focus:outline-none"
                    >
                      <HelpCircle className="w-3 h-3" />
                    </button>
                    {showLatencyTooltip && (
                      <div
                        role="tooltip"
                        className="absolute z-20 left-1/2 -translate-x-1/2 bottom-full mb-2 w-48 p-2 bg-slate-800 border border-slate-700 text-slate-200 text-[11px] rounded shadow-lg pointer-events-none text-center"
                      >
                        Server-side scoring + explanation time
                      </div>
                    )}
                  </div>
                  <div className="text-xs font-mono font-semibold text-cyan-400 mt-1">
                    {typeof result.latency_ms === 'number' ? `${result.latency_ms.toFixed(1)} ms` : '—'}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-slate-800 bg-slate-900/40 p-10 text-center text-slate-400 text-xs space-y-2">
              <PlayCircle className="w-8 h-8 text-slate-600 mx-auto" />
              <div className="font-semibold text-slate-300">No Simulation Run Yet</div>
              <p className="text-slate-500 max-w-xs mx-auto">
                Select a preset above or input custom transaction attributes, then click &quot;Score Transaction&quot; to inspect gateway routing and SHAP explanations.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default TestTransactionView;

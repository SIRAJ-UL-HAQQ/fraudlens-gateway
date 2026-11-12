import React, { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { AlertCircle, Check, Loader2 } from 'lucide-react';
import { submitReview } from '../../api/client';

export function AnalystReviewForm({ transactionId, onSuccess, className = '' }) {
  const queryClient = useQueryClient();
  const [outcome, setOutcome] = useState('');
  const [note, setNote] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!outcome) {
      setErrorMessage('Please select an adjudication outcome.');
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);
    setSuccessMessage(null);

    try {
      // Matches AnalystReviewIn schema in docs/openapi.json: { outcome, note }
      const payload = {
        outcome,
        note: note.trim() ? note.trim() : null,
      };

      await submitReview(transactionId, payload);

      // Invalidate queries to refresh detail and feed without optimistic update
      await queryClient.invalidateQueries({ queryKey: ['transaction', transactionId] });
      await queryClient.invalidateQueries({ queryKey: ['transactions'] });

      setSuccessMessage('Review submitted successfully.');
      setOutcome('');
      setNote('');

      if (onSuccess) {
        onSuccess();
      }
    } catch (err) {
      // Input preservation: preserve note and outcome on error (Condition 3)
      setErrorMessage(err.message || 'Failed to submit review.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={`p-4 rounded-lg border border-slate-800 bg-slate-900/40 space-y-4 ${className}`}
      data-testid="analyst-review-form"
    >
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
          Submit Analyst Review
        </h4>
        <span className="text-[11px] text-slate-500">Adjudicate Alert</span>
      </div>

      {errorMessage && (
        <div
          role="alert"
          className="p-2.5 rounded bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-2"
          data-testid="review-error-banner"
        >
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" aria-hidden="true" />
          <span>{errorMessage}</span>
        </div>
      )}

      {successMessage && (
        <div
          role="status"
          className="p-2.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs flex items-start gap-2"
          data-testid="review-success-banner"
        >
          <Check className="w-4 h-4 shrink-0 mt-0.5" aria-hidden="true" />
          <span>{successMessage}</span>
        </div>
      )}

      <fieldset className="space-y-2">
        <legend className="text-xs font-medium text-slate-300 mb-1">
          Adjudication Outcome <span className="text-rose-400">*</span>
        </legend>
        <div className="grid grid-cols-2 gap-2">
          <label
            className={`flex items-center gap-2 p-2.5 rounded-lg border text-xs font-medium cursor-pointer transition-colors ${
              outcome === 'Confirmed Fraud'
                ? 'bg-rose-500/15 border-rose-500/50 text-rose-300'
                : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <input
              type="radio"
              name="outcome"
              value="Confirmed Fraud"
              checked={outcome === 'Confirmed Fraud'}
              onChange={(e) => setOutcome(e.target.value)}
              disabled={isSubmitting}
              className="accent-rose-500"
            />
            <span>Confirmed Fraud</span>
          </label>

          <label
            className={`flex items-center gap-2 p-2.5 rounded-lg border text-xs font-medium cursor-pointer transition-colors ${
              outcome === 'False Positive'
                ? 'bg-emerald-500/15 border-emerald-500/50 text-emerald-300'
                : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <input
              type="radio"
              name="outcome"
              value="False Positive"
              checked={outcome === 'False Positive'}
              onChange={(e) => setOutcome(e.target.value)}
              disabled={isSubmitting}
              className="accent-emerald-500"
            />
            <span>False Positive</span>
          </label>
        </div>
      </fieldset>

      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <label htmlFor="review-note" className="text-xs font-medium text-slate-300">
            Analyst Notes (Optional)
          </label>
          <span className="text-[11px] text-slate-500 font-mono" data-testid="note-char-counter">
            {note.length} characters
          </span>
        </div>
        <textarea
          id="review-note"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Document investigation findings, corroborating factors, or false alarm rationale..."
          rows={3}
          disabled={isSubmitting}
          className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-200 text-xs placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500 transition-colors disabled:opacity-50"
        />
        <p className="text-[10px] text-slate-500 italic">
          Audit note is committed permanently to transaction record.
        </p>
      </div>

      <div className="flex justify-end pt-1">
        <button
          type="submit"
          disabled={isSubmitting || !outcome}
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-slate-950 font-semibold text-xs transition-colors shadow-sm cursor-pointer disabled:cursor-not-allowed"
          data-testid="submit-review-btn"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
              <span>Submitting...</span>
            </>
          ) : (
            <span>Submit Review</span>
          )}
        </button>
      </div>
    </form>
  );
}

export default AnalystReviewForm;

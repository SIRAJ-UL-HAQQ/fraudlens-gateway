import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ReviewHistoryTimeline from '../components/transactions/ReviewHistoryTimeline';
import AnalystReviewForm from '../components/transactions/AnalystReviewForm';
import * as client from '../api/client';

function renderWithClient(ui) {
  const testClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return render(<QueryClientProvider client={testClient}>{ui}</QueryClientProvider>);
}

describe('Review workflow components', () => {
  describe('ReviewHistoryTimeline', () => {
    it('renders empty message when no reviews exist', () => {
      render(<ReviewHistoryTimeline reviews={[]} />);
      expect(screen.getByTestId('empty-review-history')).toBeInTheDocument();
      expect(screen.getByText('No analyst reviews recorded yet.')).toBeInTheDocument();
    });

    it('renders append-only audit trail items in newest-first order with outcomes and notes', () => {
      const reviews = [
        {
          review_id: 'rev-1',
          outcome: 'False Positive',
          note: 'Customer verified travel transaction.',
          reviewed_by: 'analyst_alice',
          reviewed_at: '2026-10-15T10:00:00Z',
        },
        {
          review_id: 'rev-2',
          outcome: 'Confirmed Fraud',
          note: 'Confirmed account takeover.',
          reviewed_by: 'analyst_bob',
          reviewed_at: '2026-10-15T12:00:00Z',
        },
      ];

      render(<ReviewHistoryTimeline reviews={reviews} />);

      const items = screen.getAllByTestId('review-history-item');
      expect(items.length).toBe(2);

      // Newest (rev-2 at 12:00) should appear first
      expect(items[0]).toHaveTextContent('Confirmed Fraud');
      expect(items[0]).toHaveTextContent('analyst_bob');
      expect(items[0]).toHaveTextContent('Confirmed account takeover.');

      // Older (rev-1 at 10:00) should appear second
      expect(items[1]).toHaveTextContent('False Positive');
      expect(items[1]).toHaveTextContent('analyst_alice');
      expect(items[1]).toHaveTextContent('Customer verified travel transaction.');
    });
  });

  describe('AnalystReviewForm', () => {
    beforeEach(() => {
      vi.restoreAllMocks();
    });

    it('validates that an outcome must be selected before submission', async () => {
      renderWithClient(<AnalystReviewForm transactionId="tx-123" />);

      const submitBtn = screen.getByTestId('submit-review-btn');
      expect(submitBtn).toBeDisabled();

      // Typing only a note does not enable submit
      const textarea = screen.getByPlaceholderText(/Document investigation findings/i);
      fireEvent.change(textarea, { target: { value: 'Suspicious amount' } });
      expect(submitBtn).toBeDisabled();

      // Selecting an outcome enables submit
      const fraudRadio = screen.getByLabelText('Confirmed Fraud');
      fireEvent.click(fraudRadio);
      expect(submitBtn).not.toBeDisabled();
    });

    it('submits review successfully and calls submitReview with outcome and note', async () => {
      const submitSpy = vi.spyOn(client, 'submitReview').mockResolvedValueOnce({
        review_id: 'rev-new',
        outcome: 'Confirmed Fraud',
        note: 'High confidence fraud',
        reviewed_by: 'test_analyst',
        reviewed_at: '2026-10-15T14:00:00Z',
      });
      const onSuccess = vi.fn();

      renderWithClient(
        <AnalystReviewForm transactionId="tx-123" onSuccess={onSuccess} />
      );

      fireEvent.click(screen.getByLabelText('Confirmed Fraud'));
      const textarea = screen.getByPlaceholderText(/Document investigation findings/i);
      fireEvent.change(textarea, { target: { value: 'High confidence fraud' } });

      // Character counter updates (21 characters)
      expect(screen.getByTestId('note-char-counter')).toHaveTextContent('21 characters');

      fireEvent.click(screen.getByTestId('submit-review-btn'));

      await waitFor(() => {
        expect(submitSpy).toHaveBeenCalledWith('tx-123', {
          outcome: 'Confirmed Fraud',
          note: 'High confidence fraud',
        });
        expect(onSuccess).toHaveBeenCalled();
        expect(screen.getByTestId('review-success-banner')).toBeInTheDocument();
      });
    });

    it('preserves user input (note and outcome) when submission fails (Condition 3)', async () => {
      vi.spyOn(client, 'submitReview').mockRejectedValueOnce(
        new Error('Service temporarily unavailable. Please retry.')
      );

      renderWithClient(<AnalystReviewForm transactionId="tx-123" />);

      fireEvent.click(screen.getByLabelText('False Positive'));
      const textarea = screen.getByPlaceholderText(/Document investigation findings/i);
      fireEvent.change(textarea, { target: { value: 'Critical analyst investigation notes' } });

      fireEvent.click(screen.getByTestId('submit-review-btn'));

      await waitFor(() => {
        expect(screen.getByTestId('review-error-banner')).toHaveTextContent(
          'Service temporarily unavailable. Please retry.'
        );
      });

      // Assert user input was preserved (NOT wiped out)
      expect(textarea.value).toBe('Critical analyst investigation notes');
      expect(screen.getByLabelText('False Positive')).toBeChecked();
    });
  });
});

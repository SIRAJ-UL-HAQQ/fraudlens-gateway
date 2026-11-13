import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import FlaggedQueue from '../pages/FlaggedQueue';
import * as client from '../api/client';

function renderFlagged(initialEntries = ['/flagged']) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={initialEntries}>
        <FlaggedQueue />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('FlaggedQueue page (Condition 2: Tabs & Variant B Banner)', () => {
  const mockAnalystReviewItems = [
    {
      transaction_id: 'tx-rev-1',
      step: 25,
      type: 'TRANSFER',
      amount: 80000,
      name_orig: 'C1000',
      name_dest: 'C2000',
      risk_score: 0.78,
      gateway_decision: 'Analyst Review',
      explanation_status: 'Unstable',
      rule_flags: [],
      reviewed: false,
      created_at: '2026-10-15T12:00:00Z',
    },
  ];

  const mockHighRiskItems = [
    {
      transaction_id: 'tx-hr-1',
      step: 30,
      type: 'CASH_OUT',
      amount: 250000,
      name_orig: 'C3000',
      name_dest: 'C4000',
      risk_score: 0.96,
      gateway_decision: 'High Risk',
      explanation_status: 'Stable',
      rule_flags: ['LARGE_TRANSFER_AMOUNT'],
      reviewed: false,
      created_at: '2026-10-15T13:00:00Z',
    },
  ];

  beforeEach(() => {
    vi.restoreAllMocks();

    vi.spyOn(client, 'getTransactions').mockImplementation((params = {}) => {
      if (params.decision === 'Analyst Review') {
        return Promise.resolve({
          items: mockAnalystReviewItems,
          total: 14,
          page: params.page || 1,
          page_size: params.page_size || 50,
          total_pages: 1,
        });
      }
      if (params.decision === 'High Risk') {
        return Promise.resolve({
          items: mockHighRiskItems,
          total: 8,
          page: params.page || 1,
          page_size: params.page_size || 50,
          total_pages: 1,
        });
      }
      return Promise.resolve({
        items: [],
        total: 0,
        page: 1,
        page_size: 50,
        total_pages: 0,
      });
    });
  });

  it('renders heading, Variant B banner, and honest counts on segmented tab badges', async () => {
    renderFlagged();

    expect(screen.getByText('Flagged Transaction Queue')).toBeInTheDocument();

    // Condition 2: Variant B educational banner
    const banner = screen.getByTestId('variant-b-banner');
    expect(banner).toBeInTheDocument();
    expect(
      screen.getByText(/SRS Variant B Gateway Routing Logic/i)
    ).toBeInTheDocument();

    // Tab count badges populate honestly from API queries (14 and 8)
    await waitFor(() => {
      expect(screen.getByTestId('badge-analyst-review-count')).toHaveTextContent('14');
      expect(screen.getByTestId('badge-high-risk-count')).toHaveTextContent('8');
    });

    // Default tab is Analyst Review
    expect(screen.getByTestId('tab-analyst-review')).toHaveClass('border-amber-400');
    expect(screen.getByText('tx-rev-1')).toBeInTheDocument();

    // Decision select is hidden in filter bar to prevent conflict
    expect(screen.queryByTestId('filter-decision-select')).not.toBeInTheDocument();
  });

  it('switches to High Risk tab and displays High Risk transactions', async () => {
    renderFlagged();

    await waitFor(() => {
      expect(screen.getByText('tx-rev-1')).toBeInTheDocument();
    });

    // Click High Risk tab
    const hrTab = screen.getByTestId('tab-high-risk');
    fireEvent.click(hrTab);

    // High risk items displayed
    await waitFor(() => {
      expect(screen.getByText('tx-hr-1')).toBeInTheDocument();
      expect(screen.getByTestId('tab-high-risk')).toHaveClass('border-rose-400');
    });
  });
});

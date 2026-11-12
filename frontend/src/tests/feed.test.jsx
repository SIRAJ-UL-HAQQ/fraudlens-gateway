import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DashboardFeed from '../pages/DashboardFeed';
import * as client from '../api/client';

function renderFeed(initialEntries = ['/']) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={initialEntries}>
        <DashboardFeed />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('DashboardFeed page (Condition 2 & Condition 4)', () => {
  const mockTransactions = [
    {
      transaction_id: 'tx-001',
      step: 10,
      type: 'TRANSFER',
      amount: 50000,
      name_orig: 'C11111111',
      name_dest: 'C22222222',
      risk_score: 0.88,
      gateway_decision: 'High Risk',
      explanation_status: 'Stable',
      rule_flags: ['LARGE_TRANSFER_AMOUNT'],
      reviewed: false,
      created_at: '2026-10-15T12:00:00Z',
    },
    {
      transaction_id: 'tx-002',
      step: 12,
      type: 'PAYMENT',
      amount: 120,
      name_orig: 'C33333333',
      name_dest: 'M44444444',
      risk_score: 0.02,
      gateway_decision: 'Low Risk',
      explanation_status: 'Stable',
      rule_flags: [],
      reviewed: true,
      created_at: '2026-10-15T11:50:00Z',
    },
  ];

  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(client, 'getTransactions').mockResolvedValue({
      items: mockTransactions,
      total: 2,
      page: 1,
      page_size: 50,
      total_pages: 1,
    });
  });

  it('renders heading, server-enforced audit order banner, and transaction rows', async () => {
    renderFeed();

    expect(screen.getByText('Transaction Feed & Live Triage')).toBeInTheDocument();

    // Condition 2: Server-enforced order disclaimer
    const banner = screen.getByTestId('order-disclaimer-banner');
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent(/Ordered by created_at descending \(server-enforced audit order\)/i);

    // Wait for table to populate
    await waitFor(() => {
      expect(screen.getByText('tx-001')).toBeInTheDocument();
      expect(screen.getByText('tx-002')).toBeInTheDocument();
    });

    // Accounts origin -> dest
    expect(screen.getByText('C11111111')).toBeInTheDocument();
    expect(screen.getByText('C22222222')).toBeInTheDocument();

    // Rule flags rendered
    expect(screen.getByText('LARGE_TRANSFER_AMOUNT')).toBeInTheDocument();
  });

  it('toggles polling between Live (2s) and Paused via user control button', async () => {
    renderFeed();

    const indicator = screen.getByTestId('polling-status-indicator');
    expect(indicator).toHaveTextContent('Live (2s)');

    const toggleBtn = screen.getByTestId('toggle-polling-btn');
    fireEvent.click(toggleBtn);

    expect(indicator).toHaveTextContent('Paused');

    fireEvent.click(toggleBtn);
    expect(indicator).toHaveTextContent('Live (2s)');
  });

  it('validates min_step <= max_step and displays error when invalid', async () => {
    renderFeed();

    const minInput = screen.getByTestId('filter-step-min-input');
    const maxInput = screen.getByTestId('filter-step-max-input');
    const applyBtn = screen.getByTestId('apply-filters-btn');

    // Enter min > max
    fireEvent.change(minInput, { target: { value: '50' } });
    fireEvent.change(maxInput, { target: { value: '10' } });
    fireEvent.click(applyBtn);

    expect(screen.getByTestId('filter-validation-error')).toHaveTextContent(
      'Min step cannot exceed Max step.'
    );
  });

  it('opens TransactionDetailDrawer when a transaction row is clicked', async () => {
    renderFeed();

    await waitFor(() => {
      expect(screen.getByTestId('transaction-row-tx-001')).toBeInTheDocument();
    });

    // Click row
    fireEvent.click(screen.getByTestId('transaction-row-tx-001'));

    // Drawer should open
    await waitFor(() => {
      expect(screen.getByTestId('transaction-detail-drawer')).toBeInTheDocument();
    });
  });
});

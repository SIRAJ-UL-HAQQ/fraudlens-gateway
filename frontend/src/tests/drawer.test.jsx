import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import TransactionDetailDrawer from '../components/transactions/TransactionDetailDrawer';

function renderWithClient(ui) {
  const testClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return render(<QueryClientProvider client={testClient}>{ui}</QueryClientProvider>);
}

describe('TransactionDetailDrawer component', () => {
  const baseTx = {
    transaction_id: 'tx-1234-abcd',
    step: 42,
    type: 'TRANSFER',
    amount: 150000.5,
    name_orig: 'C10000001',
    old_balance_orig: 150000.5,
    new_balance_orig: 0.0,
    name_dest: 'M20000002',
    old_balance_dest: 0.0,
    new_balance_dest: 150000.5,
    risk_score: 0.9542,
    gateway_decision: 'High Risk',
    explanation_status: 'Stable',
    rule_flags: ['ORIGIN_ACCOUNT_DRAINED'],
    model_version: 'v1.0.0-xgb',
    imbalance_strategy: 'smote_enn',
    latency_ms: 14.8,
    shap_features: [
      { feature: 'amount', shap_value: 0.25, direction: 'increases_risk' },
    ],
    reviews: [],
  };

  it('renders nothing when isOpen is false', () => {
    const { container } = renderWithClient(
      <TransactionDetailDrawer
        transactionId="tx-1234-abcd"
        isOpen={false}
        onClose={vi.fn()}
        initialData={baseTx}
      />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders transaction details, accessibility attributes, and raw PaySim fields when open', () => {
    const onClose = vi.fn();
    renderWithClient(
      <TransactionDetailDrawer
        transactionId="tx-1234-abcd"
        isOpen={true}
        onClose={onClose}
        initialData={baseTx}
      />
    );

    const dialog = screen.getByRole('dialog');
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-labelledby', 'drawer-title');

    expect(screen.getByText('tx-1234-abcd')).toBeInTheDocument();
    expect(screen.getByText('TRANSFER')).toBeInTheDocument();
    expect(screen.getByText('C10000001')).toBeInTheDocument();
    expect(screen.getByText('M20000002')).toBeInTheDocument();

    // Balances render accurately including $0.00 (Bug 2 Fix)
    const positiveAmounts = screen.getAllByText('$150,000.50');
    expect(positiveAmounts.length).toBe(3);
    const zeroBalances = screen.getAllByText('$0.00');
    expect(zeroBalances.length).toBe(2);

    // Model version and strategy badges render (Bug 4 Fix)
    expect(screen.getByText('v1.0.0-xgb')).toBeInTheDocument();
    expect(screen.getByText('smote_enn')).toBeInTheDocument();
    expect(screen.getByText(/Server-side scoring \+ explanation latency/i)).toBeInTheDocument();

    // Rule flags rendered unconditionally
    expect(screen.getByText('ORIGIN_ACCOUNT_DRAINED')).toBeInTheDocument();

    // Stable explanation does NOT show unstable warning banner
    expect(screen.queryByTestId('unstable-warning-banner')).not.toBeInTheDocument();
  });

  it('renders prominent unstable warning banner when explanation_status is Unstable (Condition 3)', () => {
    const unstableTx = {
      ...baseTx,
      gateway_decision: 'Analyst Review',
      explanation_status: 'Unstable',
    };

    renderWithClient(
      <TransactionDetailDrawer
        transactionId="tx-1234-abcd"
        isOpen={true}
        onClose={vi.fn()}
        initialData={unstableTx}
      />
    );

    const warningBanner = screen.getByTestId('unstable-warning-banner');
    expect(warningBanner).toBeInTheDocument();
    expect(
      screen.getByText(/Unstable Explanation Detected \(SRS Variant B Routing\)/i)
    ).toBeInTheDocument();
  });

  it('calls onClose when close button or backdrop is clicked, or Escape key is pressed', () => {
    const onClose = vi.fn();
    renderWithClient(
      <TransactionDetailDrawer
        transactionId="tx-1234-abcd"
        isOpen={true}
        onClose={onClose}
        initialData={baseTx}
      />
    );

    // Click close button
    const closeBtn = screen.getByTestId('drawer-close-btn');
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalledTimes(1);

    // Click backdrop
    const backdrop = screen.getByTestId('drawer-backdrop');
    fireEvent.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(2);

    // Press Escape
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(3);
  });
});

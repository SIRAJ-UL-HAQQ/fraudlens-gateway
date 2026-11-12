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
    oldbalance_org: 150000.5,
    newbalance_orig: 0.0,
    name_dest: 'M20000002',
    oldbalance_dest: 0.0,
    newbalance_dest: 150000.5,
    is_flagged_fraud: 1,
    risk_score: 0.9542,
    gateway_decision: 'High Risk',
    explanation_status: 'Stable',
    rule_flags: ['ORIGIN_ACCOUNT_DRAINED'],
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
    expect(screen.getByText('Yes (Flagged)')).toBeInTheDocument();

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

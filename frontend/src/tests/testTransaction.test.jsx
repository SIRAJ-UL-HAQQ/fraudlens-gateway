import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import * as client from '../api/client';
import TestTransactionView from '../pages/TestTransactionView';

describe('TestTransactionView Sandbox', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders initial form with fraud preset data and preset buttons', () => {
    render(<TestTransactionView />);
    expect(screen.getByRole('heading', { name: /test single transaction/i })).toBeInTheDocument();
    expect(screen.getByTestId('preset-button-fraud')).toBeInTheDocument();
    expect(screen.getByTestId('preset-button-legit')).toBeInTheDocument();
    expect(screen.getByTestId('preset-button-borderline')).toBeInTheDocument();
    expect(screen.getByTestId('preset-button-zero')).toBeInTheDocument();

    const amountInput = screen.getByLabelText(/amount/i);
    expect(amountInput).toHaveValue(181000);
  });

  it('switches form values when preset buttons are clicked', () => {
    render(<TestTransactionView />);
    const legitBtn = screen.getByTestId('preset-button-legit');
    fireEvent.click(legitBtn);

    const amountInput = screen.getByLabelText(/amount/i);
    expect(amountInput).toHaveValue(25.5);

    const typeSelect = screen.getByLabelText(/transaction type/i);
    expect(typeSelect).toHaveValue('PAYMENT');
  });

  it('validates boundary conditions and prevents submission on invalid input', async () => {
    const scoreSpy = vi.spyOn(client, 'scoreTransaction');
    render(<TestTransactionView />);

    const stepInput = screen.getByLabelText(/simulation step/i);
    fireEvent.change(stepInput, { target: { value: '0' } });

    const amountInput = screen.getByLabelText(/amount/i);
    fireEvent.change(amountInput, { target: { value: '-50' } });

    const submitBtn = screen.getByTestId('submit-transaction-button');
    fireEvent.click(submitBtn);

    expect(await screen.findByText('Step must be an integer >= 1')).toBeInTheDocument();
    expect(screen.getByText('Amount must be >= 0')).toBeInTheDocument();
    expect(scoreSpy).not.toHaveBeenCalled();
  });

  it('scores transaction and renders full prediction result card', async () => {
    const mockPrediction = {
      transaction_id: 'pred-test-12345',
      risk_score: 0.945,
      top_features: [
        { feature: 'amount', shap_value: 0.4821, direction: 'increases_risk' },
        { feature: 'type_TRANSFER', shap_value: 0.3512, direction: 'increases_risk' },
      ],
      explanation_status: 'Stable',
      gateway_decision: 'High Risk',
      rule_flags: ['ORIGIN_ACCOUNT_DRAINED', 'LARGE_TRANSFER_AMOUNT'],
      imbalance_strategy: 'class_weight',
      model_version: 'v1.0.0-xgb',
      latency_ms: 14.2,
    };

    vi.spyOn(client, 'scoreTransaction').mockResolvedValue(mockPrediction);

    render(<TestTransactionView />);
    const submitBtn = screen.getByTestId('submit-transaction-button');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByTestId('prediction-result-card')).toBeInTheDocument();
    });

    expect(screen.getByText('High Risk')).toBeInTheDocument();
    expect(screen.getByText('Stable')).toBeInTheDocument();
    expect(screen.getByText('ORIGIN_ACCOUNT_DRAINED')).toBeInTheDocument();
    expect(screen.getByText('LARGE_TRANSFER_AMOUNT')).toBeInTheDocument();
    expect(screen.getByText('14.2 ms')).toBeInTheDocument();
    expect(screen.getByText('v1.0.0-xgb')).toBeInTheDocument();
  });

  it('displays Unstable explanation warning banner when explanation_status is Unstable', async () => {
    const mockUnstablePrediction = {
      transaction_id: 'pred-ambig-999',
      risk_score: 0.725,
      top_features: [
        { feature: 'amount', shap_value: 0.2215, direction: 'increases_risk' },
      ],
      explanation_status: 'Unstable',
      gateway_decision: 'Analyst Review',
      rule_flags: [],
      imbalance_strategy: 'smote_enn',
      model_version: 'v1.0.0-xgb',
      latency_ms: 18.7,
    };

    vi.spyOn(client, 'scoreTransaction').mockResolvedValue(mockUnstablePrediction);

    render(<TestTransactionView />);
    fireEvent.click(screen.getByTestId('preset-button-borderline'));
    fireEvent.click(screen.getByTestId('submit-transaction-button'));

    await waitFor(() => {
      expect(screen.getByTestId('unstable-explanation-banner')).toBeInTheDocument();
    });
    expect(screen.getByText(/Unstable Explanation:/i)).toBeInTheDocument();
  });

  it('displays Stub Mode warning banner when model_version is stub', async () => {
    const mockStubPrediction = {
      transaction_id: 'pred-stub-111',
      risk_score: 0.5,
      top_features: [],
      explanation_status: 'Stable',
      gateway_decision: 'Low Risk',
      rule_flags: [],
      imbalance_strategy: 'none',
      model_version: 'stub',
      latency_ms: 2.1,
    };

    vi.spyOn(client, 'scoreTransaction').mockResolvedValue(mockStubPrediction);

    render(<TestTransactionView />);
    fireEvent.click(screen.getByTestId('submit-transaction-button'));

    await waitFor(() => {
      expect(screen.getByTestId('stub-mode-banner')).toBeInTheDocument();
    });
    expect(screen.getByText(/Stub Mode Active:/i)).toBeInTheDocument();
  });

  it('displays API error alert when scoreTransaction fails', async () => {
    vi.spyOn(client, 'scoreTransaction').mockRejectedValue(new Error('Internal scoring engine timeout'));

    render(<TestTransactionView />);
    fireEvent.click(screen.getByTestId('submit-transaction-button'));

    await waitFor(() => {
      expect(screen.getByText('Internal scoring engine timeout')).toBeInTheDocument();
    });
  });
});

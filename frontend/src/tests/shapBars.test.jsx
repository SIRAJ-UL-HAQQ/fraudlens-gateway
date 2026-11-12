import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ShapFeatureBars from '../components/shared/ShapFeatureBars';

describe('ShapFeatureBars component (Condition 3)', () => {
  it('renders fallback when feature list is empty or undefined', () => {
    render(<ShapFeatureBars features={[]} />);
    expect(screen.getByText(/No feature explanations available/i)).toBeInTheDocument();
    expect(
      screen.getByText(/SHAP values measure local feature attribution for this model prediction, not causal proof/i)
    ).toBeInTheDocument();
  });

  it('safely handles zero-value attributions without NaN% division-by-zero errors', () => {
    const zeroFeatures = [
      { feature: 'oldbalanceOrg', shap_value: 0, direction: 'decreases_risk' },
      { feature: 'newbalanceOrig', shap_value: 0, direction: 'increases_risk' },
    ];
    render(<ShapFeatureBars features={zeroFeatures} />);

    const progressbars = screen.getAllByRole('progressbar');
    expect(progressbars.length).toBe(2);
    progressbars.forEach((bar) => {
      expect(bar.style.width).toBe('0%');
      expect(bar.style.width).not.toContain('NaN');
    });
  });

  it('renders explicit signs and text labels for both directions (never colour alone)', () => {
    const features = [
      { feature: 'amount', shap_value: 0.1842, direction: 'increases_risk' },
      { feature: 'oldbalanceOrg', shap_value: -0.2105, direction: 'decreases_risk' },
    ];

    render(<ShapFeatureBars features={features} />);

    // Check signed string formatting
    expect(screen.getByText('+0.1842')).toBeInTheDocument();
    expect(screen.getByText('-0.2105')).toBeInTheDocument();

    // Check explicit textual badge labels (Condition 3)
    expect(screen.getByText('Pushes toward Fraud')).toBeInTheDocument();
    expect(screen.getByText('Pushes toward Legitimate')).toBeInTheDocument();

    // Check causality disclaimer
    expect(
      screen.getByText(/SHAP values measure local feature attribution for this model prediction, not causal proof/i)
    ).toBeInTheDocument();
  });

  it('scales bar widths relative to max |SHAP| correctly', () => {
    const features = [
      { feature: 'amount', shap_value: 0.1, direction: 'increases_risk' }, // 0.1 / 0.2 = 50%
      { feature: 'oldbalanceOrg', shap_value: -0.2, direction: 'decreases_risk' }, // 0.2 / 0.2 = 100%
    ];

    render(<ShapFeatureBars features={features} />);

    const progressbars = screen.getAllByRole('progressbar');
    expect(progressbars[0].style.width).toBe('50%');
    expect(progressbars[1].style.width).toBe('100%');
  });
});

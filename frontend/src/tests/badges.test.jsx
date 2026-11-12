import React from 'react';
import fs from 'fs';
import path from 'path';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import DecisionBadge from '../components/shared/DecisionBadge';
import ReliabilityBadge from '../components/shared/ReliabilityBadge';

// Mandatory Condition 3: Read docs/openapi.json via Node fs inside test only
const openApiPath = path.resolve(__dirname, '../../../docs/openapi.json');
const openApiSpec = JSON.parse(fs.readFileSync(openApiPath, 'utf-8'));
const gatewayDecisionEnums = openApiSpec.components.schemas.GatewayDecision.enum;
const explanationStatusEnums = openApiSpec.components.schemas.ExplanationStatus.enum;

describe('DecisionBadge', () => {
  it('covers all OpenAPI GatewayDecision enum values', () => {
    expect(gatewayDecisionEnums).toEqual(['Low Risk', 'High Risk', 'Analyst Review']);

    gatewayDecisionEnums.forEach((decision) => {
      const { unmount } = render(<DecisionBadge decision={decision} />);
      expect(screen.getByText(decision)).toBeInTheDocument();

      // Condition 3: Never use a check mark for Low Risk or anywhere near badges
      if (decision === 'Low Risk') {
        const badge = screen.getByTestId('decision-badge');
        expect(badge.innerHTML).not.toContain('check');
        expect(badge.className).toContain('text-emerald-400');
      }
      unmount();
    });
  });

  it('renders unknown fallback for non-contract decision values', () => {
    render(<DecisionBadge decision="UnrecognizedDecision" />);
    expect(screen.getByText('UnrecognizedDecision')).toBeInTheDocument();
  });
});

describe('ReliabilityBadge', () => {
  it('covers all OpenAPI ExplanationStatus enum values', () => {
    expect(explanationStatusEnums).toEqual(['Stable', 'Unstable']);

    explanationStatusEnums.forEach((status) => {
      const { unmount } = render(<ReliabilityBadge status={status} />);
      expect(screen.getByText(status)).toBeInTheDocument();

      // Condition 3: Never use a check mark for Stable
      if (status === 'Stable') {
        const badge = screen.getByTestId('reliability-badge');
        expect(badge.innerHTML).not.toContain('check');
        expect(badge.className).toContain('text-emerald-400');
      }
      unmount();
    });
  });

  it('renders unknown fallback for non-contract reliability status', () => {
    render(<ReliabilityBadge status="UnknownStatus" />);
    expect(screen.getByText('UnknownStatus')).toBeInTheDocument();
  });

  it('toggles accessible tooltip and closes on Escape key', () => {
    render(<ReliabilityBadge status="Stable" />);

    const triggerButton = screen.getByRole('button', {
      name: /explanation reliability details/i,
    });
    expect(triggerButton).toBeInTheDocument();

    // Tooltip initially closed
    expect(
      screen.queryByText(/SHAP is an attribution, not proof of cause/i)
    ).not.toBeInTheDocument();

    // Click trigger opens tooltip
    fireEvent.click(triggerButton);
    expect(
      screen.getByText(/SHAP is an attribution, not proof of cause/i)
    ).toBeInTheDocument();
    expect(triggerButton).toHaveAttribute('aria-describedby');

    // Pressing Escape closes tooltip
    fireEvent.keyDown(document, { key: 'Escape', code: 'Escape' });
    expect(
      screen.queryByText(/SHAP is an attribution, not proof of cause/i)
    ).not.toBeInTheDocument();
  });
});

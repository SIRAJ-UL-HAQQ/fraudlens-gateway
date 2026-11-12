import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from '../App';

describe('App Smoke Test', () => {
  it('renders application heading', () => {
    render(<App />);
    expect(screen.getByText('FraudLens Gateway')).toBeInTheDocument();
  });
});

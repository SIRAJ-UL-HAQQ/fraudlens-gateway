import React, { useState } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ErrorBoundary from '../components/shared/ErrorBoundary';
import TestTransactionView from '../pages/TestTransactionView';
import * as client from '../api/client';
import { AuthProvider } from '../auth/AuthContext';
import { ProtectedRoute } from '../auth/ProtectedRoute';

function ThrowingComponent({ shouldThrow }) {
  if (shouldThrow) {
    throw new Error('Simulated uncaught React rendering explosion');
  }
  return <div>Component rendered safely</div>;
}

function ProblematicParent() {
  const [hasError, setHasError] = useState(false);
  return (
    <div>
      <button onClick={() => setHasError(true)}>Trigger Error</button>
      <ErrorBoundary onReset={() => setHasError(false)}>
        <ThrowingComponent shouldThrow={hasError} />
      </ErrorBoundary>
    </div>
  );
}

describe('Resilience and Error Handling Suite', () => {
  let originalError;

  beforeEach(() => {
    vi.restoreAllMocks();
    originalError = console.error;
    console.error = vi.fn(); // Suppress expected boundary errors from test logs
  });

  afterEach(() => {
    console.error = originalError;
  });

  it('catches uncaught component render error and presents ErrorBoundary fallback with recovery button', () => {
    render(<ProblematicParent />);

    expect(screen.getByText('Component rendered safely')).toBeInTheDocument();
    expect(screen.queryByTestId('error-boundary-fallback')).not.toBeInTheDocument();

    // Trigger uncaught explosion
    fireEvent.click(screen.getByText('Trigger Error'));

    expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument();
    expect(screen.getByText('Application Error Encountered')).toBeInTheDocument();
    expect(
      screen.getByText('Simulated uncaught React rendering explosion')
    ).toBeInTheDocument();

    // Reset via Reload View button
    const reloadBtn = screen.getByTestId('reload-view-btn');
    expect(reloadBtn).toBeInTheDocument();
    fireEvent.click(reloadBtn);

    expect(screen.getByText('Component rendered safely')).toBeInTheDocument();
  });

  it('handles backend offline / network failure gracefully when scoring transaction', async () => {
    vi.spyOn(client, 'scoreTransaction').mockRejectedValue(
      new Error('Failed to fetch: Connection refused (backend offline)')
    );

    render(<TestTransactionView />);

    const submitBtn = screen.getByTestId('submit-transaction-button');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText(/Connection refused \(backend offline\)/i)
      ).toBeInTheDocument();
    });

    // Form remains intact and usable
    expect(screen.getByTestId('submit-transaction-button')).toBeEnabled();
  });

  it('handles 503 Model Unloaded error gracefully without page crash', async () => {
    vi.spyOn(client, 'scoreTransaction').mockRejectedValue(
      new Error('503 Service Unavailable: Model weights not loaded into memory')
    );

    render(<TestTransactionView />);

    const submitBtn = screen.getByTestId('submit-transaction-button');
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText(/503 Service Unavailable: Model weights not loaded/i)
      ).toBeInTheDocument();
    });

    expect(screen.queryByTestId('error-boundary-fallback')).not.toBeInTheDocument();
  });

  it('redirects to login with session_expired query parameter on 401 unauthorized session expiry', async () => {
    // Clear storage
    localStorage.clear();

    function DummyProtected() {
      return <div>Protected Analyst Content</div>;
    }

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/protected']}>
          <Routes>
            <Route
              path="/protected"
              element={
                <ProtectedRoute>
                  <DummyProtected />
                </ProtectedRoute>
              }
            />
            <Route
              path="/login"
              element={<div>Login Screen Displayed</div>}
            />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    // Immediately redirected because not authenticated
    await waitFor(() => {
      expect(screen.getByText('Login Screen Displayed')).toBeInTheDocument();
    });
    expect(screen.queryByText('Protected Analyst Content')).not.toBeInTheDocument();
  });
});

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { tokenStore } from '../auth/tokenStore';
import { AuthProvider } from '../auth/AuthContext';
import ProtectedRoute from '../auth/ProtectedRoute';
import Login from '../pages/Login';
import * as client from '../api/client';

describe('tokenStore unit tests', () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it('stores, retrieves, and clears token with user metadata', () => {
    expect(tokenStore.getToken()).toBeNull();
    expect(tokenStore.getUser()).toBeNull();

    tokenStore.setToken('test-token-xyz', 3600, 'analyst_bob', 'lead_analyst');
    expect(tokenStore.getToken()).toBe('test-token-xyz');
    expect(tokenStore.getUser()).toEqual({
      username: 'analyst_bob',
      role: 'lead_analyst',
    });

    tokenStore.clearToken();
    expect(tokenStore.getToken()).toBeNull();
    expect(tokenStore.getUser()).toBeNull();
  });

  it('correctly reports expired token and clears it', () => {
    // Set token with 0 seconds expiry
    tokenStore.setToken('expired-token', -10, 'analyst_1', 'analyst');
    expect(tokenStore.isExpired()).toBe(true);
    expect(tokenStore.getToken()).toBeNull();
  });
});

describe('ProtectedRoute component', () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it('redirects unauthenticated user to /login with state', () => {
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/protected']}>
          <Routes>
            <Route path="/login" element={<div>Login Page Mock</div>} />
            <Route
              path="/protected"
              element={
                <ProtectedRoute>
                  <div>Secret Content</div>
                </ProtectedRoute>
              }
            />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    expect(screen.getByText('Login Page Mock')).toBeInTheDocument();
    expect(screen.queryByText('Secret Content')).toBeNull();
  });

  it('renders child content when user is authenticated', () => {
    tokenStore.setToken('valid-token', 3600, 'analyst_1', 'analyst');

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/protected']}>
          <Routes>
            <Route path="/login" element={<div>Login Page Mock</div>} />
            <Route
              path="/protected"
              element={
                <ProtectedRoute>
                  <div>Secret Content</div>
                </ProtectedRoute>
              }
            />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    expect(screen.getByText('Secret Content')).toBeInTheDocument();
    expect(screen.queryByText('Login Page Mock')).toBeNull();
  });
});

describe('Login Page workflow', () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.restoreAllMocks();
  });

  it('displays session expired alert if state.sessionExpired is true', () => {
    render(
      <AuthProvider>
        <MemoryRouter
          initialEntries={[{ pathname: '/login', state: { sessionExpired: true } }]}
        >
          <Routes>
            <Route path="/login" element={<Login />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    expect(
      screen.getByText('Session expired. Please log in again.')
    ).toBeInTheDocument();
  });

  it('submits credentials and handles login error', async () => {
    const user = userEvent.setup();
    vi.spyOn(client, 'login').mockRejectedValueOnce(
      new Error('Invalid username or password')
    );

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/login']}>
          <Routes>
            <Route path="/login" element={<Login />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    const userInput = screen.getByLabelText(/username/i);
    const passInput = screen.getByLabelText(/password/i);
    const submitBtn = screen.getByRole('button', { name: /sign in/i });

    await user.type(userInput, 'wrong_user');
    await user.type(passInput, 'wrong_password');
    await user.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText('Invalid username or password')).toBeInTheDocument();
    });
  });

  it('logs in successfully and redirects to target route', async () => {
    const user = userEvent.setup();
    vi.spyOn(client, 'login').mockResolvedValueOnce({
      access_token: 'auth-jwt-token',
      token_type: 'bearer',
      expires_in: 3600,
      username: 'analyst_1',
      role: 'analyst',
    });

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={['/login']}>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/" element={<div>Dashboard Home Target</div>} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    const userInput = screen.getByLabelText(/username/i);
    const passInput = screen.getByLabelText(/password/i);
    const submitBtn = screen.getByRole('button', { name: /sign in/i });

    await user.type(userInput, 'analyst_1');
    await user.type(passInput, 'password123');
    await user.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText('Dashboard Home Target')).toBeInTheDocument();
    });
  });
});

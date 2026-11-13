import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import StatusBanner from '../components/layout/StatusBanner';
import * as healthHook from '../hooks/useHealthStatus';

describe('StatusBanner component', () => {
  it('renders nothing when gateway is healthy and model is active', () => {
    vi.spyOn(healthHook, 'useHealthStatus').mockReturnValue({
      isStub: false,
      isUnloaded: false,
      isDbDisconnected: false,
      isUnreachable: false,
      healthData: { status: 'healthy', model: { mode: 'active' }, database: 'connected' },
    });

    const { container } = render(<StatusBanner />);
    expect(container.firstChild).toBeNull();
  });

  it('renders role="status" banner for stub mode', () => {
    vi.spyOn(healthHook, 'useHealthStatus').mockReturnValue({
      isStub: true,
      isUnloaded: false,
      isDbDisconnected: false,
      isUnreachable: false,
      healthData: { status: 'degraded', model: { mode: 'stub' }, database: 'connected' },
    });

    render(<StatusBanner />);
    const banner = screen.getByRole('status');
    expect(banner).toBeInTheDocument();
    expect(
      screen.getByText(/Gateway running in STUB MODE/i)
    ).toBeInTheDocument();
  });

  it('renders role="alert" banner for model unloaded state', () => {
    vi.spyOn(healthHook, 'useHealthStatus').mockReturnValue({
      isStub: false,
      isUnloaded: true,
      isDbDisconnected: false,
      isUnreachable: false,
      healthData: { status: 'degraded', model: { mode: 'unloaded' }, database: 'connected' },
    });

    render(<StatusBanner />);
    const banner = screen.getByRole('alert');
    expect(banner).toBeInTheDocument();
    expect(
      screen.getByText(/Model unloaded \/ Gateway not ready/i)
    ).toBeInTheDocument();
  });

  it('renders role="alert" banner for database disconnected state', () => {
    vi.spyOn(healthHook, 'useHealthStatus').mockReturnValue({
      isStub: false,
      isUnloaded: false,
      isDbDisconnected: true,
      isUnreachable: false,
      healthData: { status: 'degraded', model: { mode: 'active' }, database: 'disconnected' },
    });

    render(<StatusBanner />);
    const banner = screen.getByRole('alert');
    expect(banner).toBeInTheDocument();
    expect(
      screen.getByText(/Database disconnected \/ Gateway degraded/i)
    ).toBeInTheDocument();
  });

  it('renders role="alert" banner when gateway is unreachable', () => {
    vi.spyOn(healthHook, 'useHealthStatus').mockReturnValue({
      isStub: false,
      isUnloaded: false,
      isDbDisconnected: false,
      isUnreachable: true,
      healthData: null,
    });

    render(<StatusBanner />);
    const banner = screen.getByRole('alert');
    expect(banner).toBeInTheDocument();
    expect(screen.getByText(/Gateway unreachable/i)).toBeInTheDocument();
  });
});

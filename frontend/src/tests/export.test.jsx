import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import * as client from '../api/client';
import ExportView from '../pages/ExportView';

describe('ExportView Page', () => {
  let clickSpy;

  beforeEach(() => {
    vi.restoreAllMocks();
    window.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    window.URL.revokeObjectURL = vi.fn();
    clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  });

  afterEach(() => {
    clickSpy?.mockRestore();
    vi.restoreAllMocks();
  });

  it('renders default state with CSV selected and filter inputs', () => {
    render(<ExportView />);
    expect(screen.getByRole('heading', { name: /export audit logs/i })).toBeInTheDocument();
    expect(screen.getByTestId('format-option-csv')).toHaveTextContent('Selected');
    expect(screen.getByTestId('download-export-button')).toHaveTextContent('Download CSV Audit Log');
  });

  it('switches format between CSV and PDF', () => {
    render(<ExportView />);
    const pdfOption = screen.getByTestId('format-option-pdf');
    fireEvent.click(pdfOption);

    expect(screen.getByTestId('format-option-pdf')).toHaveTextContent('Selected');
    expect(screen.getByTestId('download-export-button')).toHaveTextContent('Download PDF Audit Log');
  });

  it('validates step boundary rules and invalid ranges', async () => {
    const exportSpy = vi.spyOn(client, 'exportTransactions');
    render(<ExportView />);

    const stepMinInput = screen.getByLabelText(/minimum step filter/i);
    const stepMaxInput = screen.getByLabelText(/maximum step filter/i);

    // Invalid minimum step < 1
    fireEvent.change(stepMinInput, { target: { value: '0' } });
    fireEvent.click(screen.getByTestId('download-export-button'));

    expect(await screen.findByText('Step min must be an integer >= 1')).toBeInTheDocument();
    expect(exportSpy).not.toHaveBeenCalled();

    // Step min > Step max
    fireEvent.change(stepMinInput, { target: { value: '20' } });
    fireEvent.change(stepMaxInput, { target: { value: '10' } });
    fireEvent.click(screen.getByTestId('download-export-button'));

    expect(await screen.findByText('Step min cannot exceed Step max')).toBeInTheDocument();
    expect(exportSpy).not.toHaveBeenCalled();
  });

  it('submits valid export request and triggers file download', async () => {
    const mockBlob = new Blob(['sample-content'], { type: 'application/pdf' });
    vi.spyOn(client, 'exportTransactions').mockResolvedValue({
      blob: mockBlob,
      filename: 'custom_export_2026.pdf',
    });

    render(<ExportView />);
    fireEvent.click(screen.getByTestId('format-option-pdf'));

    const decisionSelect = screen.getByLabelText(/gateway decision/i);
    fireEvent.change(decisionSelect, { target: { value: 'High Risk' } });

    const stepMinInput = screen.getByLabelText(/minimum step filter/i);
    fireEvent.change(stepMinInput, { target: { value: '5' } });

    fireEvent.click(screen.getByTestId('download-export-button'));

    await waitFor(() => {
      expect(client.exportTransactions).toHaveBeenCalledWith({
        format: 'pdf',
        decision: 'High Risk',
        step_min: 5,
      });
    });

    expect(await screen.findByTestId('export-success-banner')).toBeInTheDocument();
    expect(screen.getByText('custom_export_2026.pdf')).toBeInTheDocument();
  });

  it('displays safety cap error banner on HTTP 400 with server message', async () => {
    const error400 = new Error('Export matching 5200 records exceeds safety cap of 5000 rows. Please narrow your query filters.');
    error400.status = 400;
    vi.spyOn(client, 'exportTransactions').mockRejectedValue(error400);

    render(<ExportView />);
    fireEvent.click(screen.getByTestId('download-export-button'));

    await waitFor(() => {
      expect(screen.getByTestId('export-error-banner')).toBeInTheDocument();
    });

    expect(screen.getByText('Export Row Limit Exceeded')).toBeInTheDocument();
    expect(screen.getByText(/exceeds safety cap of 5000 rows/i)).toBeInTheDocument();
  });

  it('displays service unavailable banner on HTTP 503', async () => {
    const error503 = new Error('Database connection pool timeout');
    error503.status = 503;
    vi.spyOn(client, 'exportTransactions').mockRejectedValue(error503);

    render(<ExportView />);
    fireEvent.click(screen.getByTestId('download-export-button'));

    await waitFor(() => {
      expect(screen.getByTestId('export-error-banner')).toBeInTheDocument();
    });

    expect(screen.getByText('Database Service Unavailable')).toBeInTheDocument();
    expect(screen.getByText(/database is currently unreachable/i)).toBeInTheDocument();
  });

  it('resets filters and clears messages on Reset Filters click', async () => {
    render(<ExportView />);
    const stepMinInput = screen.getByLabelText(/minimum step filter/i);
    fireEvent.change(stepMinInput, { target: { value: '50' } });
    expect(stepMinInput).toHaveValue(50);

    fireEvent.click(screen.getByRole('button', { name: /reset filters/i }));
    expect(stepMinInput).toHaveValue(null);
  });
});

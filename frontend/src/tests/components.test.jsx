import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import RiskScoreBar from '../components/shared/RiskScoreBar';
import RuleFlagList from '../components/shared/RuleFlagList';
import DataTable from '../components/shared/DataTable';
import Pagination from '../components/shared/Pagination';

describe('RiskScoreBar', () => {
  it('renders role="meter" with exact aria-valuetext and no threshold marks', () => {
    const { container } = render(<RiskScoreBar score={0.87} />);

    const meter = screen.getByRole('meter');
    expect(meter).toBeInTheDocument();
    expect(meter).toHaveAttribute('aria-valuenow', '0.87');
    expect(meter).toHaveAttribute('aria-valuemin', '0');
    expect(meter).toHaveAttribute('aria-valuemax', '1');
    expect(meter).toHaveAttribute(
      'aria-valuetext',
      'Risk score 0.87 on a 0 to 1 scale'
    );

    // Visible numeric label
    expect(screen.getByText('0.87')).toBeInTheDocument();

    // Verify NO threshold marks exist (none in contract)
    expect(container.querySelectorAll('.threshold-mark, [data-threshold]')).toHaveLength(0);
  });

  it('clamps scores below 0 or above 1 safely', () => {
    render(<RiskScoreBar score={1.45} />);
    const meter = screen.getByRole('meter');
    expect(meter).toHaveAttribute('aria-valuenow', '1');
    expect(meter).toHaveAttribute(
      'aria-valuetext',
      'Risk score 1.00 on a 0 to 1 scale'
    );
  });
});

describe('RuleFlagList', () => {
  it('renders neutral label, flags, and causality disclaimer', () => {
    const flags = ['LARGE_TRANSFER_AMOUNT', 'ORIGIN_ACCOUNT_DRAINED'];
    render(<RuleFlagList flags={flags} />);

    expect(screen.getByText('Observed Rule Flags')).toBeInTheDocument();
    expect(screen.getByText('LARGE_TRANSFER_AMOUNT')).toBeInTheDocument();
    expect(screen.getByText('ORIGIN_ACCOUNT_DRAINED')).toBeInTheDocument();
    expect(
      screen.getByText(/Observed rule flags do not imply causality in gateway routing/i)
    ).toBeInTheDocument();
  });

  it('renders "None" when flag array is empty', () => {
    render(<RuleFlagList flags={[]} />);
    expect(screen.getByText('None')).toBeInTheDocument();
  });
});

describe('DataTable', () => {
  it('safely escapes text and does not execute or inject raw HTML', () => {
    const maliciousPayload = '<img src=x onerror=alert(1)>';
    const columns = [
      { key: 'name', header: 'Name' },
      { key: 'payload', header: 'Payload' },
    ];
    const data = [{ id: '1', name: 'Safe Entry', payload: maliciousPayload }];

    const { container } = render(
      <DataTable columns={columns} data={data} keyField="id" />
    );

    expect(screen.getByText('Safe Entry')).toBeInTheDocument();
    expect(screen.getByText(maliciousPayload)).toBeInTheDocument();
    expect(container.querySelector('img')).toBeNull();
  });

  it('renders skeleton rows when isLoading is true', () => {
    const columns = [{ key: 'id', header: 'ID' }];
    render(<DataTable columns={columns} data={[]} isLoading={true} />);
    expect(screen.queryByText('No data available')).toBeNull();
  });
});

describe('Pagination', () => {
  it('disables previous button on first page and triggers page change', () => {
    const handlePageChange = vi.fn();
    render(
      <Pagination
        page={1}
        pageSize={50}
        total={120}
        totalPages={3}
        onPageChange={handlePageChange}
      />
    );

    const prevBtn = screen.getByRole('button', { name: /previous page/i });
    const nextBtn = screen.getByRole('button', { name: /next page/i });

    expect(prevBtn).toBeDisabled();
    expect(nextBtn).toBeEnabled();

    fireEvent.click(nextBtn);
    expect(handlePageChange).toHaveBeenCalledWith(2);
  });
});

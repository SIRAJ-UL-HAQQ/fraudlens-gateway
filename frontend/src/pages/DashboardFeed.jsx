import React, { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Activity, Play, Pause, RefreshCw, AlertCircle, Info } from 'lucide-react';
import { getTransactions } from '../api/client';
import useTransactionPolling from '../hooks/useTransactionPolling';
import TransactionFilterBar from '../components/transactions/TransactionFilterBar';
import TransactionTable from '../components/transactions/TransactionTable';
import TransactionDetailDrawer from '../components/transactions/TransactionDetailDrawer';
import Pagination from '../components/shared/Pagination';

export function DashboardFeed() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedTransactionId, setSelectedTransactionId] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  // Parse search params for API query
  const page = parseInt(searchParams.get('page') || '1', 10) || 1;
  const pageSize = parseInt(searchParams.get('page_size') || '50', 10) || 50;
  const decision = searchParams.get('decision') || undefined;
  const explanationStatus = searchParams.get('explanation_status') || undefined;
  const stepMin = searchParams.get('step_min') ? Number(searchParams.get('step_min')) : undefined;
  const stepMax = searchParams.get('step_max') ? Number(searchParams.get('step_max')) : undefined;
  const createdFrom = searchParams.get('created_from') || undefined;
  const createdTo = searchParams.get('created_to') || undefined;

  // 2-second real-time polling hook with tab visibility check
  const {
    isPollingEnabled,
    togglePolling,
    isTabVisible,
    pollingInterval,
  } = useTransactionPolling(true, 2000);

  // TanStack Query for transactions list
  const queryParams = {
    page,
    page_size: pageSize,
    decision,
    explanation_status: explanationStatus,
    step_min: stepMin,
    step_max: stepMax,
    created_from: createdFrom,
    created_to: createdTo,
  };

  const {
    data,
    isLoading,
    error,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ['transactions', queryParams],
    queryFn: () => getTransactions(queryParams),
    refetchInterval: pollingInterval,
  });

  const transactions = data?.items || [];
  const total = data?.total || 0;
  const totalPages = data?.total_pages || 1;

  const handleRowClick = (txId) => {
    setSelectedTransactionId(txId);
    setIsDrawerOpen(true);
  };

  const handleCloseDrawer = () => {
    setIsDrawerOpen(false);
  };

  const handlePageChange = (newPage) => {
    const updated = new URLSearchParams(searchParams);
    updated.set('page', String(newPage));
    setSearchParams(updated);
  };

  const handlePageSizeChange = (newSize) => {
    const updated = new URLSearchParams(searchParams);
    updated.set('page_size', String(newSize));
    updated.set('page', '1');
    setSearchParams(updated);
  };

  return (
    <div className="space-y-6" data-testid="dashboard-feed-page">
      {/* Page Title & Real-time Polling Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <Activity className="w-5 h-5 text-emerald-400" aria-hidden="true" />
            <span>Transaction Feed & Live Triage</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time PaySim transaction ingestion, SHAP attribution stability, and gateway routing stream.
          </p>
        </div>

        {/* Polling & Refresh Controls */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300">
            <span
              className={`w-2 h-2 rounded-full ${
                isPollingEnabled && isTabVisible
                  ? 'bg-emerald-400 animate-pulse'
                  : 'bg-amber-400'
              }`}
              aria-hidden="true"
            />
            <span className="font-mono text-[11px]" data-testid="polling-status-indicator">
              {!isTabVisible
                ? 'Paused (tab hidden)'
                : isPollingEnabled
                ? 'Live (2s)'
                : 'Paused'}
            </span>
            <button
              type="button"
              onClick={togglePolling}
              className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors ml-1 cursor-pointer"
              aria-label={isPollingEnabled ? 'Pause real-time updates' : 'Resume real-time updates'}
              title={isPollingEnabled ? 'Pause polling' : 'Resume polling'}
              data-testid="toggle-polling-btn"
            >
              {isPollingEnabled ? (
                <Pause className="w-3.5 h-3.5" aria-hidden="true" />
              ) : (
                <Play className="w-3.5 h-3.5" aria-hidden="true" />
              )}
            </button>
          </div>

          <button
            type="button"
            onClick={() => refetch()}
            disabled={isFetching}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 text-xs font-medium transition-colors disabled:opacity-50 cursor-pointer"
            aria-label="Refresh feed"
            data-testid="refresh-feed-btn"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin text-emerald-400' : ''}`}
              aria-hidden="true"
            />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Server Ordering Disclaimer Banner (Condition 2) */}
      <div
        className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 text-xs text-slate-400 flex items-center gap-2"
        data-testid="order-disclaimer-banner"
      >
        <Info className="w-4 h-4 text-slate-500 shrink-0" aria-hidden="true" />
        <span>Ordered by <code className="text-slate-300">created_at</code> descending (server-enforced audit order).</span>
      </div>

      {/* Filter Bar */}
      <TransactionFilterBar />

      {/* Error state */}
      {error && (
        <div
          role="alert"
          className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-3"
          data-testid="feed-error-banner"
        >
          <AlertCircle className="w-5 h-5 shrink-0" aria-hidden="true" />
          <div>
            <div className="font-semibold">Unable to fetch transactions</div>
            <div>{error.message || 'Network error occurred while contacting gateway server.'}</div>
          </div>
        </div>
      )}

      {/* Table */}
      <TransactionTable
        transactions={transactions}
        isLoading={isLoading}
        onRowClick={handleRowClick}
      />

      {/* Pagination */}
      <Pagination
        page={page}
        pageSize={pageSize}
        total={total}
        totalPages={totalPages}
        onPageChange={handlePageChange}
        onPageSizeChange={handlePageSizeChange}
      />

      {/* Transaction Detail Drawer */}
      <TransactionDetailDrawer
        transactionId={selectedTransactionId}
        isOpen={isDrawerOpen}
        onClose={handleCloseDrawer}
        initialData={transactions.find((t) => t.transaction_id === selectedTransactionId)}
      />
    </div>
  );
}

export default DashboardFeed;

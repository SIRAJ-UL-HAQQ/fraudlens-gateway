import React, { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { AlertTriangle, Clock, ShieldAlert, Info, RefreshCw } from 'lucide-react';
import { getTransactions } from '../api/client';
import TransactionFilterBar from '../components/transactions/TransactionFilterBar';
import TransactionTable from '../components/transactions/TransactionTable';
import TransactionDetailDrawer from '../components/transactions/TransactionDetailDrawer';
import Pagination from '../components/shared/Pagination';

export function FlaggedQueue() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedTransactionId, setSelectedTransactionId] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  // Active tab synchronized with URL search params (default: 'Analyst Review')
  const currentDecisionParam = searchParams.get('decision');
  const activeTab = currentDecisionParam === 'High Risk' ? 'High Risk' : 'Analyst Review';

  // Parse remaining filters
  const page = parseInt(searchParams.get('page') || '1', 10) || 1;
  const pageSize = parseInt(searchParams.get('page_size') || '50', 10) || 50;
  const explanationStatus = searchParams.get('explanation_status') || undefined;
  const stepMin = searchParams.get('step_min') ? Number(searchParams.get('step_min')) : undefined;
  const stepMax = searchParams.get('step_max') ? Number(searchParams.get('step_max')) : undefined;
  const createdFrom = searchParams.get('created_from') || undefined;
  const createdTo = searchParams.get('created_to') || undefined;

  // Honest count query for 'Analyst Review' tab
  const { data: analystReviewCountData } = useQuery({
    queryKey: ['transactions-count', 'Analyst Review'],
    queryFn: () => getTransactions({ decision: 'Analyst Review', page_size: 1 }),
  });

  // Honest count query for 'High Risk' tab
  const { data: highRiskCountData } = useQuery({
    queryKey: ['transactions-count', 'High Risk'],
    queryFn: () => getTransactions({ decision: 'High Risk', page_size: 1 }),
  });

  const countAnalystReview = analystReviewCountData?.total;
  const countHighRisk = highRiskCountData?.total;

  // Query for current tab's transactions
  const queryParams = {
    page,
    page_size: pageSize,
    decision: activeTab,
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
  });

  const transactions = data?.items || [];
  const total = data?.total || 0;
  const totalPages = data?.total_pages || 1;

  const handleTabChange = (decision) => {
    const updated = new URLSearchParams(searchParams);
    updated.set('decision', decision);
    updated.set('page', '1');
    setSearchParams(updated);
  };

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
    <div className="space-y-6" data-testid="flagged-queue-page">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2.5">
            <AlertTriangle className="w-5 h-5 text-amber-400" aria-hidden="true" />
            <span>Flagged Transaction Queue</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Priority adjudication queue for High Risk transactions and Unstable explanation routings.
          </p>
        </div>

        <button
          type="button"
          onClick={() => refetch()}
          disabled={isFetching}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 text-xs font-medium transition-colors disabled:opacity-50 cursor-pointer self-start sm:self-auto"
          aria-label="Refresh flagged queue"
          data-testid="refresh-flagged-btn"
        >
          <RefreshCw
            className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin text-amber-400' : ''}`}
            aria-hidden="true"
          />
          <span>Refresh Queue</span>
        </button>
      </div>

      {/* Variant B Routing Educational Banner (Condition 2) */}
      <div
        className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs space-y-1.5"
        data-testid="variant-b-banner"
      >
        <div className="flex items-center gap-2 font-semibold text-amber-200">
          <Info className="w-4 h-4 shrink-0 text-amber-400" aria-hidden="true" />
          <span>SRS Variant B Gateway Routing Logic</span>
        </div>
        <p className="text-[11px] text-amber-300/90 leading-relaxed">
          Transactions scored as <strong>High Risk</strong> whose local SHAP explanation is <strong>Unstable</strong> are
          automatically diverted from automated blocking into the <strong>Analyst Review</strong> queue for human
          investigation. Heuristic rule flags are score-gated and observed unconditionally across all transactions.
        </p>
      </div>

      {/* Segmented Tabs (Condition 2: Honest Counts) */}
      <div className="flex items-center border-b border-slate-800 gap-2">
        <button
          type="button"
          onClick={() => handleTabChange('Analyst Review')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
            activeTab === 'Analyst Review'
              ? 'border-amber-400 text-amber-300 bg-amber-500/5'
              : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
          }`}
          data-testid="tab-analyst-review"
        >
          <Clock className="w-4 h-4 shrink-0" aria-hidden="true" />
          <span>Analyst Review Queue</span>
          <span
            className="px-2 py-0.5 rounded-full text-[11px] font-mono bg-slate-800 text-slate-300 border border-slate-700"
            data-testid="badge-analyst-review-count"
          >
            {countAnalystReview !== undefined ? countAnalystReview : '...'}
          </span>
        </button>

        <button
          type="button"
          onClick={() => handleTabChange('High Risk')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
            activeTab === 'High Risk'
              ? 'border-rose-400 text-rose-300 bg-rose-500/5'
              : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
          }`}
          data-testid="tab-high-risk"
        >
          <ShieldAlert className="w-4 h-4 shrink-0" aria-hidden="true" />
          <span>High Risk Blocked Queue</span>
          <span
            className="px-2 py-0.5 rounded-full text-[11px] font-mono bg-slate-800 text-slate-300 border border-slate-700"
            data-testid="badge-high-risk-count"
          >
            {countHighRisk !== undefined ? countHighRisk : '...'}
          </span>
        </button>
      </div>

      {/* Filter Bar (Decision locked/hidden to avoid tab conflict) */}
      <TransactionFilterBar hideDecision={true} />

      {/* Error State */}
      {error && (
        <div
          role="alert"
          className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-3"
          data-testid="flagged-error-banner"
        >
          <AlertTriangle className="w-5 h-5 shrink-0" aria-hidden="true" />
          <div>
            <div className="font-semibold">Unable to load flagged queue</div>
            <div>{error.message || 'Error occurred while contacting gateway server.'}</div>
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

export default FlaggedQueue;

import { describe, it, expect, beforeEach } from 'vitest';
import { handleMockRequest, resetMockTransactions, setMockConfig } from '../mocks/mockService';

describe('mockService filtering, detail, and review submission', () => {
  beforeEach(() => {
    resetMockTransactions();
    setMockConfig({ enabled: true, state: 'normal' });
  });

  it('filters transactions by decision and returns paginated envelope', async () => {
    const res = await handleMockRequest('/api/v1/transactions?decision=Analyst%20Review');
    expect(res.items.length).toBeGreaterThan(0);
    res.items.forEach((item) => {
      expect(item.gateway_decision).toBe('Analyst Review');
    });
  });

  it('filters transactions by explanation_status', async () => {
    const res = await handleMockRequest('/api/v1/transactions?explanation_status=Unstable');
    expect(res.items.length).toBeGreaterThan(0);
    res.items.forEach((item) => {
      expect(item.explanation_status).toBe('Unstable');
    });
  });

  it('filters transactions by step_min and step_max', async () => {
    const res = await handleMockRequest('/api/v1/transactions?step_min=10&step_max=35');
    res.items.forEach((item) => {
      expect(item.step).toBeGreaterThanOrEqual(10);
      expect(item.step).toBeLessThanOrEqual(35);
    });
  });

  it('paginates transactions accurately according to page and page_size', async () => {
    const page1 = await handleMockRequest('/api/v1/transactions?page=1&page_size=2');
    expect(page1.items.length).toBe(2);
    expect(page1.page).toBe(1);
    expect(page1.page_size).toBe(2);
    expect(page1.total_pages).toBeGreaterThanOrEqual(2);

    const page2 = await handleMockRequest('/api/v1/transactions?page=2&page_size=2');
    expect(page2.items.length).toBeGreaterThan(0);
    expect(page2.page).toBe(2);
    expect(page2.items[0].transaction_id).not.toBe(page1.items[0].transaction_id);
  });

  it('fetches single transaction detail by ID or returns 404', async () => {
    const tx = await handleMockRequest('/api/v1/transactions/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d');
    expect(tx.transaction_id).toBe('9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d');
    expect(tx.prediction).toBeDefined();
    expect(tx.transaction_data).toBeDefined();

    await expect(handleMockRequest('/api/v1/transactions/non-existent-id')).rejects.toMatchObject({
      status: 404,
    });
  });

  it('submits an analyst review, appends to review history, and marks reviewed', async () => {
    const txId = 'c4a7e28b-90f1-4682-a982-f3fbc06d9124';

    const reviewRes = await handleMockRequest(`/api/v1/transactions/${txId}/review`, {
      method: 'POST',
      body: JSON.stringify({
        outcome: 'False Positive',
        note: 'Valid customer business transfer after telephone verification.',
      }),
    });

    expect(reviewRes.transaction_id).toBe(txId);
    expect(reviewRes.outcome).toBe('False Positive');
    expect(reviewRes.note).toBe('Valid customer business transfer after telephone verification.');
    expect(reviewRes.review_id).toBeDefined();
    expect(reviewRes.reviewed_at).toBeDefined();

    // Verify detail reflects the review
    const updatedTx = await handleMockRequest(`/api/v1/transactions/${txId}`);
    expect(updatedTx.reviews.length).toBeGreaterThan(0);
    expect(updatedTx.latest_review.outcome).toBe('False Positive');

    // Verify list item reflects reviewed: true
    const listRes = await handleMockRequest('/api/v1/transactions');
    const listItem = listRes.items.find((item) => item.transaction_id === txId);
    expect(listItem.reviewed).toBe(true);
  });

  it('rejects invalid review submission with 422', async () => {
    const txId = 'c4a7e28b-90f1-4682-a982-f3fbc06d9124';
    await expect(
      handleMockRequest(`/api/v1/transactions/${txId}/review`, {
        method: 'POST',
        body: JSON.stringify({ outcome: 'InvalidOutcome' }),
      })
    ).rejects.toMatchObject({ status: 422 });
  });

  it('includes Variant B Low Risk item with non-empty rule_flags in mock store', async () => {
    const res = await handleMockRequest('/api/v1/transactions?decision=Low%20Risk');
    const withFlags = res.items.find((item) => item.rule_flags && item.rule_flags.length > 0);
    expect(withFlags).toBeDefined();
    expect(withFlags.gateway_decision).toBe('Low Risk');
    expect(withFlags.rule_flags).toContain('LARGE_TRANSFER_AMOUNT');
  });
});

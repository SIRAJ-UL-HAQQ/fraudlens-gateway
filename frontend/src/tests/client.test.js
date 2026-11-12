import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  normalizeError,
  login,
  getHealth,
  getTransactions,
  exportTransactions,
} from '../api/client';
import { tokenStore } from '../auth/tokenStore';

describe('API Client normalizeError', () => {
  it('normalizes 400 with string, array, and dict details', () => {
    const errString = normalizeError(400, { detail: 'Export limit exceeded: max 5000 rows' });
    expect(errString.status).toBe(400);
    expect(errString.message).toBe('Export limit exceeded: max 5000 rows');

    const errArray = normalizeError(400, {
      detail: [{ loc: ['body', 'amount'], msg: 'Amount must be positive' }],
    });
    expect(errArray.status).toBe(400);
    expect(errArray.fieldErrors.amount).toBe('Amount must be positive');

    const errDict = normalizeError(400, {
      detail: { amount: 'Invalid range' },
    });
    expect(errDict.status).toBe(400);
    expect(errDict.fieldErrors.amount).toBe('Invalid range');
  });

  it('normalizes 401 error', () => {
    const err = normalizeError(401, { detail: 'Could not validate credentials' });
    expect(err.status).toBe(401);
    expect(err.message).toBe('Authentication required or session expired.');
  });

  it('normalizes 404 error', () => {
    const err = normalizeError(404, { detail: 'Transaction not found' });
    expect(err.status).toBe(404);
    expect(err.message).toBe('Requested resource was not found.');
  });

  it('normalizes 422 error and maps field validation errors', () => {
    const err = normalizeError(422, {
      detail: [
        { loc: ['body', 'username'], msg: 'Field required' },
        { loc: ['body', 'amount'], msg: 'Value error, amount must be > 0' },
      ],
    });
    expect(err.status).toBe(422);
    expect(err.fieldErrors.username).toBe('Field required');
    expect(err.fieldErrors.amount).toBe('Value error, amount must be > 0');
  });

  it('normalizes 500 error', () => {
    const err = normalizeError(500, { detail: 'Internal server error' });
    expect(err.status).toBe(500);
    expect(err.message).toBe('Internal gateway error. Please try again later.');
    expect(err.retryable).toBe(false);
  });

  it('normalizes 503 as retryable and preserves health payload', () => {
    const healthPayload = {
      status: 'degraded',
      model: { mode: 'unloaded' },
      database: 'connected',
    };
    const err = normalizeError(503, healthPayload);
    expect(err.status).toBe(503);
    expect(err.retryable).toBe(true);
    expect(err.data).toEqual(healthPayload);
  });

  it('normalizes network error (status 0) as retryable', () => {
    const err = normalizeError(0, { detail: 'Failed to fetch' });
    expect(err.status).toBe(0);
    expect(err.retryable).toBe(true);
    expect(err.message).toBe('Network error. Gateway is unreachable.');
  });
});

describe('API Client Network Calls & 401 Interception', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    sessionStorage.clear();
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('fetches health endpoint successfully', async () => {
    const mockHealth = {
      status: 'healthy',
      model: { mode: 'active', version: 'xgb-smote-v1' },
      database: 'connected',
    };
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: async () => mockHealth,
    });

    const res = await getHealth();
    expect(res).toEqual(mockHealth);
  });

  it('does NOT trigger authExpired notification on 401 during login call', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      headers: { get: () => 'application/json' },
      json: async () => ({ detail: 'Invalid username or password' }),
    });

    const onExpired = vi.fn();
    const unsub = tokenStore.onAuthExpired(onExpired);

    await expect(
      login({ username: 'baduser', password: 'badpassword' })
    ).rejects.toThrow();

    expect(onExpired).not.toHaveBeenCalled();
    unsub();
  });

  it('clears token and triggers authExpired on 401 during protected calls with de-duplication', async () => {
    tokenStore.setToken('valid-token-123', 3600, 'analyst_1', 'analyst');
    expect(tokenStore.getToken()).toBe('valid-token-123');

    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      headers: { get: () => 'application/json' },
      json: async () => ({ detail: 'Token expired' }),
    });

    const onExpired = vi.fn();
    const unsub = tokenStore.onAuthExpired(onExpired);

    // Call two protected endpoints concurrently
    const call1 = getTransactions();
    const call2 = getTransactions();

    await expect(call1).rejects.toThrow();
    await expect(call2).rejects.toThrow();

    // Verify token was cleared
    expect(tokenStore.getToken()).toBeNull();

    // De-duplication: only called once despite concurrent 401 responses
    expect(onExpired).toHaveBeenCalledTimes(1);
    unsub();
  });

  it('exportTransactions returns blob on success and normalizes error on failure', async () => {
    const mockBlob = new Blob(['csv,data\n1,2'], { type: 'text/csv' });
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      blob: async () => mockBlob,
    });

    const blobResult = await exportTransactions({ format: 'csv' });
    expect(blobResult).toBeInstanceOf(Blob);

    // Test failure path
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'Export exceeds 5000 records' }),
    });

    await expect(exportTransactions({ format: 'csv' })).rejects.toThrow(
      'Export exceeds 5000 records'
    );
  });
});

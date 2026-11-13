import { describe, it, expect } from 'vitest';
import { normalizeError } from '../api/client';
import { calculateHealthPollingInterval } from '../hooks/useHealthStatus';

describe('Carry-over F1 Fix: client.js 503 error normalization', () => {
  it('unconditionally sets message to fixed copy while preserving error.data', () => {
    // 503 with string detail
    const errString = normalizeError(503, { detail: 'Model is not loaded right now' });
    expect(errString.status).toBe(503);
    expect(errString.message).toBe('Service temporarily unavailable. Please retry.');
    expect(errString.retryable).toBe(true);
    expect(errString.data).toEqual({ detail: 'Model is not loaded right now' });

    // 503 with structured HealthResponse object
    const healthBody = {
      status: 'degraded',
      model: { mode: 'unloaded', version: null },
      database: 'connected',
    };
    const errObject = normalizeError(503, healthBody);
    expect(errObject.status).toBe(503);
    expect(errObject.message).toBe('Service temporarily unavailable. Please retry.');
    expect(errObject.retryable).toBe(true);
    expect(errObject.data).toEqual(healthBody);
  });
});

describe('Carry-over F1 Fix: calculateHealthPollingInterval', () => {
  it('returns 10000 ms for healthy normal status', () => {
    const queryInstance = {
      state: {
        data: { status: 'healthy', model: { mode: 'active' }, database: 'connected' },
        error: null,
      },
    };
    expect(calculateHealthPollingInterval(queryInstance)).toBe(10000);
  });

  it('returns 10000 ms for stub mode (stable operational state)', () => {
    const queryInstance = {
      state: {
        data: { status: 'degraded', model: { mode: 'stub' }, database: 'connected' },
        error: null,
      },
    };
    expect(calculateHealthPollingInterval(queryInstance)).toBe(10000);
  });

  it('returns 3000 ms when model is unloaded without hard query error', () => {
    const queryInstance = {
      state: {
        data: { status: 'degraded', model: { mode: 'unloaded' }, database: 'connected' },
        error: null,
      },
    };
    expect(calculateHealthPollingInterval(queryInstance)).toBe(3000);
  });

  it('returns 3000 ms when database is disconnected without hard query error', () => {
    const queryInstance = {
      state: {
        data: { status: 'degraded', model: { mode: 'active' }, database: 'disconnected' },
        error: null,
      },
    };
    expect(calculateHealthPollingInterval(queryInstance)).toBe(3000);
  });

  it('applies exponential backoff on consecutive query errors (3s, 6s, 12s, 24s, up to 30s cap)', () => {
    const testCases = [
      { errorUpdateCount: 1, expected: 3000 },
      { errorUpdateCount: 2, expected: 6000 },
      { errorUpdateCount: 3, expected: 12000 },
      { errorUpdateCount: 4, expected: 24000 },
      { errorUpdateCount: 5, expected: 30000 }, // capped at 30s
      { errorUpdateCount: 10, expected: 30000 }, // capped at 30s
    ];

    testCases.forEach(({ errorUpdateCount, expected }) => {
      const queryInstance = {
        state: {
          data: null,
          error: new Error('503 or Network error'),
          errorUpdateCount,
        },
      };
      expect(calculateHealthPollingInterval(queryInstance)).toBe(expected);
    });
  });
});

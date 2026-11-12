import fixtures from './fixtures.json';
import { mockOverrides } from './overrides';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function getMockConfig() {
  try {
    const raw = localStorage.getItem('fraudlens_mock_mode');
    return raw ? JSON.parse(raw) : { enabled: false, state: 'normal' };
  } catch {
    return { enabled: false, state: 'normal' };
  }
}

export function setMockConfig(config) {
  try {
    localStorage.setItem('fraudlens_mock_mode', JSON.stringify(config));
  } catch (err) {
    console.error('Failed to set mock config', err);
  }
}

export async function handleMockRequest(url, options = {}) {
  const config = getMockConfig();
  const state = config.state || mockOverrides.activeState;
  await delay(mockOverrides.simulatedDelayMs);

  if (state === 'network_error') {
    throw new TypeError('Failed to fetch (Mock Network Error)');
  }

  if (url.startsWith('/health')) {
    if (state === '503') {
      const err = new Error('HTTP 503');
      err.status = 503;
      err.data = mockOverrides.healthStates.unloaded;
      throw err;
    }
    if (state in mockOverrides.healthStates) {
      return mockOverrides.healthStates[state];
    }
    return fixtures.health_response;
  }

  if (state === '503') {
    const err = new Error('HTTP 503 Service Unavailable');
    err.status = 503;
    err.data = { detail: 'Model service is not loaded. Transaction scoring unavailable.' };
    throw err;
  }

  if (url === '/api/v1/auth/login') {
    const body = options.body ? JSON.parse(options.body) : {};
    if (
      body.username === mockOverrides.fakeCredentials.username &&
      body.password === mockOverrides.fakeCredentials.password
    ) {
      return {
        access_token: mockOverrides.fakeCredentials.token,
        token_type: 'bearer',
        expires_in: 3600,
        username: mockOverrides.fakeCredentials.username,
        role: mockOverrides.fakeCredentials.role,
      };
    }
    const err = new Error('HTTP 401 Unauthorized');
    err.status = 401;
    err.data = { detail: 'Invalid username or password' };
    throw err;
  }

  if (url.startsWith('/api/v1/predict')) {
    return fixtures.single_prediction;
  }

  if (url.startsWith('/api/v1/transactions/')) {
    return fixtures.transactions[0];
  }

  if (url.startsWith('/api/v1/transactions')) {
    const items = fixtures.transactions.map((tx) => ({
      transaction_id: tx.transaction_id,
      step: tx.transaction_data.step,
      type: tx.transaction_data.type,
      amount: tx.transaction_data.amount,
      name_orig: tx.transaction_data.name_orig,
      name_dest: tx.transaction_data.name_dest,
      risk_score: tx.prediction.risk_score,
      gateway_decision: tx.prediction.gateway_decision,
      explanation_status: tx.prediction.explanation_status,
      rule_flags: tx.prediction.rule_flags,
      created_at: tx.created_at,
      reviewed: tx.reviews && tx.reviews.length > 0,
    }));
    return {
      items,
      total: items.length,
      page: 1,
      page_size: 50,
      total_pages: 1,
    };
  }

  if (url.startsWith('/api/v1/metrics')) {
    return fixtures.metrics;
  }

  if (url.startsWith('/api/v1/export')) {
    const csvContent =
      'Transaction ID,Step,Type,Amount,Origin,Destination,Risk Score,Gateway Decision,Explanation Status,Rule Flags,Created At\n' +
      '9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d,1,TRANSFER,181000.0,C1231006815,C553264065,0.94,High Risk,Stable,ORIGIN_ACCOUNT_DRAINED;LARGE_TRANSFER_AMOUNT,2026-10-24T09:00:00Z\n';
    return new Blob([csvContent], { type: 'text/csv; charset=utf-8' });
  }

  return {};
}

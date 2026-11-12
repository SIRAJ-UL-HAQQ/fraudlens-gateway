import fixtures from './fixtures.json';
import { mockOverrides } from './overrides';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Deep clone fixtures + Variant B sample for in-memory mutable mock store
let mockTransactionsStore = initMockTransactions();

function initMockTransactions() {
  const base = JSON.parse(JSON.stringify(fixtures.transactions));
  if (mockOverrides.variantBLowRiskTransaction) {
    base.push(JSON.parse(JSON.stringify(mockOverrides.variantBLowRiskTransaction)));
  }
  return base;
}

export function resetMockTransactions() {
  mockTransactionsStore = initMockTransactions();
}

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

  // POST /api/v1/transactions/{transaction_id}/review
  if (url.match(/\/api\/v1\/transactions\/([^/?#]+)\/review/)) {
    const match = url.match(/\/api\/v1\/transactions\/([^/?#]+)\/review/);
    const txId = decodeURIComponent(match[1]);
    const body = options.body ? JSON.parse(options.body) : {};

    if (!body.outcome || !['Confirmed Fraud', 'False Positive'].includes(body.outcome)) {
      const err = new Error('Validation Error');
      err.status = 422;
      err.data = {
        detail: [
          {
            loc: ['body', 'outcome'],
            msg: "Input should be 'Confirmed Fraud' or 'False Positive'",
          },
        ],
      };
      throw err;
    }

    const tx = mockTransactionsStore.find((t) => t.transaction_id === txId);
    if (!tx) {
      const err = new Error('Not Found');
      err.status = 404;
      err.data = { detail: `Transaction '${txId}' not found` };
      throw err;
    }

    const reviewRecord = {
      review_id: `rev-${Date.now()}-${Math.floor(Math.random() * 10000)}`,
      transaction_id: txId,
      outcome: body.outcome,
      note: body.note ? String(body.note).trim() : null,
      reviewed_by: 'analyst_1',
      reviewed_at: new Date().toISOString(),
    };

    tx.reviews = [reviewRecord, ...(tx.reviews || [])];
    tx.latest_review = reviewRecord;
    return reviewRecord;
  }

  // GET /api/v1/transactions/{transaction_id}
  if (url.match(/\/api\/v1\/transactions\/([^/?#]+)$/)) {
    const match = url.match(/\/api\/v1\/transactions\/([^/?#]+)$/);
    const txId = decodeURIComponent(match[1]);
    const tx = mockTransactionsStore.find((t) => t.transaction_id === txId);
    if (!tx) {
      const err = new Error('Not Found');
      err.status = 404;
      err.data = { detail: `Transaction '${txId}' not found` };
      throw err;
    }
    return JSON.parse(JSON.stringify(tx));
  }

  // GET /api/v1/transactions
  if (url.startsWith('/api/v1/transactions')) {
    const parsedUrl = new URL(url, 'http://localhost');
    const decision = parsedUrl.searchParams.get('decision');
    const explanationStatus = parsedUrl.searchParams.get('explanation_status');
    const stepMin = parsedUrl.searchParams.get('step_min');
    const stepMax = parsedUrl.searchParams.get('step_max');
    const createdFrom = parsedUrl.searchParams.get('created_from');
    const createdTo = parsedUrl.searchParams.get('created_to');
    const page = parseInt(parsedUrl.searchParams.get('page') || '1', 10) || 1;
    const pageSize = Math.min(200, parseInt(parsedUrl.searchParams.get('page_size') || '50', 10) || 50);

    let filtered = [...mockTransactionsStore];

    if (decision) {
      filtered = filtered.filter((tx) => tx.prediction.gateway_decision === decision);
    }
    if (explanationStatus) {
      filtered = filtered.filter((tx) => tx.prediction.explanation_status === explanationStatus);
    }
    if (stepMin !== null && stepMin !== '') {
      filtered = filtered.filter((tx) => tx.transaction_data.step >= Number(stepMin));
    }
    if (stepMax !== null && stepMax !== '') {
      filtered = filtered.filter((tx) => tx.transaction_data.step <= Number(stepMax));
    }
    if (createdFrom) {
      filtered = filtered.filter((tx) => new Date(tx.created_at) >= new Date(createdFrom));
    }
    if (createdTo) {
      filtered = filtered.filter((tx) => new Date(tx.created_at) <= new Date(createdTo));
    }

    const total = filtered.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    const startIdx = (page - 1) * pageSize;
    const pageItems = filtered.slice(startIdx, startIdx + pageSize);

    const items = pageItems.map((tx) => ({
      transaction_id: tx.transaction_id,
      step: tx.transaction_data.step,
      type: tx.transaction_data.type,
      amount: tx.transaction_data.amount,
      name_orig: tx.transaction_data.name_orig,
      name_dest: tx.transaction_data.name_dest,
      risk_score: tx.prediction.risk_score,
      gateway_decision: tx.prediction.gateway_decision,
      explanation_status: tx.prediction.explanation_status,
      rule_flags: tx.prediction.rule_flags || [],
      created_at: tx.created_at,
      reviewed: Boolean(tx.reviews && tx.reviews.length > 0),
    }));

    return {
      items,
      total,
      page,
      page_size: pageSize,
      total_pages: totalPages,
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

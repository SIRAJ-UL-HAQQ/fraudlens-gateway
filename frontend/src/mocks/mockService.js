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
    const body = options.body ? (typeof options.body === 'string' ? JSON.parse(options.body) : options.body) : {};
    const amount = Number(body.amount || 0);
    const oldBal = Number(body.old_balance_orig ?? body.oldbalance_org ?? 0);
    const newBal = Number(body.new_balance_orig ?? body.newbalance_orig ?? 0);
    const txType = body.type || 'TRANSFER';

    // Preset 1: Obvious Fraud (Draining origin balance >= 90% or large transfer >= 100k)
    if ((oldBal > 0 && newBal === 0 && amount >= 50000) || (amount >= 150000 && (txType === 'TRANSFER' || txType === 'CASH_OUT'))) {
      return {
        transaction_id: `pred-fraud-${Date.now()}`,
        risk_score: 0.945,
        top_features: [
          { feature: 'amount', shap_value: 0.4821, direction: 'increases_risk' },
          { feature: `type_${txType}`, shap_value: 0.3512, direction: 'increases_risk' },
          { feature: 'balance_delta_orig', shap_value: -0.1205, direction: 'decreases_risk' },
        ],
        explanation_status: 'Stable',
        gateway_decision: 'High Risk',
        rule_flags: ['ORIGIN_ACCOUNT_DRAINED', 'LARGE_TRANSFER_AMOUNT'],
        imbalance_strategy: 'class_weight',
        model_version: 'v1.0.0-xgb',
        latency_ms: 14.2,
      };
    }

    // Preset 4: Zero Amount edge case
    if (amount === 0) {
      return {
        transaction_id: `pred-zero-${Date.now()}`,
        risk_score: 0.012,
        top_features: [
          { feature: 'amount', shap_value: -0.4512, direction: 'decreases_risk' },
          { feature: `type_${txType}`, shap_value: -0.1834, direction: 'decreases_risk' },
        ],
        explanation_status: 'Stable',
        gateway_decision: 'Low Risk',
        rule_flags: [],
        imbalance_strategy: 'class_weight',
        model_version: 'v1.0.0-xgb',
        latency_ms: 11.5,
      };
    }

    // Preset 3: Borderline / Ambiguous (Moderate TRANSFER, triggers Analyst Review & Unstable SHAP)
    if (amount >= 40000 && amount <= 90000 && txType === 'TRANSFER') {
      return {
        transaction_id: `pred-ambig-${Date.now()}`,
        risk_score: 0.725,
        top_features: [
          { feature: 'amount', shap_value: 0.2215, direction: 'increases_risk' },
          { feature: 'old_balance_orig', shap_value: 0.1842, direction: 'increases_risk' },
          { feature: 'new_balance_orig', shap_value: -0.0912, direction: 'decreases_risk' },
        ],
        explanation_status: 'Unstable',
        gateway_decision: 'Analyst Review',
        rule_flags: [],
        imbalance_strategy: 'smote_enn',
        model_version: 'v1.0.0-xgb',
        latency_ms: 18.7,
      };
    }

    // Preset 2: Obvious Legitimate (small payment, consistent balances)
    if (amount < 1000 || txType === 'PAYMENT') {
      return {
        transaction_id: `pred-legit-${Date.now()}`,
        risk_score: 0.024,
        top_features: [
          { feature: 'amount', shap_value: -0.3214, direction: 'decreases_risk' },
          { feature: `type_${txType}`, shap_value: -0.2518, direction: 'decreases_risk' },
          { feature: 'old_balance_orig', shap_value: -0.1412, direction: 'decreases_risk' },
        ],
        explanation_status: 'Stable',
        gateway_decision: 'Low Risk',
        rule_flags: [],
        imbalance_strategy: 'class_weight',
        model_version: 'v1.0.0-xgb',
        latency_ms: 10.8,
      };
    }

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

    // Server-enforced audit order: created_at desc
    filtered.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

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
    if (config.state === 'metrics_unavailable' || config.metricsUnavailable) {
      return fixtures.metrics_unavailable;
    }
    const metricsCopy = JSON.parse(JSON.stringify(fixtures.metrics));
    if (metricsCopy.ml_metrics && mockOverrides.mockConfusionMatrix) {
      metricsCopy.ml_metrics.confusion_matrix = { ...mockOverrides.mockConfusionMatrix };
      if (metricsCopy.ml_metrics.strategy_comparison?.class_weight) {
        metricsCopy.ml_metrics.strategy_comparison.class_weight.metrics_at_high = {
          confusion_matrix: { ...mockOverrides.mockConfusionMatrix },
        };
      }
    }
    return metricsCopy;
  }

  if (url.startsWith('/api/v1/export')) {
    const parsedUrl = new URL(url, 'http://localhost');
    const format = parsedUrl.searchParams.get('format') || 'csv';

    if (config.exportError || mockOverrides.exportError) {
      const err = new Error('Bad Request');
      err.status = 400;
      err.data = {
        detail: 'Export matching 5200 records exceeds safety cap of 5000 rows. Please narrow your query filters.',
      };
      throw err;
    }

    if (format === 'pdf') {
      const pdfBytes = new Uint8Array([0x25, 0x50, 0x44, 0x46]); // %PDF
      return {
        blob: new Blob([pdfBytes], { type: 'application/pdf' }),
        filename: 'transactions_export.pdf',
      };
    }

    const csvContent =
      'Transaction ID,Step,Type,Amount,Origin,Destination,Risk Score,Gateway Decision,Explanation Status,Rule Flags,Created At\n' +
      '9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d,1,TRANSFER,181000.0,C1231006815,C553264065,0.94,High Risk,Stable,ORIGIN_ACCOUNT_DRAINED;LARGE_TRANSFER_AMOUNT,2026-10-24T09:00:00Z\n';
    return {
      blob: new Blob([csvContent], { type: 'text/csv; charset=utf-8' }),
      filename: 'transactions_export.csv',
    };
  }

  return {};
}

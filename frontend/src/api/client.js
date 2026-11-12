import { tokenStore } from '../auth/tokenStore';

let isHandling401 = false;

export function normalizeError(status, errorBody) {
  let message = 'An unexpected error occurred.';
  const fieldErrors = {};
  let retryable = false;

  const detail = errorBody?.detail !== undefined ? errorBody.detail : errorBody;

  if (status === 400) {
    if (typeof detail === 'string') {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = 'Bad request. Validation failed.';
      detail.forEach((err) => {
        if (err.loc && err.loc.length > 0) {
          const field = err.loc[err.loc.length - 1];
          fieldErrors[field] = err.msg || 'Invalid value';
        }
      });
    } else if (typeof detail === 'object' && detail !== null) {
      Object.assign(fieldErrors, detail);
      message = 'Bad request. Please verify inputs.';
    } else {
      message = 'Bad request. Please verify inputs.';
    }
  } else if (status === 401) {
    message = 'Authentication required or session expired.';
  } else if (status === 404) {
    message = 'Requested resource was not found.';
  } else if (status === 422) {
    message = 'Validation failed. Please check the highlighted fields.';
    if (Array.isArray(detail)) {
      detail.forEach((err) => {
        if (err.loc && err.loc.length > 0) {
          const field = err.loc[err.loc.length - 1];
          fieldErrors[field] = err.msg || 'Invalid value';
        }
      });
    } else if (typeof detail === 'object' && detail !== null) {
      Object.assign(fieldErrors, detail);
    }
  } else if (status === 500) {
    message = 'Internal gateway error. Please try again later.';
  } else if (status === 503) {
    retryable = true;
    message = 'Service temporarily unavailable. Please retry.';
  } else if (status === 0) {
    retryable = true;
    message = 'Network error. Gateway is unreachable.';
  }

  const error = new Error(message);
  error.status = status;
  error.fieldErrors = fieldErrors;
  error.retryable = retryable;
  error.data = errorBody;
  return error;
}

async function request(url, options = {}) {
  // Mock mode interceptor (dev only, tree-shaken in production)
  if (import.meta.env.DEV) {
    try {
      const raw = localStorage.getItem('fraudlens_mock_mode');
      const cfg = raw ? JSON.parse(raw) : null;
      if (cfg && cfg.enabled) {
        const { handleMockRequest } = await import('../mocks/mockService.js');
        return await handleMockRequest(url, options);
      }
    } catch (mockErr) {
      if (mockErr.status) {
        throw normalizeError(mockErr.status, mockErr.data || { detail: mockErr.message });
      }
      throw normalizeError(0, { detail: mockErr.message });
    }
  }

  const headers = { ...options.headers };
  const token = tokenStore.getToken();

  if (token && !headers.Authorization) {
    headers.Authorization = `Bearer ${token}`;
  }

  let response;
  try {
    response = await fetch(url, { ...options, headers });
  } catch (netErr) {
    throw normalizeError(0, { detail: netErr.message });
  }

  if (!response.ok) {
    let errorData = null;
    try {
      errorData = await response.json();
    } catch {
      errorData = { detail: await response.text() };
    }

    // 401 handling
    if (response.status === 401) {
      const isLoginCall = url.includes('/auth/login');
      const isHealthCall = url.includes('/health');

      if (!isLoginCall && !isHealthCall) {
        if (!isHandling401) {
          isHandling401 = true;
          tokenStore.clearToken();
          tokenStore.notifyAuthExpired();
          setTimeout(() => {
            isHandling401 = false;
          }, 1000);
        }
      }
    }

    throw normalizeError(response.status, errorData);
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return await response.json();
  }
  return response;
}

export async function login(credentials) {
  return request('/api/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });
}

export async function getHealth() {
  return request('/health');
}

export async function scoreTransaction(payload) {
  return request('/api/v1/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function getTransactions(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, val]) => {
    if (val !== undefined && val !== null && val !== '') {
      query.append(key, val);
    }
  });
  const queryString = query.toString();
  const url = queryString ? `/api/v1/transactions?${queryString}` : '/api/v1/transactions';
  return request(url);
}

export async function getTransaction(transactionId) {
  return request(`/api/v1/transactions/${encodeURIComponent(transactionId)}`);
}

export async function submitReview(transactionId, reviewData) {
  return request(`/api/v1/transactions/${encodeURIComponent(transactionId)}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reviewData),
  });
}

export async function getMetrics() {
  return request('/api/v1/metrics');
}

export async function exportTransactions(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, val]) => {
    if (val !== undefined && val !== null && val !== '') {
      query.append(key, val);
    }
  });
  const queryString = query.toString();
  const url = queryString ? `/api/v1/export?${queryString}` : '/api/v1/export';

  // Mock mode interceptor (dev only, tree-shaken in production)
  if (import.meta.env.DEV) {
    try {
      const raw = localStorage.getItem('fraudlens_mock_mode');
      const cfg = raw ? JSON.parse(raw) : null;
      if (cfg && cfg.enabled) {
        const { handleMockRequest } = await import('../mocks/mockService.js');
        const mockRes = await handleMockRequest(url);
        if (mockRes && mockRes.blob) {
          return mockRes;
        }
        const format = params.format || 'csv';
        return { blob: mockRes, filename: `transactions_export.${format}` };
      }
    } catch (mockErr) {
      if (mockErr.status) {
        throw normalizeError(mockErr.status, mockErr.data || { detail: mockErr.message });
      }
      throw normalizeError(0, { detail: mockErr.message });
    }
  }

  // Export fetches as blob, but on !ok parses JSON error body
  const headers = {};
  const token = tokenStore.getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(url, { headers });
  } catch (err) {
    throw normalizeError(0, { detail: err.message });
  }

  if (!response.ok) {
    let errorData = null;
    try {
      errorData = await response.json();
    } catch {
      errorData = { detail: await response.text() };
    }
    throw normalizeError(response.status, errorData);
  }

  const contentDisposition = response.headers.get('content-disposition') || '';
  let filename = '';
  const match = contentDisposition.match(/filename\*?=['"]?(?:UTF-\d['"]*)?([^;\r\n"']*)['"]?/i);
  if (match && match[1]) {
    filename = decodeURIComponent(match[1].trim());
  }
  if (!filename) {
    const format = params.format || 'csv';
    filename = `transactions_export.${format}`;
  }

  const blob = await response.blob();
  return { blob, filename };
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export async function scoreTransaction(payload) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  return response.json();
}

export async function getTransactions(decisionFilter = null) {
  const url = decisionFilter
    ? `${API_BASE_URL}/transactions?decision=${decisionFilter}`
    : `${API_BASE_URL}/transactions`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  return response.json();
}

export async function getMetrics() {
  const response = await fetch(`${API_BASE_URL}/metrics`);
  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  return response.json();
}

export function getExportUrl(format = 'csv') {
  return `${API_BASE_URL}/export?format=${format}`;
}

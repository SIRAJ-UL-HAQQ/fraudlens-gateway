#!/usr/bin/env bash
# ==============================================================================
# FraudLens Gateway - Contract & Integration Smoke Test
# ==============================================================================
# Usage:
#   ./scripts/smoke_test.sh [BASE_URL]
# Example:
#   ./scripts/smoke_test.sh http://localhost:8000
# ==============================================================================

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
echo "=========================================================="
echo " Starting FraudLens Gateway Smoke Test"
echo " Target Base URL: ${BASE_URL}"
echo "=========================================================="

# Helper: parse json field using python
json_field() {
    local json="$1"
    local field="$2"
    python -c "import sys, json; data=json.loads(sys.argv[1]); print(data.get('$field', ''))" "$json"
}

# 1. Health Check (Public)
echo "[1/6] Checking /health..."
HEALTH_RESP=$(curl -fsS -X GET "${BASE_URL}/health")
STATUS=$(json_field "${HEALTH_RESP}" "status")
if [ "${STATUS}" != "healthy" ]; then
    echo "ERROR: /health check failed! Response: ${HEALTH_RESP}"
    exit 1
fi
echo "      -> /health OK: ${HEALTH_RESP}"

# 2. Authenticate Analyst
echo "[2/6] Authenticating via POST /api/v1/auth/login..."
LOGIN_PAYLOAD='{"username":"analyst_smoke","password":"smoke_test_password"}'
LOGIN_RESP=$(curl -fsS -X POST "${BASE_URL}/api/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d "${LOGIN_PAYLOAD}")
TOKEN=$(json_field "${LOGIN_RESP}" "access_token")
if [ -z "${TOKEN}" ]; then
    echo "ERROR: Login failed to return access_token! Response: ${LOGIN_RESP}"
    exit 1
fi
echo "      -> Authentication OK (Token acquired)"

AUTH_HEADER="Authorization: Bearer ${TOKEN}"

# 3. Post Sample Transaction to /predict
echo "[3/6] Scoring sample transaction via POST /api/v1/predict..."
PREDICT_PAYLOAD='{
  "step": 1,
  "type": "TRANSFER",
  "amount": 250000.0,
  "name_orig": "C_SMOKE_ORIG_001",
  "old_balance_orig": 250000.0,
  "new_balance_orig": 0.0,
  "name_dest": "C_SMOKE_DEST_001",
  "old_balance_dest": 0.0,
  "new_balance_dest": 0.0
}'

PREDICT_RESP=$(curl -fsS -X POST "${BASE_URL}/api/v1/predict" \
    -H "Content-Type: application/json" \
    -H "${AUTH_HEADER}" \
    -d "${PREDICT_PAYLOAD}")

TX_ID=$(json_field "${PREDICT_RESP}" "transaction_id")
DECISION=$(json_field "${PREDICT_RESP}" "gateway_decision")
EXPLANATION_STATUS=$(json_field "${PREDICT_RESP}" "explanation_status")

if [ -z "${TX_ID}" ] || [ -z "${DECISION}" ] || [ -z "${EXPLANATION_STATUS}" ]; then
    echo "ERROR: Prediction response failed schema check! Response: ${PREDICT_RESP}"
    exit 1
fi
echo "      -> Prediction OK! Transaction ID: ${TX_ID}, Decision: ${DECISION}, Stability: ${EXPLANATION_STATUS}"

# 4. Verify transaction appears in GET /transactions
echo "[4/6] Verifying transaction in GET /api/v1/transactions..."
TXS_RESP=$(curl -fsS -X GET "${BASE_URL}/api/v1/transactions?page=1&page_size=20" \
    -H "${AUTH_HEADER}")

if ! echo "${TXS_RESP}" | grep -q "${TX_ID}"; then
    echo "ERROR: Scored transaction ${TX_ID} not found in /transactions listing!"
    echo "Response: ${TXS_RESP}"
    exit 1
fi
echo "      -> Verified transaction ${TX_ID} present in feed listing."

# 5. Fetch Detail and Submit Review
echo "[5/6] Testing GET /api/v1/transactions/{id} and POST review..."
DETAIL_RESP=$(curl -fsS -X GET "${BASE_URL}/api/v1/transactions/${TX_ID}" \
    -H "${AUTH_HEADER}")
if ! echo "${DETAIL_RESP}" | grep -q "${TX_ID}"; then
    echo "ERROR: Failed to retrieve transaction detail for ${TX_ID}"
    exit 1
fi

REVIEW_PAYLOAD='{"outcome":"Confirmed Fraud","note":"Smoke test confirmed drainage rule."}'
REVIEW_RESP=$(curl -fsS -X POST "${BASE_URL}/api/v1/transactions/${TX_ID}/review" \
    -H "Content-Type: application/json" \
    -H "${AUTH_HEADER}" \
    -d "${REVIEW_PAYLOAD}")

REVIEW_ID=$(json_field "${REVIEW_RESP}" "review_id")
if [ -z "${REVIEW_ID}" ]; then
    echo "ERROR: Review submission failed! Response: ${REVIEW_RESP}"
    exit 1
fi
echo "      -> Review successfully appended: ${REVIEW_ID}"

# 6. Verify Metrics and Export
echo "[6/6] Verifying GET /api/v1/metrics and GET /api/v1/export..."
METRICS_RESP=$(curl -fsS -X GET "${BASE_URL}/api/v1/metrics" -H "${AUTH_HEADER}")
TOTAL_TXS=$(json_field "${METRICS_RESP}" "total_transactions")
if [ -z "${TOTAL_TXS}" ] || [ "${TOTAL_TXS}" -lt 1 ]; then
    echo "ERROR: Metrics check failed! Response: ${METRICS_RESP}"
    exit 1
fi

EXPORT_RESP=$(curl -fsS -X GET "${BASE_URL}/api/v1/export?format=csv" -H "${AUTH_HEADER}")
if ! echo "${EXPORT_RESP}" | grep -q "${TX_ID}"; then
    echo "ERROR: Exported CSV did not contain ${TX_ID}!"
    exit 1
fi
echo "      -> Metrics and Export verified successfully."

echo "=========================================================="
echo " ALL SMOKE TESTS PASSED CLEANLY!"
echo "=========================================================="
exit 0

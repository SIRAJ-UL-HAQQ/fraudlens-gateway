#!/usr/bin/env bash
# ==============================================================================
# FraudLens Gateway - Contract & Integration Smoke Test
# ==============================================================================
# Usage:
#   ./scripts/smoke_test.sh [BASE_URL]
# Example:
#   ./scripts/smoke_test.sh http://127.0.0.1:8000
# ==============================================================================

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
EXPECT_MODEL_MODE="${EXPECT_MODEL_MODE:-}"

# Ingest analyst credentials: from environment or fallback to .env
ANALYST_USERNAME="${ANALYST_USERNAME:-analyst_1}"
if [ -z "${ANALYST_PASSWORD:-}" ] && [ -f .env ]; then
    ANALYST_PASSWORD=$(grep -E "^ANALYST_PASSWORD=" .env | cut -d '=' -f2- | tr -d '\r"' || true)
fi

if [ -z "${ANALYST_PASSWORD:-}" ]; then
    echo "ERROR: ANALYST_PASSWORD is not set and could not be loaded from .env"
    exit 1
fi

echo "=========================================================="
echo " Starting FraudLens Gateway Smoke Test"
echo " Target Base URL:    ${BASE_URL}"
echo " Analyst Username:   ${ANALYST_USERNAME}"
echo " Expect Model Mode:  ${EXPECT_MODEL_MODE:-(any)}"
echo "=========================================================="

# Helper: select python executable (python3 or python)
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "ERROR: python3 or python is required to run smoke tests"
    exit 1
fi

# Helper: parse json field using python
json_field() {
    local json="$1"
    local field="$2"
    "$PYTHON_CMD" -c "import sys, json; data=json.loads(sys.argv[1]); print(data.get(sys.argv[2], ''))" "$json" "$field"
}

# 1. Health Check (Public)
echo "[1/6] Checking /health..."
HEALTH_RESP=$(curl -fsS -X GET "${BASE_URL}/health")
STATUS=$(json_field "${HEALTH_RESP}" "status")
if [ "${STATUS}" != "healthy" ]; then
    echo "ERROR: /health check failed! Response: ${HEALTH_RESP}"
    exit 1
fi

if [ -n "${EXPECT_MODEL_MODE}" ]; then
    ACTUAL_MODE=$("$PYTHON_CMD" -c "import sys, json; print(json.loads(sys.argv[1]).get('model', {}).get('mode', ''))" "${HEALTH_RESP}")
    if [ "${ACTUAL_MODE}" != "${EXPECT_MODEL_MODE}" ]; then
        echo "ERROR: Expected model mode '${EXPECT_MODEL_MODE}', got '${ACTUAL_MODE}' in /health response"
        exit 1
    fi
    echo "      -> Verified model mode '${ACTUAL_MODE}' in /health response."
fi
echo "      -> /health OK: ${HEALTH_RESP}"

# 2. Authenticate Analyst
echo "[2/6] Authenticating via POST /api/v1/auth/login..."
LOGIN_PAYLOAD=$("$PYTHON_CMD" -c "import sys, json; print(json.dumps({'username': sys.argv[1], 'password': sys.argv[2]}))" "${ANALYST_USERNAME}" "${ANALYST_PASSWORD}")
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
MODEL_VERSION=$(json_field "${PREDICT_RESP}" "model_version")

if [ -z "${TX_ID}" ] || [ -z "${DECISION}" ] || [ -z "${EXPLANATION_STATUS}" ]; then
    echo "ERROR: Prediction response failed schema check! Response: ${PREDICT_RESP}"
    exit 1
fi

if [ -n "${EXPECT_MODEL_MODE}" ]; then
    if [ "${EXPECT_MODEL_MODE}" = "stub" ] && [ "${MODEL_VERSION}" != "stub" ]; then
        echo "ERROR: Expected stub model_version in prediction, got '${MODEL_VERSION}'"
        exit 1
    fi
    echo "      -> Verified model_version '${MODEL_VERSION}' matches expected mode '${EXPECT_MODEL_MODE}'."
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

METRICS_STATUS=$(json_field "${METRICS_RESP}" "ml_metrics_status")
if [ -z "${METRICS_STATUS}" ]; then
    echo "ERROR: Metrics response missing ml_metrics_status! Response: ${METRICS_RESP}"
    exit 1
fi

if [ -n "${EXPECT_MODEL_MODE}" ]; then
    if [ "${EXPECT_MODEL_MODE}" = "stub" ] && [ "${METRICS_STATUS}" != "unavailable" ]; then
        echo "ERROR: In stub mode, expected ml_metrics_status 'unavailable', got '${METRICS_STATUS}'"
        exit 1
    fi
    echo "      -> Verified ml_metrics_status '${METRICS_STATUS}' matches expected mode '${EXPECT_MODEL_MODE}'."
else
    echo "      -> Verified ml_metrics_status: ${METRICS_STATUS}"
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

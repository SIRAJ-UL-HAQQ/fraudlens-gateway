# FraudLens Gateway — API Contract & System Specification

> **Source of Truth**: This document and [`backend/app/models/schemas.py`](file:///backend/app/models/schemas.py) define the authoritative API contract for FraudLens Gateway. Any modification requires updating schemas, OpenAPI specification, contract tests, and frontend clients in unison.

---

## 1. System Overview & Conventions

### Base URLs
- **Root Health**: `http://localhost:8000/health`
- **API Version 1**: `http://localhost:8000/api/v1`

### Authentication
- Protected endpoints require an HTTP Bearer JWT token in the `Authorization` header:
  ```http
  Authorization: Bearer <jwt_access_token>
  ```
- Public endpoints:
  - `GET /health`
  - `POST /api/v1/auth/login`
- All other endpoints return `HTTP 401 Unauthorized` if the token is missing or invalid.

### PaySim Dataset & Time Dimension Conventions
- **Dataset Time**: Represented by `step` (simulation hour, integer where `step >= 1`). PaySim synthetic data has no calendar dates. Filter parameters for dataset time are strictly:
  - `step_min`: integer ($\ge 1$)
  - `step_max`: integer ($\ge 1$)
- **Gateway Scoring Time**: Represented by `created_at` (ISO 8601 UTC timestamp recording when the gateway scored the transaction). Filter parameters for system scoring time are strictly:
  - `created_from`: ISO 8601 string (e.g. `2026-10-24T00:00:00Z`)
  - `created_to`: ISO 8601 string (e.g. `2026-10-24T23:59:59Z`)
- **Naming**: The API strictly uses `snake_case` (`old_balance_orig`, `new_balance_orig`, `old_balance_dest`, `new_balance_dest`). PaySim's raw columnar names (`oldbalanceOrg`, `newbalanceOrig`, etc.) are mapped in a single ML ingestion function and never exposed across the network boundary.
- **Amounts and Balances**: `amount >= 0.0` (PaySim contains real zero-amount transactions), and all balance fields are $\ge 0.0$. Negative values are rejected with `HTTP 422 Unprocessable Entity`.

### Tri-State Gateway Risk Routing
1. **Low Risk**: Model score below review threshold and no heuristic rules triggered.
2. **High Risk**: High risk score or critical heuristic rule triggered, with **Stable** explanation.
3. **Analyst Review**: Borderline score OR High Risk prediction where SHAP explanation stability is **Unstable** across imbalance-handling strategies.

### Append-Only Audit Trail
Analyst reviews (`POST /api/v1/transactions/{id}/review`) append new review records to an immutable audit log. Transaction history is never overwritten. `TransactionDetail` exposes the full review log and a computed `latest_review`.

---

## 2. Data Models & Enums

### Enums
- **`TransactionType`**: `"PAYMENT"`, `"TRANSFER"`, `"CASH_OUT"`, `"DEBIT"`, `"CASH_IN"`
- **`GatewayDecision`**: `"Low Risk"`, `"High Risk"`, `"Analyst Review"`
- **`ExplanationStatus`**: `"Stable"`, `"Unstable"`
- **`ReviewOutcome`**: `"Confirmed Fraud"`, `"False Positive"`
- **`ShapDirection`**: `"increases_risk"`, `"decreases_risk"` (validated to match the sign of `shap_value`)
- **`ExportFormat`**: `"csv"`, `"pdf"`

---

## 3. Endpoints

### 3.1. Authentication: Login
- **Endpoint**: `POST /api/v1/auth/login`
- **Access**: Public
- **Description**: Authenticate analyst credentials and issue a Bearer token.

#### Request Body
```json
{
  "username": "analyst_1",
  "password": "<analyst_password>"
}
```

#### Response (`200 OK`)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbmFseXN0XzEifQ.sample_signature",
  "token_type": "bearer",
  "expires_in": 3600,
  "username": "analyst_1",
  "role": "analyst"
}
```

#### Errors
- `400 Bad Request`: Missing username or password.

---

### 3.2. Health Check
- **Endpoint**: `GET /health`
- **Access**: Public
- **Description**: Service liveness and dependency status.

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "service": "FraudLens Gateway",
  "version": "1.0.0",
  "database": "connected",
  "model": {
    "mode": "stub",
    "version": "stub"
  }
}
```

---

### 3.3. Score Transaction: Predict
- **Endpoint**: `POST /api/v1/predict`
- **Access**: Protected (Bearer token required)
- **Description**: Ingest a transaction, evaluate heuristic rules, compute ML probability and SHAP attribution, assess explanation reliability, and determine gateway routing.

#### Request Body (`TransactionIn`)
```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 181000.0,
  "name_orig": "C1231006815",
  "old_balance_orig": 181000.0,
  "new_balance_orig": 0.0,
  "name_dest": "C553264065",
  "old_balance_dest": 0.0,
  "new_balance_dest": 0.0
}
```

#### Response (`200 OK` - `PredictionOut`)
```json
{
  "transaction_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "risk_score": 0.94,
  "top_features": [
    {
      "feature": "amount",
      "shap_value": 0.48,
      "direction": "increases_risk"
    },
    {
      "feature": "type_TRANSFER",
      "shap_value": 0.35,
      "direction": "increases_risk"
    },
    {
      "feature": "balance_delta_orig",
      "shap_value": -0.12,
      "direction": "decreases_risk"
    }
  ],
  "explanation_status": "Stable",
  "gateway_decision": "High Risk",
  "rule_flags": [
    "ORIGIN_ACCOUNT_DRAINED",
    "LARGE_TRANSFER_AMOUNT"
  ],
  "imbalance_strategy": "class_weight",
  "model_version": "stub",
  "latency_ms": 14.2
}
```

#### Errors
- `401 Unauthorized`: Missing or invalid Bearer token.
- `422 Unprocessable Entity`: Validation failure (e.g. `step < 1`, `amount < 0`, negative balances).

---

### 3.4. List Transactions (Feed & Filtering)
- **Endpoint**: `GET /api/v1/transactions`
- **Access**: Protected (Bearer token required)
- **Description**: Paginated transaction feed supporting filtering by dataset step, gateway scoring time, decision, and explanation reliability.

#### Query Parameters
| Parameter | Type | Required | Description |
|---|---|---|---|
| `step_min` | integer ($\ge 1$) | No | Filter transactions where `step >= step_min` |
| `step_max` | integer ($\ge 1$) | No | Filter transactions where `step <= step_max` |
| `created_from` | string (ISO 8601) | No | Scored at or after timestamp |
| `created_to` | string (ISO 8601) | No | Scored at or before timestamp |
| `decision` | string (`GatewayDecision`) | No | Filter by `"Low Risk"`, `"High Risk"`, or `"Analyst Review"` |
| `explanation_status` | string (`ExplanationStatus`) | No | Filter by `"Stable"` or `"Unstable"` |
| `page` | integer ($\ge 1$) | No | Page number (default: `1`) |
| `page_size` | integer ($1 \dots 200$) | No | Items per page (default: `50`) |

#### Response (`200 OK` - `PaginatedTransactionsResponse`)
```json
{
  "items": [
    {
      "transaction_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "step": 1,
      "type": "TRANSFER",
      "amount": 181000.0,
      "name_orig": "C1231006815",
      "name_dest": "C553264065",
      "risk_score": 0.94,
      "gateway_decision": "High Risk",
      "explanation_status": "Stable",
      "rule_flags": [
        "ORIGIN_ACCOUNT_DRAINED",
        "LARGE_TRANSFER_AMOUNT"
      ],
      "created_at": "2026-10-24T09:00:00Z",
      "reviewed": false
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 50,
  "total_pages": 1
}
```

---

### 3.5. Get Transaction Detail
- **Endpoint**: `GET /api/v1/transactions/{transaction_id}`
- **Access**: Protected (Bearer token required)
- **Description**: Fetch detailed record including original input data, full prediction attributes, and complete review audit log.

#### Response (`200 OK` - `TransactionDetail`)
```json
{
  "transaction_id": "c4a7e28b-90f1-4682-a982-f3fbc06d9124",
  "transaction_data": {
    "step": 34,
    "type": "TRANSFER",
    "amount": 75000.0,
    "name_orig": "C998124561",
    "old_balance_orig": 95000.0,
    "new_balance_orig": 20000.0,
    "name_dest": "C112233445",
    "old_balance_dest": 10000.0,
    "new_balance_dest": 85000.0
  },
  "prediction": {
    "transaction_id": "c4a7e28b-90f1-4682-a982-f3fbc06d9124",
    "risk_score": 0.78,
    "top_features": [
      {
        "feature": "amount",
        "shap_value": 0.36,
        "direction": "increases_risk"
      },
      {
        "feature": "new_balance_orig",
        "shap_value": -0.18,
        "direction": "decreases_risk"
      }
    ],
    "explanation_status": "Unstable",
    "gateway_decision": "Analyst Review",
    "rule_flags": [],
    "imbalance_strategy": "class_weight",
    "model_version": "stub",
    "latency_ms": 16.5
  },
  "created_at": "2026-10-24T09:15:00Z",
  "reviews": [
    {
      "review_id": "5f3a0984-dfae-481b-aef3-d6c29c8e1462",
      "transaction_id": "c4a7e28b-90f1-4682-a982-f3fbc06d9124",
      "outcome": "False Positive",
      "note": "Verified customer regular business payroll transfer.",
      "reviewed_by": "analyst_1",
      "reviewed_at": "2026-10-24T09:30:00Z"
    }
  ],
  "latest_review": {
    "review_id": "5f3a0984-dfae-481b-aef3-d6c29c8e1462",
    "transaction_id": "c4a7e28b-90f1-4682-a982-f3fbc06d9124",
    "outcome": "False Positive",
    "note": "Verified customer regular business payroll transfer.",
    "reviewed_by": "analyst_1",
    "reviewed_at": "2026-10-24T09:30:00Z"
  }
}
```

#### Errors
- `404 Not Found`: Transaction does not exist.

---

### 3.6. Submit Analyst Review (Append-Only)
- **Endpoint**: `POST /api/v1/transactions/{transaction_id}/review`
- **Access**: Protected (Bearer token required)
- **Description**: Record an analyst disposition. Appends a new immutable audit record; never modifies or overwrites historical reviews or transactions.

#### Request Body (`AnalystReviewIn`)
```json
{
  "outcome": "Confirmed Fraud",
  "note": "Account balance emptied in single step to unknown counterparty."
}
```

#### Response (`200 OK` - `AnalystReviewRecord`)
```json
{
  "review_id": "5f3a0984-dfae-481b-aef3-d6c29c8e1462",
  "transaction_id": "c4a7e28b-90f1-4682-a982-f3fbc06d9124",
  "outcome": "Confirmed Fraud",
  "note": "Account balance emptied in single step to unknown counterparty.",
  "reviewed_by": "analyst_1",
  "reviewed_at": "2026-10-24T09:30:00Z"
}
```

#### Errors
- `404 Not Found`: Transaction not found.
- `422 Unprocessable Entity`: Invalid outcome enum.

---

### 3.7. Operational & ML Metrics
- **Endpoint**: `GET /api/v1/metrics`
- **Access**: Protected (Bearer token required)
- **Description**: Returns live system telemetry (decision breakdown, p50/p95 latency) alongside ML research benchmark metrics comparing imbalance strategies. When benchmark reports (`ml/reports/metrics.json`) are missing or ungenerated, `strategy_comparison` and `evaluation_metrics` return `null` ('unavailable') so the frontend dashboard displays an appropriate unavailable notice rather than invented values.

#### Response (`200 OK` - `MetricsSummary`)
```json
{
  "total_transactions": 1450,
  "decision_distribution": {
    "Low Risk": 1280,
    "High Risk": 95,
    "Analyst Review": 75
  },
  "explanation_stability_ratio": 0.88,
  "average_risk_score": 0.14,
  "latency_p50_ms": 13.4,
  "latency_p95_ms": 24.8,
  "strategy_comparison": {
    "none": {
      "precision": 0.88,
      "recall": 0.62,
      "f1": 0.73,
      "pr_auc": 0.74,
      "stability": 0.71
    },
    "class_weight": {
      "precision": 0.91,
      "recall": 0.84,
      "f1": 0.87,
      "pr_auc": 0.89,
      "stability": 0.88
    },
    "smote_enn": {
      "precision": 0.85,
      "recall": 0.87,
      "f1": 0.86,
      "pr_auc": 0.87,
      "stability": 0.80
    }
  },
  "evaluation_metrics": {
    "precision": 0.91,
    "recall": 0.84,
    "f1": 0.87,
    "pr_auc": 0.89,
    "mcc": 0.86
  }
}
```

---

### 3.8. Compliance & Audit Export
- **Endpoint**: `GET /api/v1/export`
- **Access**: Protected (Bearer token required)
- **Description**: Export filtered transactions as CSV or PDF report.

#### Query Parameters
- `format`: `"csv"` (default) or `"pdf"`
- `step_min`: integer ($\ge 1$)
- `step_max`: integer ($\ge 1$)
- `created_from`: ISO 8601 string
- `created_to`: ISO 8601 string
- `decision`: `"Low Risk"`, `"High Risk"`, or `"Analyst Review"`
- `explanation_status`: `"Stable"` or `"Unstable"`

#### Response
- **CSV**:
  - `Content-Type`: `text/csv; charset=utf-8`
  - `Content-Disposition`: `attachment; filename="transactions_export.csv"`
- **PDF**:
  - `Content-Type`: `application/pdf`
  - `Content-Disposition`: `attachment; filename="transactions_export.pdf"`

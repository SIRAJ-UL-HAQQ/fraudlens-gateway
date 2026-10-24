import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.auth import VALID_DEV_TOKEN
from app.main import app
from app.models.schemas import (
    AnalystReviewRecord,
    FeatureExplanation,
    GatewayDecision,
    HealthResponse,
    MetricsSummary,
    PaginatedTransactionsResponse,
    PredictionOut,
    ReviewOutcome,
    ShapDirection,
    TokenResponse,
    TransactionDetail,
    TransactionIn,
    TransactionType,
)

client = TestClient(app)
AUTH_HEADERS = {"Authorization": f"Bearer {VALID_DEV_TOKEN}"}


# -------------------------------------------------------------------------
# Task 5 & Requirement 5: Validate frontend fixtures against Pydantic models
# -------------------------------------------------------------------------
def test_fixtures_json_conforms_to_pydantic_schemas():
    """Verify that frontend/src/mocks/fixtures.json parses cleanly into Pydantic models."""
    fixtures_path = Path(__file__).resolve().parents[3] / "frontend" / "src" / "mocks" / "fixtures.json"
    assert fixtures_path.exists(), f"fixtures.json not found at {fixtures_path}"

    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures_data = json.load(f)

    # 1. Transactions list
    assert "transactions" in fixtures_data
    assert len(fixtures_data["transactions"]) > 0
    for tx_raw in fixtures_data["transactions"]:
        validated_tx = TransactionDetail.model_validate(tx_raw)
        assert validated_tx.transaction_id
        assert validated_tx.prediction.risk_score >= 0.0

    # 2. Single prediction
    assert "single_prediction" in fixtures_data
    validated_pred = PredictionOut.model_validate(fixtures_data["single_prediction"])
    assert validated_pred.gateway_decision in [
        GatewayDecision.LOW_RISK,
        GatewayDecision.HIGH_RISK,
        GatewayDecision.ANALYST_REVIEW,
    ]

    # 3. Metrics
    assert "metrics" in fixtures_data
    validated_metrics = MetricsSummary.model_validate(fixtures_data["metrics"])
    assert validated_metrics.total_transactions >= 0

    # 4. Auth response
    assert "auth_response" in fixtures_data
    validated_auth = TokenResponse.model_validate(fixtures_data["auth_response"])
    assert validated_auth.access_token

    # 5. Health response
    assert "health_response" in fixtures_data
    validated_health = HealthResponse.model_validate(fixtures_data["health_response"])
    assert validated_health.status == "healthy"

    # 6. Sample review
    assert "sample_review_record" in fixtures_data
    validated_review = AnalystReviewRecord.model_validate(fixtures_data["sample_review_record"])
    assert validated_review.outcome in [ReviewOutcome.CONFIRMED_FRAUD, ReviewOutcome.FALSE_POSITIVE]


# -------------------------------------------------------------------------
# Requirement 1, 2, 7: Model validation tests
# -------------------------------------------------------------------------
def test_transaction_in_amount_and_step_rules():
    """Zero amount is allowed; negative amount and step < 1 are rejected; large steps allowed."""
    # Zero amount is allowed
    tx_zero = TransactionIn(
        step=1,
        type=TransactionType.CASH_OUT,
        amount=0.0,
        name_orig="C12345",
        old_balance_orig=100.0,
        new_balance_orig=100.0,
        name_dest="C54321",
        old_balance_dest=0.0,
        new_balance_dest=0.0,
    )
    assert tx_zero.amount == 0.0

    # Large step without upper bound allowed
    tx_large_step = TransactionIn(
        step=50000,
        type=TransactionType.PAYMENT,
        amount=10.0,
        name_orig="C12345",
        old_balance_orig=100.0,
        new_balance_orig=90.0,
        name_dest="M54321",
        old_balance_dest=0.0,
        new_balance_dest=0.0,
    )
    assert tx_large_step.step == 50000

    # Negative amount rejected
    with pytest.raises(ValidationError):
        TransactionIn(
            step=1,
            type=TransactionType.PAYMENT,
            amount=-0.01,
            name_orig="C12345",
            old_balance_orig=100.0,
            new_balance_orig=90.0,
            name_dest="M54321",
            old_balance_dest=0.0,
            new_balance_dest=0.0,
        )

    # Step < 1 rejected
    with pytest.raises(ValidationError):
        TransactionIn(
            step=0,
            type=TransactionType.PAYMENT,
            amount=10.0,
            name_orig="C12345",
            old_balance_orig=100.0,
            new_balance_orig=90.0,
            name_dest="M54321",
            old_balance_dest=0.0,
            new_balance_dest=0.0,
        )


def test_feature_explanation_direction_sign_consistency():
    """SHAP direction must match sign of shap_value."""
    # Consistent positive
    feat_pos = FeatureExplanation(
        feature="amount",
        shap_value=0.5,
        direction=ShapDirection.INCREASES_RISK,
    )
    assert feat_pos.direction == ShapDirection.INCREASES_RISK

    # Consistent negative
    feat_neg = FeatureExplanation(
        feature="balance_delta",
        shap_value=-0.2,
        direction=ShapDirection.DECREASES_RISK,
    )
    assert feat_neg.direction == ShapDirection.DECREASES_RISK

    # Inconsistent: positive shap_value with decreases_risk
    with pytest.raises(ValidationError):
        FeatureExplanation(
            feature="amount",
            shap_value=0.4,
            direction=ShapDirection.DECREASES_RISK,
        )

    # Inconsistent: negative shap_value with increases_risk
    with pytest.raises(ValidationError):
        FeatureExplanation(
            feature="amount",
            shap_value=-0.4,
            direction=ShapDirection.INCREASES_RISK,
        )


# -------------------------------------------------------------------------
# Requirement 8: Authentication tests (public vs protected)
# -------------------------------------------------------------------------
def test_public_endpoints_accessible_without_token():
    """Health check and login endpoints do not require authentication."""
    # /health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # /api/v1/auth/login
    res_login = client.post(
        "/api/v1/auth/login",
        json={"username": "analyst", "password": "secure_password"},
    )
    assert res_login.status_code == 200
    data = res_login.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_protected_endpoints_return_401_without_token():
    """All core data and scoring endpoints require valid authentication."""
    endpoints = [
        ("GET", "/api/v1/transactions"),
        ("GET", "/api/v1/transactions/test-id-123"),
        ("POST", "/api/v1/transactions/test-id-123/review"),
        ("POST", "/api/v1/predict"),
        ("GET", "/api/v1/metrics"),
        ("GET", "/api/v1/export"),
    ]
    for method, path in endpoints:
        if method == "GET":
            res = client.get(path)
        else:
            res = client.post(path, json={})
        assert res.status_code == 401, f"Expected 401 for {method} {path}, got {res.status_code}"


def test_protected_endpoints_reject_invalid_token():
    """Supplying a garbage token yields 401."""
    res = client.get("/api/v1/metrics", headers={"Authorization": "Bearer bad-token-123"})
    assert res.status_code == 401


# -------------------------------------------------------------------------
# Requirement 3 & 4: In-memory repository, Predict, Feed, Review, Metrics, Export
# -------------------------------------------------------------------------
def test_predict_and_transaction_lifecycle():
    """End-to-end contract flow: Predict -> Query Feed -> Get Detail -> Submit Review -> Export."""
    sample_tx = {
        "step": 15,
        "type": "TRANSFER",
        "amount": 250000.0,
        "name_orig": "C99887766",
        "old_balance_orig": 250000.0,
        "new_balance_orig": 0.0,
        "name_dest": "C11223344",
        "old_balance_dest": 0.0,
        "new_balance_dest": 0.0,
    }

    # 1. POST /predict
    res_predict = client.post("/api/v1/predict", json=sample_tx, headers=AUTH_HEADERS)
    assert res_predict.status_code == 200
    pred = res_predict.json()
    validated_pred = PredictionOut.model_validate(pred)
    assert validated_pred.gateway_decision in [
        GatewayDecision.LOW_RISK,
        GatewayDecision.HIGH_RISK,
        GatewayDecision.ANALYST_REVIEW,
    ]
    assert len(validated_pred.top_features) > 0
    tx_id = validated_pred.transaction_id

    # 2. GET /transactions (Verify it appears in list)
    res_list = client.get("/api/v1/transactions?page=1&page_size=10", headers=AUTH_HEADERS)
    assert res_list.status_code == 200
    page_data = PaginatedTransactionsResponse.model_validate(res_list.json())
    assert page_data.total >= 1
    found = any(item.transaction_id == tx_id for item in page_data.items)
    assert found, "Scored transaction did not appear in /transactions listing"

    # Test filters: step_min and step_max
    res_step_filter = client.get(
        "/api/v1/transactions?step_min=15&step_max=15",
        headers=AUTH_HEADERS,
    )
    assert res_step_filter.status_code == 200
    filtered_items = res_step_filter.json()["items"]
    assert all(item["step"] == 15 for item in filtered_items)

    # 3. GET /transactions/{id}
    res_detail = client.get(f"/api/v1/transactions/{tx_id}", headers=AUTH_HEADERS)
    assert res_detail.status_code == 200
    detail = TransactionDetail.model_validate(res_detail.json())
    assert detail.transaction_id == tx_id
    assert detail.transaction_data.amount == 250000.0
    assert detail.latest_review is None

    # 4. POST /transactions/{id}/review (Append-only review)
    review_payload = {
        "outcome": "Confirmed Fraud",
        "note": "Account emptied immediately after single high-value transfer.",
    }
    res_review = client.post(
        f"/api/v1/transactions/{tx_id}/review",
        json=review_payload,
        headers=AUTH_HEADERS,
    )
    assert res_review.status_code == 200
    review_record = AnalystReviewRecord.model_validate(res_review.json())
    assert review_record.transaction_id == tx_id
    assert review_record.outcome == ReviewOutcome.CONFIRMED_FRAUD

    # 5. Confirm latest_review is now updated on detail
    res_detail_after = client.get(f"/api/v1/transactions/{tx_id}", headers=AUTH_HEADERS)
    assert res_detail_after.status_code == 200
    detail_after = TransactionDetail.model_validate(res_detail_after.json())
    assert len(detail_after.reviews) == 1
    assert detail_after.latest_review is not None
    assert detail_after.latest_review.outcome == ReviewOutcome.CONFIRMED_FRAUD

    # 6. GET /metrics
    res_metrics = client.get("/api/v1/metrics", headers=AUTH_HEADERS)
    assert res_metrics.status_code == 200
    metrics = MetricsSummary.model_validate(res_metrics.json())
    assert metrics.total_transactions >= 1

    # 7. GET /export
    res_export_csv = client.get("/api/v1/export?format=csv", headers=AUTH_HEADERS)
    assert res_export_csv.status_code == 200
    assert "text/csv" in res_export_csv.headers["content-type"]
    assert tx_id in res_export_csv.text

    res_export_pdf = client.get("/api/v1/export?format=pdf", headers=AUTH_HEADERS)
    assert res_export_pdf.status_code == 200
    assert "application/pdf" in res_export_pdf.headers["content-type"]

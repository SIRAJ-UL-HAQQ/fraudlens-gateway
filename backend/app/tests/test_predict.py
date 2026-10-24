import pytest
from fastapi.testclient import TestClient

from app.core.auth import create_test_token
from app.main import app

client = TestClient(app)
AUTH_HEADERS = {"Authorization": f"Bearer {create_test_token()}"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "FraudLens Gateway"


def test_predict_normal_transaction():
    payload = {
        "step": 10,
        "type": "PAYMENT",
        "amount": 45.0,
        "name_orig": "C123456789",
        "old_balance_orig": 500.0,
        "new_balance_orig": 455.0,
        "name_dest": "M987654321",
        "old_balance_dest": 0.0,
        "new_balance_dest": 0.0,
    }
    response = client.post("/api/v1/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert data["gateway_decision"] in ["Low Risk", "High Risk", "Analyst Review"]
    assert "top_features" in data
    assert "explanation_status" in data


def test_predict_critical_amount_flagged():
    payload = {
        "step": 12,
        "type": "TRANSFER",
        "amount": 250000.0,
        "name_orig": "C22334455",
        "old_balance_orig": 250000.0,
        "new_balance_orig": 0.0,
        "name_dest": "C99887766",
        "old_balance_dest": 0.0,
        "new_balance_dest": 0.0,
    }
    response = client.post("/api/v1/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["gateway_decision"] in ["High Risk", "Analyst Review"]
    assert len(data["rule_flags"]) > 0


def test_metrics_endpoint():
    response = client.get("/api/v1/metrics", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "total_transactions" in data
    assert "decision_distribution" in data


@pytest.mark.unloaded_model
def test_uninitialized_app_cannot_score():
    """Uninitialized app without lifespan must return 503 on /health and /predict without persisting rows."""
    from app.db.repository import get_repository
    from app.models.schemas import HealthResponse

    # 1. GET /health -> returns 503 in HealthResponse shape with mode="unloaded", version="none"
    res_health = client.get("/health")
    assert res_health.status_code == 503
    health_data = HealthResponse.model_validate(res_health.json())
    assert health_data.status == "unhealthy"
    assert health_data.model.mode == "unloaded"
    assert health_data.model.version == "none"

    # Count existing rows before attempt
    repo = get_repository()
    initial_count = repo.list_transactions(page=1, page_size=1).total

    # 2. POST /api/v1/predict -> returns 503
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 50000.0,
        "name_orig": "C123456",
        "old_balance_orig": 50000.0,
        "new_balance_orig": 0.0,
        "name_dest": "C654321",
        "old_balance_dest": 0.0,
        "new_balance_dest": 50000.0,
    }
    res_predict = client.post(
        "/api/v1/predict", json=payload, headers=AUTH_HEADERS
    )
    assert res_predict.status_code == 503
    assert "not loaded" in res_predict.json()["detail"].lower()

    # 3. Assert no transaction was persisted
    post_count = repo.list_transactions(page=1, page_size=1).total
    assert post_count == initial_count

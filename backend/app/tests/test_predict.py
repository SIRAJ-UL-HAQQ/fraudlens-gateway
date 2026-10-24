from fastapi.testclient import TestClient

from app.core.auth import VALID_DEV_TOKEN
from app.main import app

client = TestClient(app)
AUTH_HEADERS = {"Authorization": f"Bearer {VALID_DEV_TOKEN}"}


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

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict_normal_transaction():
    payload = {
        "user_id": "usr_test_01",
        "amount": 45.0,
        "currency": "USD",
        "merchant_id": "merch_grocery",
        "location": "US",
        "device_id": "dev_trusted_phone"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "risk_score" in data
    assert data["decision"] in ["APPROVE", "REVIEW", "DECLINE"]
    assert "shap_explanation" in data

def test_predict_critical_amount_declined():
    payload = {
        "user_id": "usr_test_02",
        "amount": 250000.0,
        "currency": "USD",
        "merchant_id": "merch_luxury",
        "location": "US"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "DECLINE"
    assert len(data["rules_triggered"]) > 0

def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_transactions" in data
    assert "approved_count" in data

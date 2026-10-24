import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.core.auth import create_test_token
from app.db.repository import DatabaseUnavailableError, get_repository
from app.main import app

client = TestClient(app)
AUTH_HEADERS = {"Authorization": f"Bearer {create_test_token()}"}


def test_health_check_db_down():
    """When DB connectivity check fails, /health returns 503 with database: disconnected."""
    with patch("app.main.check_db_connection", return_value=False):
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["database"] == "disconnected"
        assert "model" in data


def test_predict_db_down_returns_503():
    """When DB is down during scoring persistence, return 503 without leaking unpersisted decision."""
    class FailingRepo:
        def save_transaction(self, tx_in, pred_out):
            raise DatabaseUnavailableError("Database connection timed out.")

    app.dependency_overrides[get_repository] = lambda: FailingRepo()
    try:
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
        response = client.post("/api/v1/predict", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 503
        assert "database" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()

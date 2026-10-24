from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.auth import create_test_token
from app.core.config import Settings, settings
from app.main import app

client = TestClient(app)


def test_login_success():
    """Valid credentials return a valid JWT token."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "analyst_1", "password": settings.get_analyst_password()},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["username"] == "analyst_1"


def test_login_invalid_password():
    """Invalid password returns 401 Unauthorized."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "analyst_1", "password": "wrong_password"},
    )
    assert response.status_code == 401


def test_login_missing_fields():
    """Empty username or password returns 400 or 422."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "", "password": ""},
    )
    assert response.status_code in [400, 422]


def test_expired_jwt_rejected():
    """Expired JWT token returns 401 with appropriate message."""
    expired_token = create_test_token(expires_delta=timedelta(seconds=-10))
    response = client.get(
        "/api/v1/transactions",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_tampered_jwt_rejected():
    """Tampered token signature returns 401."""
    token = create_test_token()
    tampered_token = token[:-5] + "XXXXX"
    response = client.get(
        "/api/v1/transactions",
        headers={"Authorization": f"Bearer {tampered_token}"},
    )
    assert response.status_code == 401


def test_dev_token_rejected_now():
    """Hardcoded dev token is no longer accepted."""
    response = client.get(
        "/api/v1/transactions",
        headers={"Authorization": "Bearer old-deprecated-static-token"},
    )
    assert response.status_code == 401


def test_production_mode_fails_with_placeholder_secrets():
    """In production mode, placeholder or missing secrets raise ValidationError."""
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="prod",
            JWT_SECRET_KEY="placeholder_jwt_secret_value",
            ANALYST_PASSWORD="secure_pass_12345678",
        )

    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="prod",
            JWT_SECRET_KEY="a_very_secure_random_key_here_at_least_32_bytes_long",
            ANALYST_PASSWORD="placeholder_analyst_password_value",
        )

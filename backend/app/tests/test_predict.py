import time

import pytest
from fastapi.testclient import TestClient

from app.core.auth import create_test_token
from app.main import app
from app.tests.test_repository import get_postgres_engine_or_skip

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
    assert "latency_ms" in data
    assert data["latency_ms"] > 0.0


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
    from app.models.schemas import HealthResponse, TransactionIn, TransactionType
    from app.services.scoring import ModelNotLoadedError, scoring_service

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

    # 2. Direct scoring_service.predict call raises ModelNotLoadedError before timing
    tx_in = TransactionIn(
        step=1,
        type=TransactionType.TRANSFER,
        amount=50000.0,
        name_orig="C123456",
        old_balance_orig=50000.0,
        new_balance_orig=0.0,
        name_dest="C654321",
        old_balance_dest=0.0,
        new_balance_dest=50000.0,
    )
    with pytest.raises(ModelNotLoadedError) as exc_info:
        scoring_service.predict(tx_in)
    assert "loaded" in str(exc_info.value).lower()

    # 3. POST /api/v1/predict -> returns 503
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

    # 4. Assert no transaction was persisted
    post_count = repo.list_transactions(page=1, page_size=1).total
    assert post_count == initial_count


def test_predict_rule_flag_below_review_routes_low_risk(monkeypatch):
    """Integration test: a rule-flagged transaction with score < review_threshold routes to Low Risk.

    Score control method:
      Monkeypatch `scoring_service._predict_stub` to return an explicit controlled score of 0.20
      (well below the review threshold of 0.40) while using the real rule evaluator.
    Assertions:
      - Response gateway_decision is 'Low Risk' (no alert dispatched since alerts are High Risk only).
      - Rule flags (ORIGIN_ACCOUNT_DRAINED) are present in the response.
      - The persisted transaction record in the repository retains rule_flags and decision 'Low Risk'.
    """
    from app.db.repository import get_repository
    from app.models.schemas import (
        ExplanationStatus,
        FeatureExplanation,
        GatewayDecision,
        ShapDirection,
    )
    from app.services.scoring import scoring_service

    # Control the score to 0.20 (below threshold_review=0.40)
    def mock_predict_stub(tx, features_df):
        return (
            0.20,
            [
                FeatureExplanation(
                    feature="amount",
                    shap_value=0.15,
                    direction=ShapDirection.INCREASES_RISK,
                )
            ],
            ExplanationStatus.STABLE,
        )

    monkeypatch.setattr(scoring_service, "_predict_stub", mock_predict_stub)

    # Transaction triggers ORIGIN_ACCOUNT_DRAINED (drains 100k balance)
    payload = {
        "step": 20,
        "type": "TRANSFER",
        "amount": 100000.0,
        "name_orig": "C99887711",
        "old_balance_orig": 100000.0,
        "new_balance_orig": 0.0,
        "name_dest": "C11223344",
        "old_balance_dest": 0.0,
        "new_balance_dest": 100000.0,
    }

    response = client.post("/api/v1/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()

    # Decision is Low Risk; alert dispatch is not triggered (alerts are High Risk only)
    assert data["gateway_decision"] == GatewayDecision.LOW_RISK.value
    assert data["risk_score"] == 0.20
    assert "ORIGIN_ACCOUNT_DRAINED" in data["rule_flags"]

    # Verify persisted record preserves rule_flags and Low Risk decision
    repo = get_repository()
    tx_record = repo.get_transaction(data["transaction_id"])
    assert tx_record is not None
    assert tx_record.prediction.gateway_decision == GatewayDecision.LOW_RISK
    assert "ORIGIN_ACCOUNT_DRAINED" in tx_record.prediction.rule_flags


def test_predict_stub_latency_simulated_delay(monkeypatch):
    """Test (b): Monkeypatch _predict_stub to sleep 20ms, assert latency_ms >= 19.0.

    Wraps the original _predict_stub, calls it, sleeps 20ms, returns its result via monkeypatch.setattr,
    and ensures service is in stub mode.
    """
    from app.services.scoring import scoring_service

    monkeypatch.setattr(scoring_service, "mode", "stub")
    monkeypatch.setattr(scoring_service, "model_version", "stub")

    original_predict_stub = scoring_service._predict_stub

    def slow_stub(tx, features_df):
        res = original_predict_stub(tx, features_df)
        time.sleep(0.020)
        return res

    monkeypatch.setattr(scoring_service, "_predict_stub", slow_stub)

    payload = {
        "step": 15,
        "type": "PAYMENT",
        "amount": 100.0,
        "name_orig": "C11223344",
        "old_balance_orig": 200.0,
        "new_balance_orig": 100.0,
        "name_dest": "M99887766",
        "old_balance_dest": 0.0,
        "new_balance_dest": 0.0,
    }
    response = client.post("/api/v1/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["latency_ms"] >= 19.0


@pytest.fixture(params=["sqlite", "postgres"])
def api_db_backend(request):
    """Parameterize repository between SQLite and PostgreSQL for API persist testing."""
    from app.db.repository import SqlAlchemyTransactionRepository, get_repository
    from app.db.session import Base

    if request.param == "sqlite":
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        repo = SqlAlchemyTransactionRepository(session_factory=Session)
        app.dependency_overrides[get_repository] = lambda: repo
        yield repo
        app.dependency_overrides.pop(get_repository, None)
        engine.dispose()

    elif request.param == "postgres":
        from sqlalchemy import text
        from sqlalchemy.orm import sessionmaker

        engine = get_postgres_engine_or_skip()
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE transactions, analyst_reviews CASCADE;"))
            conn.commit()

        Session = sessionmaker(bind=engine)
        repo = SqlAlchemyTransactionRepository(session_factory=Session)
        app.dependency_overrides[get_repository] = lambda: repo
        yield repo
        app.dependency_overrides.pop(get_repository, None)
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE transactions, analyst_reviews CASCADE;"))
            conn.commit()


def test_predict_persist_round_trip_stored_latency_equals_response(api_db_backend):
    """Test (c): Persist round trip through API: POST /predict, then GET /api/v1/transactions/{id}.

    Asserts stored latency_ms equals response latency_ms, parameterized for SQLite and Postgres.
    """
    payload = {
        "step": 25,
        "type": "TRANSFER",
        "amount": 75000.0,
        "name_orig": "C10203040",
        "old_balance_orig": 100000.0,
        "new_balance_orig": 25000.0,
        "name_dest": "C90807060",
        "old_balance_dest": 0.0,
        "new_balance_dest": 75000.0,
    }
    res_predict = client.post("/api/v1/predict", json=payload, headers=AUTH_HEADERS)
    assert res_predict.status_code == 200
    pred_data = res_predict.json()
    tx_id = pred_data["transaction_id"]
    response_latency = pred_data["latency_ms"]
    assert response_latency > 0.0

    res_detail = client.get(f"/api/v1/transactions/{tx_id}", headers=AUTH_HEADERS)
    assert res_detail.status_code == 200
    detail_data = res_detail.json()
    assert detail_data["prediction"]["latency_ms"] == response_latency


def test_metrics_latency_percentiles_positive_and_ordered():
    """Test (d): Multiple /predict calls produce positive latency_p50_ms and latency_p95_ms with p95 >= p50.

    Creates its own rows and does not rely on a clean table.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db.repository import SqlAlchemyTransactionRepository, get_repository
    from app.db.session import Base

    isolated_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(isolated_engine)
    Session = sessionmaker(bind=isolated_engine)
    repo = SqlAlchemyTransactionRepository(session_factory=Session)
    app.dependency_overrides[get_repository] = lambda: repo

    try:
        for i in range(10):
            payload = {
                "step": 30 + i,
                "type": "PAYMENT",
                "amount": 50.0 + (i * 10.0),
                "name_orig": f"C_metric_tx_{i}",
                "old_balance_orig": 500.0,
                "new_balance_orig": 450.0 - (i * 10.0),
                "name_dest": f"M_metric_dest_{i}",
                "old_balance_dest": 0.0,
                "new_balance_dest": 0.0,
            }
            res = client.post("/api/v1/predict", json=payload, headers=AUTH_HEADERS)
            assert res.status_code == 200
            assert res.json()["latency_ms"] > 0.0

        res_metrics = client.get("/api/v1/metrics", headers=AUTH_HEADERS)
        assert res_metrics.status_code == 200
        metrics = res_metrics.json()
        assert metrics["latency_p50_ms"] > 0.0
        assert metrics["latency_p95_ms"] > 0.0
        assert metrics["latency_p95_ms"] >= metrics["latency_p50_ms"]
    finally:
        app.dependency_overrides.pop(get_repository, None)
        isolated_engine.dispose()


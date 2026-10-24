from datetime import UTC, datetime
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import TransactionRecord
from app.db.repository import (
    DatabaseUnavailableError,
    InMemoryTransactionRepository,
    SqlAlchemyTransactionRepository,
    sanitize_csv_cell,
)
from app.db.session import Base
from app.models.schemas import (
    AnalystReviewIn,
    ExplanationStatus,
    ExportFormat,
    FeatureExplanation,
    GatewayDecision,
    PredictionOut,
    ReviewOutcome,
    ShapDirection,
    TransactionIn,
    TransactionType,
)


@pytest.fixture(params=["in_memory", "sqlalchemy"])
def repo(request):
    if request.param == "in_memory":
        return InMemoryTransactionRepository()
    else:
        # SQLite in-memory for testing SQLAlchemy repository
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return SqlAlchemyTransactionRepository(session_factory=Session)


def make_sample_tx(tx_id: str, amount: float = 100.0, step: int = 1, decision: GatewayDecision = GatewayDecision.LOW_RISK):
    tx_in = TransactionIn(
        step=step,
        type=TransactionType.TRANSFER,
        amount=amount,
        name_orig="C111",
        old_balance_orig=amount,
        new_balance_orig=0.0,
        name_dest="C222",
        old_balance_dest=0.0,
        new_balance_dest=amount,
    )
    pred_out = PredictionOut(
        transaction_id=tx_id,
        risk_score=0.15 if decision == GatewayDecision.LOW_RISK else 0.85,
        top_features=[
            FeatureExplanation(feature="amount", shap_value=0.25, direction=ShapDirection.INCREASES_RISK),
            FeatureExplanation(feature="old_balance_orig", shap_value=-0.15, direction=ShapDirection.DECREASES_RISK),
        ],
        explanation_status=ExplanationStatus.STABLE,
        gateway_decision=decision,
        rule_flags=[],
        imbalance_strategy="class_weight",
        model_version="stub",
        latency_ms=12.5,
    )
    return tx_in, pred_out


def test_repository_save_and_get(repo):
    tx_in, pred_out = make_sample_tx("tx-001", amount=250.0)
    detail = repo.save_transaction(tx_in, pred_out)

    assert detail.transaction_id == "tx-001"
    assert detail.transaction_data.amount == 250.0
    assert detail.prediction.risk_score == 0.15

    fetched = repo.get_transaction("tx-001")
    assert fetched is not None
    assert fetched.transaction_id == "tx-001"
    assert fetched.transaction_data.amount == 250.0
    assert len(fetched.prediction.top_features) == 2


def test_repository_list_and_pagination(repo):
    for i in range(15):
        tx_in, pred_out = make_sample_tx(
            f"tx-{i:03d}",
            amount=100.0 + i,
            step=i + 1,
            decision=GatewayDecision.HIGH_RISK if i % 2 == 0 else GatewayDecision.LOW_RISK,
        )
        repo.save_transaction(tx_in, pred_out)

    # Filter by decision
    high_page = repo.list_transactions(decision=GatewayDecision.HIGH_RISK, page=1, page_size=10)
    assert high_page.total == 8
    assert len(high_page.items) == 8
    for item in high_page.items:
        assert item.gateway_decision == GatewayDecision.HIGH_RISK

    # Filter by step_min
    step_page = repo.list_transactions(step_min=10, page=1, page_size=20)
    assert step_page.total == 6


def test_repository_add_review_audit_log(repo):
    tx_in, pred_out = make_sample_tx("tx-rev-test")
    repo.save_transaction(tx_in, pred_out)

    rev_in = AnalystReviewIn(
        outcome=ReviewOutcome.CONFIRMED_FRAUD,
        note="Confirmed fraudulent origin account takeover.",
    )
    record = repo.add_review("tx-rev-test", rev_in, reviewed_by="analyst_alice")
    assert record.transaction_id == "tx-rev-test"
    assert record.outcome == ReviewOutcome.CONFIRMED_FRAUD
    assert record.reviewed_by == "analyst_alice"

    fetched = repo.get_transaction("tx-rev-test")
    assert fetched is not None
    assert len(fetched.reviews) == 1
    assert fetched.latest_review is not None
    assert fetched.latest_review.review_id == record.review_id


def test_repository_metrics(repo):
    tx1_in, pred1_out = make_sample_tx("tx-m1", decision=GatewayDecision.LOW_RISK)
    tx2_in, pred2_out = make_sample_tx("tx-m2", decision=GatewayDecision.HIGH_RISK)
    repo.save_transaction(tx1_in, pred1_out)
    repo.save_transaction(tx2_in, pred2_out)

    metrics = repo.get_metrics()
    assert metrics.total_transactions == 2
    assert metrics.decision_distribution["Low Risk"] == 1
    assert metrics.decision_distribution["High Risk"] == 1
    assert metrics.latency_p50_ms > 0.0


def test_repository_export_csv_and_pdf(repo):
    tx_in, pred_out = make_sample_tx("tx-exp")
    repo.save_transaction(tx_in, pred_out)

    csv_data, media_type, filename = repo.export_transactions(format=ExportFormat.CSV)
    assert media_type == "text/csv; charset=utf-8"
    assert filename == "transactions_export.csv"
    assert "tx-exp" in csv_data

    pdf_data, pdf_media, pdf_filename = repo.export_transactions(format=ExportFormat.PDF)
    assert pdf_media == "application/pdf"
    assert pdf_filename == "transactions_export.pdf"
    assert isinstance(pdf_data, bytes)
    assert len(pdf_data) > 100


def test_csv_injection_sanitization():
    """Formulas starting with =, +, -, @, \\t, \\r are prefixed with single quote."""
    assert sanitize_csv_cell("=1+1") == "'=1+1"
    assert sanitize_csv_cell("+cmd|'/C calc'!A0") == "'+cmd|'/C calc'!A0"
    assert sanitize_csv_cell("-5+2") == "'-5+2"
    assert sanitize_csv_cell("@SUM(A1:A10)") == "'@SUM(A1:A10)"
    assert sanitize_csv_cell("\tTAB") == "'\tTAB"
    assert sanitize_csv_cell("normal_text") == "normal_text"
    assert sanitize_csv_cell(123) == "123"


def test_persistence_failure_raises_db_unavailable():
    """When DB connection or write fails, DatabaseUnavailableError is raised."""
    # Bad session factory that fails on connect
    bad_engine = create_engine("sqlite:///non_existent_dir_123/bad.db")
    Session = sessionmaker(bind=bad_engine)
    bad_repo = SqlAlchemyTransactionRepository(session_factory=Session)

    tx_in, pred_out = make_sample_tx("tx-fail")
    with pytest.raises(DatabaseUnavailableError):
        bad_repo.save_transaction(tx_in, pred_out)

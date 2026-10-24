import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

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


def get_postgres_engine_or_skip():
    """Retrieve and validate PostgreSQL test engine.

    Refuses execution unless database name ends in '_test' to prevent data loss.
    Fails if REQUIRE_POSTGRES=1 and TEST_DATABASE_URL is unset.
    """
    test_db_url = os.environ.get("TEST_DATABASE_URL")
    if not test_db_url:
        if os.environ.get("REQUIRE_POSTGRES") == "1":
            pytest.fail(
                "REQUIRE_POSTGRES=1 is set but TEST_DATABASE_URL is not configured."
            )
        pytest.skip("TEST_DATABASE_URL is unset; skipping PostgreSQL repository tests.")

    url = make_url(test_db_url)
    db_name = url.database
    if not db_name or not db_name.endswith("_test"):
        raise RuntimeError(
            f"Refusing to execute tests on database '{db_name}': "
            "database name must end with '_test' to prevent data loss."
        )

    return create_engine(test_db_url)


@pytest.fixture(params=["in_memory", "sqlite", "postgres"])
def repo(request):
    if request.param == "in_memory":
        yield InMemoryTransactionRepository()
    elif request.param == "sqlite":
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        yield SqlAlchemyTransactionRepository(session_factory=Session)
    elif request.param == "postgres":
        engine = get_postgres_engine_or_skip()
        Base.metadata.create_all(engine)

        # Pre-test isolation using raw SQL TRUNCATE CASCADE
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE transactions, analyst_reviews CASCADE;"))
            conn.commit()

        Session = sessionmaker(bind=engine)
        yield SqlAlchemyTransactionRepository(session_factory=Session)

        # Post-test cleanup
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE transactions, analyst_reviews CASCADE;"))
            conn.commit()


def make_sample_tx(
    tx_id: str,
    amount: float = 100.0,
    step: int = 1,
    decision: GatewayDecision = GatewayDecision.LOW_RISK,
    created_at: datetime | None = None,
):
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
            FeatureExplanation(
                feature="amount",
                shap_value=0.25,
                direction=ShapDirection.INCREASES_RISK,
            ),
            FeatureExplanation(
                feature="old_balance_orig",
                shap_value=-0.15,
                direction=ShapDirection.DECREASES_RISK,
            ),
        ],
        explanation_status=ExplanationStatus.STABLE,
        gateway_decision=decision,
        rule_flags=["ORIGIN_ACCOUNT_DRAINED"],
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
            decision=GatewayDecision.HIGH_RISK
            if i % 2 == 0
            else GatewayDecision.LOW_RISK,
        )
        repo.save_transaction(tx_in, pred_out)

    # Filter by decision
    high_page = repo.list_transactions(
        decision=GatewayDecision.HIGH_RISK, page=1, page_size=10
    )
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

    pdf_data, pdf_media, pdf_filename = repo.export_transactions(
        format=ExportFormat.PDF
    )
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
    bad_engine = create_engine("sqlite:///non_existent_dir_123/bad.db")
    Session = sessionmaker(bind=bad_engine)
    bad_repo = SqlAlchemyTransactionRepository(session_factory=Session)

    tx_in, pred_out = make_sample_tx("tx-fail")
    with pytest.raises(DatabaseUnavailableError):
        bad_repo.save_transaction(tx_in, pred_out)


# -------------------------------------------------------------------------
# PostgreSQL Specific Tests (JSONB, UTC-aware timestamps, ordering, migrations)
# -------------------------------------------------------------------------
def _truncate_postgres(engine):
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE transactions, analyst_reviews CASCADE;"))
        conn.commit()


def test_postgres_jsonb_roundtrip():
    """Verify JSONB serialization and deserialization of rule_flags and top_features."""
    engine = get_postgres_engine_or_skip()
    _truncate_postgres(engine)
    Session = sessionmaker(bind=engine)
    repo = SqlAlchemyTransactionRepository(session_factory=Session)

    tx_in, pred_out = make_sample_tx("tx-jsonb")
    pred_out.rule_flags = ["ORIGIN_ACCOUNT_DRAINED", "LARGE_TRANSFER_AMOUNT"]
    pred_out.top_features = [
        FeatureExplanation(
            feature="amount", shap_value=0.45, direction=ShapDirection.INCREASES_RISK
        ),
        FeatureExplanation(
            feature="balance_delta_orig",
            shap_value=-0.25,
            direction=ShapDirection.DECREASES_RISK,
        ),
    ]

    repo.save_transaction(tx_in, pred_out)
    fetched = repo.get_transaction("tx-jsonb")

    assert fetched is not None
    assert fetched.prediction.rule_flags == [
        "ORIGIN_ACCOUNT_DRAINED",
        "LARGE_TRANSFER_AMOUNT",
    ]
    assert len(fetched.prediction.top_features) == 2
    assert fetched.prediction.top_features[0].feature == "amount"
    assert fetched.prediction.top_features[0].shap_value == 0.45
    assert fetched.prediction.top_features[0].direction == ShapDirection.INCREASES_RISK


def test_postgres_utc_aware_created_at():
    """Verify timestamps are persisted and retrieved as timezone-aware UTC."""
    engine = get_postgres_engine_or_skip()
    _truncate_postgres(engine)
    Session = sessionmaker(bind=engine)
    repo = SqlAlchemyTransactionRepository(session_factory=Session)

    now_utc = datetime.now(UTC)
    tx_in, pred_out = make_sample_tx("tx-utc")
    repo.save_transaction(tx_in, pred_out, created_at=now_utc)
    fetched = repo.get_transaction("tx-utc")

    assert fetched is not None
    assert fetched.created_at.tzinfo is not None
    # Compare timestamps within 1 second tolerance
    diff = abs((fetched.created_at - now_utc).total_seconds())
    assert diff < 1.0


def test_postgres_deterministic_ordering():
    """Verify deterministic ordering: created_at desc, transaction_id asc."""
    engine = get_postgres_engine_or_skip()
    _truncate_postgres(engine)
    Session = sessionmaker(bind=engine)
    repo = SqlAlchemyTransactionRepository(session_factory=Session)

    base_time = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC)
    # Save transactions with varying timestamps and IDs
    t1_in, p1_out = make_sample_tx("tx-c", created_at=base_time)
    t2_in, p2_out = make_sample_tx("tx-a", created_at=base_time)
    t3_in, p3_out = make_sample_tx("tx-b", created_at=base_time + timedelta(minutes=5))

    repo.save_transaction(t1_in, p1_out, created_at=base_time)
    repo.save_transaction(t2_in, p2_out, created_at=base_time)
    repo.save_transaction(t3_in, p3_out, created_at=base_time + timedelta(minutes=5))

    page = repo.list_transactions(page=1, page_size=10)
    tx_ids = [item.transaction_id for item in page.items]

    # tx-b is newest; tx-a and tx-c have same created_at, so tie-breaker is transaction_id asc
    assert tx_ids == ["tx-b", "tx-a", "tx-c"]


def test_postgres_alembic_migrations():
    """Verify Alembic migration lifecycle: upgrade head -> downgrade base -> upgrade head -> check."""
    engine = get_postgres_engine_or_skip()
    from alembic import command
    from alembic.config import Config

    # Drop existing tables and alembic version tracking to test full migration lifecycle cleanly
    with engine.connect() as conn:
        conn.execute(
            text(
                "DROP TABLE IF EXISTS transactions, analyst_reviews, alembic_version CASCADE;"
            )
        )
        conn.commit()

    alembic_cfg_path = Path(__file__).resolve().parents[3] / "alembic.ini"
    if not alembic_cfg_path.exists():
        alembic_cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    assert alembic_cfg_path.exists(), f"alembic.ini not found at {alembic_cfg_path}"

    alembic_cfg = Config(str(alembic_cfg_path))
    alembic_cfg.set_main_option("sqlalchemy.url", os.environ["TEST_DATABASE_URL"])

    # 1. Upgrade to head
    command.upgrade(alembic_cfg, "head")

    # 2. Downgrade to base
    command.downgrade(alembic_cfg, "base")

    # 3. Re-upgrade to head
    command.upgrade(alembic_cfg, "head")

    # 4. Run alembic check to ensure no ungenerated migration drift
    command.check(alembic_cfg)

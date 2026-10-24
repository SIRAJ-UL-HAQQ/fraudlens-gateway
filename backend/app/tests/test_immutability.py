import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import AnalystReviewModel, ImmutableRecordError, TransactionRecord
from app.db.session import Base


@pytest.fixture
def db_session():
    # In-memory SQLite for immutability tests
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_transaction_update_forbidden(db_session):
    tx = TransactionRecord(
        transaction_id="tx-100",
        step=1,
        type="TRANSFER",
        amount=500.0,
        name_orig="C1",
        old_balance_orig=500.0,
        new_balance_orig=0.0,
        name_dest="C2",
        old_balance_dest=0.0,
        new_balance_dest=500.0,
        risk_score=0.85,
        gateway_decision="High Risk",
        explanation_status="Stable",
        rule_flags=[],
        top_features=[],
        imbalance_strategy="class_weight",
        model_version="stub",
        latency_ms=10.0,
    )
    db_session.add(tx)
    db_session.commit()

    # Attempting update raises ImmutableRecordError
    tx.amount = 999.0
    with pytest.raises(ImmutableRecordError) as exc_info:
        db_session.commit()
    assert "append-only" in str(exc_info.value).lower()
    db_session.rollback()


def test_transaction_delete_forbidden(db_session):
    tx = TransactionRecord(
        transaction_id="tx-200",
        step=1,
        type="PAYMENT",
        amount=50.0,
        name_orig="C1",
        old_balance_orig=500.0,
        new_balance_orig=450.0,
        name_dest="M2",
        old_balance_dest=0.0,
        new_balance_dest=0.0,
        risk_score=0.10,
        gateway_decision="Low Risk",
        explanation_status="Stable",
        rule_flags=[],
        top_features=[],
        imbalance_strategy="class_weight",
        model_version="stub",
        latency_ms=8.0,
    )
    db_session.add(tx)
    db_session.commit()

    with pytest.raises(ImmutableRecordError) as exc_info:
        db_session.delete(tx)
        db_session.commit()
    assert "append-only" in str(exc_info.value).lower()
    db_session.rollback()


def test_analyst_review_update_and_delete_forbidden(db_session):
    tx = TransactionRecord(
        transaction_id="tx-300",
        step=1,
        type="TRANSFER",
        amount=100.0,
        name_orig="C1",
        old_balance_orig=100.0,
        new_balance_orig=0.0,
        name_dest="C2",
        old_balance_dest=0.0,
        new_balance_dest=100.0,
        risk_score=0.75,
        gateway_decision="High Risk",
        explanation_status="Stable",
        rule_flags=[],
        top_features=[],
        imbalance_strategy="class_weight",
        model_version="stub",
        latency_ms=12.0,
    )
    db_session.add(tx)
    db_session.commit()

    review = AnalystReviewModel(
        review_id="rev-1",
        transaction_id="tx-300",
        outcome="Confirmed Fraud",
        note="Initial investigation notes",
        reviewed_by="analyst_1",
    )
    db_session.add(review)
    db_session.commit()

    # Attempting update raises ImmutableRecordError
    review.note = "Tampered notes"
    with pytest.raises(ImmutableRecordError) as exc_info:
        db_session.commit()
    assert "append-only" in str(exc_info.value).lower()
    db_session.rollback()

    # Attempting delete raises ImmutableRecordError
    rev_to_del = db_session.query(AnalystReviewModel).filter_by(review_id="rev-1").first()
    with pytest.raises(ImmutableRecordError) as exc_info:
        db_session.delete(rev_to_del)
        db_session.commit()
    assert "append-only" in str(exc_info.value).lower()
    db_session.rollback()

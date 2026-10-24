import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    event,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import relationship

from app.db.session import Base


class ImmutableRecordError(Exception):
    """Raised when an attempt is made to update or delete an immutable record."""


# JSON column with JSONB variant on PostgreSQL
JSONType = JSON().with_variant(postgresql.JSONB, "postgresql")


class TransactionRecord(Base):
    """Immutable transaction scoring and feature record."""

    __tablename__ = "transactions"
    __table_args__ = (
        Index("ix_transactions_decision_created_at", "gateway_decision", "created_at"),
    )

    transaction_id = Column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    step = Column(Integer, nullable=False)
    type = Column(String(16), nullable=False)
    amount = Column(Float, nullable=False)
    name_orig = Column(String(64), nullable=False)
    old_balance_orig = Column(Float, nullable=False)
    new_balance_orig = Column(Float, nullable=False)
    name_dest = Column(String(64), nullable=False)
    old_balance_dest = Column(Float, nullable=False)
    new_balance_dest = Column(Float, nullable=False)

    risk_score = Column(Float, nullable=False)
    gateway_decision = Column(String(32), nullable=False, index=True)
    explanation_status = Column(String(16), nullable=False)
    rule_flags = Column(JSONType, nullable=False, default=list)
    top_features = Column(JSONType, nullable=False, default=list)
    imbalance_strategy = Column(String(32), nullable=False)
    model_version = Column(String(64), nullable=False)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    reviews = relationship(
        "AnalystReviewModel",
        back_populates="transaction",
        cascade="all, delete-orphan",
        order_by="AnalystReviewModel.reviewed_at.asc()",
    )


class AnalystReviewModel(Base):
    """Immutable audit trail for human analyst dispositions."""

    __tablename__ = "analyst_reviews"

    review_id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id = Column(
        String(64),
        ForeignKey("transactions.transaction_id"),
        nullable=False,
        index=True,
    )
    outcome = Column(String(32), nullable=False)
    note = Column(Text, nullable=True)
    reviewed_by = Column(String(64), nullable=False)
    reviewed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    transaction = relationship("TransactionRecord", back_populates="reviews")


# -------------------------------------------------------------------------
# Immutability Listeners: Disallow UPDATE and DELETE on audit trail tables
# -------------------------------------------------------------------------
@event.listens_for(TransactionRecord, "before_update")
def disallow_transaction_update(mapper, connection, target):
    raise ImmutableRecordError(
        "Transactions table is append-only: updates are strictly forbidden."
    )


@event.listens_for(TransactionRecord, "before_delete")
def disallow_transaction_delete(mapper, connection, target):
    raise ImmutableRecordError(
        "Transactions table is append-only: deletions are strictly forbidden."
    )


@event.listens_for(AnalystReviewModel, "before_update")
def disallow_review_update(mapper, connection, target):
    raise ImmutableRecordError(
        "Analyst reviews table is append-only: updates are strictly forbidden."
    )


@event.listens_for(AnalystReviewModel, "before_delete")
def disallow_review_delete(mapper, connection, target):
    raise ImmutableRecordError(
        "Analyst reviews table is append-only: deletions are strictly forbidden."
    )

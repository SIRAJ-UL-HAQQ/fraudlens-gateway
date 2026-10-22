import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Text
from app.db.session import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(64), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(8), default="USD")
    merchant_id = Column(String(64), nullable=True)
    location = Column(String(64), default="US")
    device_id = Column(String(64), nullable=True)
    ip_address = Column(String(64), nullable=True)
    fraud_probability = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=False)
    decision = Column(String(16), index=True, nullable=False) # APPROVE, REVIEW, DECLINE
    rules_triggered = Column(Text, default="[]")              # JSON serialized
    shap_top_features = Column(Text, default="[]")            # JSON serialized
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id = Column(String(64), index=True, nullable=False)
    actor = Column(String(64), default="SYSTEM")
    action = Column(String(32), nullable=False) # AUTO_DECISION, MANUAL_OVERRIDE, COMMENT
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class RuleViolation(BaseModel):
    rule_name: str
    description: str
    severity: str  # LOW, MEDIUM, CRITICAL

class FeatureShap(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str  # INCREASES_RISK, DECREASES_RISK

class PredictionRequest(BaseModel):
    transaction_id: Optional[str] = None
    user_id: str = Field(..., example="usr_9281")
    amount: float = Field(..., gt=0, example=1250.00)
    currency: str = Field(default="USD", example="USD")
    merchant_id: str = Field(..., example="merch_55")
    location: str = Field(default="US", example="US")
    device_id: Optional[str] = Field(default="dev_unknown", example="dev_a81f")
    ip_address: Optional[str] = Field(default="127.0.0.1", example="192.168.1.1")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    custom_features: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_score: int  # 0 to 100
    decision: str    # APPROVE, REVIEW, DECLINE
    rules_triggered: List[RuleViolation]
    shap_explanation: List[FeatureShap]
    shap_reliability_status: str  # HIGH, MODERATE, LOW
    latency_ms: float
    timestamp: datetime

class TransactionSummary(BaseModel):
    id: str
    user_id: str
    amount: float
    currency: str
    decision: str
    risk_score: int
    created_at: datetime

    class Config:
        from_attributes = True

class MetricsSummary(BaseModel):
    total_transactions: int
    approved_count: int
    review_count: int
    declined_count: int
    fraud_rate_pct: float
    average_risk_score: float
    avg_latency_ms: float

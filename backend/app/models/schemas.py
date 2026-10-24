from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TransactionType(str, Enum):
    """PaySim transaction types."""

    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    CASH_IN = "CASH_IN"


class GatewayDecision(str, Enum):
    """Tri-state gateway risk routing decision."""

    LOW_RISK = "Low Risk"
    HIGH_RISK = "High Risk"
    ANALYST_REVIEW = "Analyst Review"


class ExplanationStatus(str, Enum):
    """SHAP explanation reliability under class imbalance."""

    STABLE = "Stable"
    UNSTABLE = "Unstable"


class ReviewOutcome(str, Enum):
    """Analyst review disposition."""

    CONFIRMED_FRAUD = "Confirmed Fraud"
    FALSE_POSITIVE = "False Positive"


class ShapDirection(str, Enum):
    """Direction of SHAP feature impact on fraud probability."""

    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"


class ExportFormat(str, Enum):
    """Supported export formats."""

    CSV = "csv"
    PDF = "pdf"


class TransactionIn(BaseModel):
    """Incoming transaction schema using snake_case PaySim field mappings."""

    step: int = Field(..., ge=1, description="Simulation step in hours (>= 1)")
    type: TransactionType = Field(..., description="Transaction type")
    amount: float = Field(
        ..., ge=0.0, description="Transaction amount in currency units (>= 0)"
    )
    name_orig: str = Field(
        ..., min_length=1, description="Origin account ID (e.g. C123456789)"
    )
    old_balance_orig: float = Field(
        ..., ge=0.0, description="Initial balance of origin before transaction"
    )
    new_balance_orig: float = Field(
        ..., ge=0.0, description="New balance of origin after transaction"
    )
    name_dest: str = Field(
        ...,
        min_length=1,
        description="Destination account ID (e.g. M123456789 or C987654321)",
    )
    old_balance_dest: float = Field(
        ..., ge=0.0, description="Initial balance of destination before transaction"
    )
    new_balance_dest: float = Field(
        ..., ge=0.0, description="New balance of destination after transaction"
    )


class FeatureExplanation(BaseModel):
    """SHAP feature attribution with strict direction sign validation."""

    feature: str = Field(..., min_length=1, description="Feature name")
    shap_value: float = Field(..., description="SHAP attribution value")
    direction: ShapDirection = Field(
        ..., description="Direction: 'increases_risk' or 'decreases_risk'"
    )

    @model_validator(mode="after")
    def validate_direction_consistency(self) -> "FeatureExplanation":
        if self.shap_value > 0 and self.direction != ShapDirection.INCREASES_RISK:
            raise ValueError(
                "direction must be 'increases_risk' when shap_value is positive"
            )
        if self.shap_value < 0 and self.direction != ShapDirection.DECREASES_RISK:
            raise ValueError(
                "direction must be 'decreases_risk' when shap_value is negative"
            )
        return self


class PredictionOut(BaseModel):
    """System prediction output containing score, SHAP explanation, and gateway decision."""

    transaction_id: str = Field(..., description="Unique transaction UUID")
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated risk probability score (0.0 to 1.0)",
    )
    top_features: list[FeatureExplanation] = Field(
        default_factory=list, description="Top SHAP feature attributions"
    )
    explanation_status: ExplanationStatus = Field(
        ..., description="Explanation stability: Stable or Unstable"
    )
    gateway_decision: GatewayDecision = Field(
        ..., description="Routing decision: Low Risk, High Risk, or Analyst Review"
    )
    rule_flags: list[str] = Field(
        default_factory=list, description="List of rule flags triggered"
    )
    imbalance_strategy: str = Field(
        ...,
        description="Imbalance strategy used (e.g. 'none', 'class_weight', 'smote_enn')",
    )
    model_version: str = Field(..., description="Model version tag or 'stub'")
    latency_ms: float = Field(
        ..., ge=0.0, description="Prediction inference latency in milliseconds"
    )


class AnalystReviewIn(BaseModel):
    """Payload for submitting an analyst review decision."""

    outcome: ReviewOutcome = Field(..., description="Confirmed Fraud or False Positive")
    note: str | None = Field(
        default=None, description="Analyst review notes or investigation comments"
    )


class AnalystReviewRecord(BaseModel):
    """Append-only audit log entry for an analyst review."""

    review_id: str = Field(..., description="Unique review UUID")
    transaction_id: str = Field(..., description="Associated transaction UUID")
    outcome: ReviewOutcome = Field(..., description="Review outcome")
    note: str | None = Field(default=None, description="Analyst review notes")
    reviewed_by: str = Field(..., description="Analyst username")
    reviewed_at: datetime = Field(..., description="Timestamp of the review")


class TransactionSummary(BaseModel):
    """Summary item for transaction list feeds and tables."""

    transaction_id: str = Field(..., description="Transaction UUID")
    step: int = Field(..., description="Step hour")
    type: TransactionType = Field(..., description="Transaction type")
    amount: float = Field(..., description="Transaction amount")
    name_orig: str = Field(..., description="Origin account ID")
    name_dest: str = Field(..., description="Destination account ID")
    risk_score: float = Field(..., description="Risk score")
    gateway_decision: GatewayDecision = Field(..., description="Gateway decision")
    explanation_status: ExplanationStatus = Field(
        ..., description="Explanation stability"
    )
    rule_flags: list[str] = Field(default_factory=list, description="Rule flags")
    created_at: datetime = Field(..., description="Timestamp of scoring")
    reviewed: bool = Field(
        default=False, description="Whether transaction has been reviewed"
    )


class PaginatedTransactionsResponse(BaseModel):
    """Paginated response envelope for transaction feeds."""

    items: list[TransactionSummary] = Field(
        default_factory=list, description="Page items"
    )
    total: int = Field(..., ge=0, description="Total matching records count")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")


class TransactionDetail(BaseModel):
    """Full transaction record including input data, prediction output, and audit log."""

    transaction_id: str = Field(..., description="Transaction UUID")
    transaction_data: TransactionIn = Field(
        ..., description="Original transaction input data"
    )
    prediction: PredictionOut = Field(
        ..., description="Gateway prediction and explanation"
    )
    created_at: datetime = Field(..., description="Timestamp of initial scoring")
    reviews: list[AnalystReviewRecord] = Field(
        default_factory=list, description="Append-only audit review log"
    )
    latest_review: AnalystReviewRecord | None = Field(
        default=None, description="Derived latest review record"
    )

    @model_validator(mode="after")
    def compute_latest_review(self) -> "TransactionDetail":
        if self.reviews and not self.latest_review:
            self.latest_review = self.reviews[-1]
        return self


class MLMetrics(BaseModel):
    """Machine learning benchmark evaluation metrics from research reports."""

    strategy_comparison: dict[str, Any] = Field(
        ..., description="Breakdown across imbalance strategies"
    )
    evaluation_metrics: dict[str, float] = Field(
        ..., description="Evaluation metrics (Precision, Recall, F1, PR-AUC, MCC)"
    )


class OperationalMetrics(BaseModel):
    """Database-aggregated live operational metrics."""

    total_transactions: int = Field(
        ..., ge=0, description="Total transactions processed"
    )
    decision_distribution: dict[str, int] = Field(
        ..., description="Count of decisions by GatewayDecision"
    )
    explanation_stability_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of explanations scored Stable"
    )
    average_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Average risk score across transactions"
    )
    latency_p50_ms: float = Field(
        ..., ge=0.0, description="Median latency in milliseconds"
    )
    latency_p95_ms: float = Field(
        ..., ge=0.0, description="95th percentile latency in milliseconds"
    )


class MetricsSummary(BaseModel):
    """Aggregated operational telemetry and ML research metrics summary."""

    total_transactions: int = Field(
        ..., ge=0, description="Total transactions processed"
    )
    decision_distribution: dict[str, int] = Field(
        ..., description="Count of decisions by GatewayDecision"
    )
    explanation_stability_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of explanations scored Stable"
    )
    average_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Average risk score across transactions"
    )
    latency_p50_ms: float = Field(
        ..., ge=0.0, description="Median latency in milliseconds"
    )
    latency_p95_ms: float = Field(
        ..., ge=0.0, description="95th percentile latency in milliseconds"
    )
    ml_metrics: MLMetrics | None = Field(
        default=None, description="ML benchmark metrics or null when unavailable"
    )
    ml_metrics_status: Literal["available", "unavailable"] = Field(
        ..., description="Availability status of ML metrics"
    )
    ml_metrics_reason: str | None = Field(
        default=None, description="Explanation when ml_metrics is unavailable"
    )

    @model_validator(mode="after")
    def validate_metrics_availability(self) -> "MetricsSummary":
        if self.ml_metrics_status == "unavailable":
            if self.ml_metrics is not None:
                raise ValueError(
                    "ml_metrics must be None when ml_metrics_status is 'unavailable'."
                )
            if not self.ml_metrics_reason:
                raise ValueError(
                    "ml_metrics_reason must be provided when ml_metrics_status is 'unavailable'."
                )
        elif self.ml_metrics_status == "available":
            if self.ml_metrics is None:
                raise ValueError(
                    "ml_metrics cannot be None when ml_metrics_status is 'available'."
                )
            if self.ml_metrics_reason is not None:
                raise ValueError(
                    "ml_metrics_reason must be None when ml_metrics_status is 'available'."
                )
        return self


class LoginRequest(BaseModel):
    """Analyst authentication credentials."""

    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")


class TokenResponse(BaseModel):
    """JWT bearer token response."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(default=3600, description="Token lifetime in seconds")
    username: str = Field(..., description="Authenticated username")
    role: str = Field(default="analyst", description="User role")


class ModelInfo(BaseModel):
    """Runtime model mode and version information."""

    mode: Literal["trained", "stub", "unloaded"] = Field(
        default="stub", description="Model runtime mode: 'trained', 'stub', or 'unloaded'"
    )
    version: str = Field(default="stub", description="Model version tag or identifier")


class HealthResponse(BaseModel):
    """System health check status."""

    status: str = Field(default="healthy", description="Health status indicator")
    service: str = Field(default="FraudLens Gateway", description="Service name")
    version: str = Field(default="1.0.0", description="Service version")
    database: str = Field(
        default="connected", description="Database connectivity state"
    )
    model: ModelInfo = Field(
        default_factory=ModelInfo, description="Model runtime mode and version"
    )

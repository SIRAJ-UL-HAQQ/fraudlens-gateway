import time
import uuid

from app.models.schemas import (
    ExplanationStatus,
    FeatureExplanation,
    GatewayDecision,
    PredictionOut,
    ShapDirection,
    TransactionIn,
    TransactionType,
)


class ScoringService:
    """Deterministic stub scoring service for contract testing and local execution."""

    def predict(self, req: TransactionIn) -> PredictionOut:
        start_time = time.perf_counter()
        tx_id = str(uuid.uuid4())

        rule_flags: list[str] = []
        # Rule 1: Drained balance on cash out / transfer
        if (
            req.type in [TransactionType.TRANSFER, TransactionType.CASH_OUT]
            and req.old_balance_orig > 0
            and req.new_balance_orig == 0
            and req.amount >= req.old_balance_orig * 0.9
        ):
            rule_flags.append("ORIGIN_ACCOUNT_DRAINED")

        # Rule 2: Large amount threshold
        if req.amount >= 200_000.0 and req.type in [TransactionType.TRANSFER, TransactionType.CASH_OUT]:
            rule_flags.append("LARGE_TRANSFER_AMOUNT")

        # Deterministic stub risk calculation
        if "ORIGIN_ACCOUNT_DRAINED" in rule_flags or "LARGE_TRANSFER_AMOUNT" in rule_flags:
            risk_score = 0.92
            explanation_status = ExplanationStatus.STABLE
        elif req.amount > 50_000.0 and req.type == TransactionType.TRANSFER:
            risk_score = 0.76
            explanation_status = ExplanationStatus.UNSTABLE
        elif req.amount > 10_000.0:
            risk_score = 0.35
            explanation_status = ExplanationStatus.STABLE
        else:
            risk_score = 0.05
            explanation_status = ExplanationStatus.STABLE

        # Tri-state gateway decision logic:
        # High score / rule triggered => candidate High Risk.
        # If High Risk candidate + Unstable explanation => Analyst Review.
        # Below low threshold => Low Risk.
        if risk_score >= 0.70 or bool(rule_flags):
            if explanation_status == ExplanationStatus.UNSTABLE:
                gateway_decision = GatewayDecision.ANALYST_REVIEW
            else:
                gateway_decision = GatewayDecision.HIGH_RISK
        elif risk_score >= 0.40:
            gateway_decision = GatewayDecision.ANALYST_REVIEW
        else:
            gateway_decision = GatewayDecision.LOW_RISK

        # Top SHAP features with verified direction consistency
        if risk_score >= 0.50:
            top_features = [
                FeatureExplanation(
                    feature="amount",
                    shap_value=0.45,
                    direction=ShapDirection.INCREASES_RISK,
                ),
                FeatureExplanation(
                    feature="type_TRANSFER",
                    shap_value=0.32,
                    direction=ShapDirection.INCREASES_RISK,
                ),
                FeatureExplanation(
                    feature="balance_delta_orig",
                    shap_value=-0.15,
                    direction=ShapDirection.DECREASES_RISK,
                ),
            ]
        else:
            top_features = [
                FeatureExplanation(
                    feature="amount",
                    shap_value=-0.30,
                    direction=ShapDirection.DECREASES_RISK,
                ),
                FeatureExplanation(
                    feature="old_balance_orig",
                    shap_value=-0.22,
                    direction=ShapDirection.DECREASES_RISK,
                ),
                FeatureExplanation(
                    feature="type_PAYMENT",
                    shap_value=-0.18,
                    direction=ShapDirection.DECREASES_RISK,
                ),
            ]

        latency = (time.perf_counter() - start_time) * 1000.0

        return PredictionOut(
            transaction_id=tx_id,
            risk_score=round(risk_score, 4),
            top_features=top_features,
            explanation_status=explanation_status,
            gateway_decision=gateway_decision,
            rule_flags=rule_flags,
            imbalance_strategy="class_weight",
            model_version="stub",
            latency_ms=round(max(latency, 1.2), 2),
        )


scoring_service = ScoringService()

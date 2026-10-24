import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import uuid

import joblib
import pandas as pd

from app.core.config import settings
from app.models.schemas import (
    ExplanationStatus,
    FeatureExplanation,
    GatewayDecision,
    PredictionOut,
    ShapDirection,
    TransactionIn,
    TransactionType,
)
from app.services.explainer import ExplainerService, RealExplainer, StubExplainer
from app.services.rules import evaluate_rules

logger = logging.getLogger(__name__)


class ScoringService:
    """Production scoring service managing ML inference, explainability, and gateway routing."""

    def __init__(self) -> None:
        self.model: Any = None
        self.replicates: List[Any] = []
        self.manifest: Dict[str, Any] = {}
        self.mode: str = "stub"
        self.model_version: str = "stub"
        self.imbalance_strategy: str = "class_weight"
        self.explainer: ExplainerService = StubExplainer()
        self.threshold_high: float = settings.FRAUD_THRESHOLD_HIGH
        self.threshold_review: float = settings.FRAUD_THRESHOLD_REVIEW

    def load_artifact(self, model_path_str: str | None) -> None:
        """Load trained model bundle or initialize deterministic stub per environment policy."""
        if not model_path_str:
            if settings.APP_ENV.lower() in ["dev", "development", "test"]:
                logger.info("MODEL_PATH is unset; initializing deterministic STUB model for dev/test.")
                self.mode = "stub"
                self.model_version = "stub"
                self.imbalance_strategy = "class_weight"
                self.explainer = StubExplainer()
                self.threshold_high = settings.FRAUD_THRESHOLD_HIGH
                self.threshold_review = settings.FRAUD_THRESHOLD_REVIEW
                return
            else:
                raise RuntimeError("MODEL_PATH must be configured in non-dev/production environments.")

        bundle_dir = Path(model_path_str)
        if bundle_dir.is_file():
            bundle_dir = bundle_dir.parent

        manifest_path = bundle_dir / "manifest.json"
        model_file = bundle_dir / "model.joblib"

        if not bundle_dir.exists() or not manifest_path.exists() or not model_file.exists():
            raise RuntimeError(
                f"Model artifact directory '{bundle_dir}' is invalid or missing required files "
                "(manifest.json, model.joblib)."
            )

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)

            self.model = joblib.load(model_file)
            self.model_version = self.manifest.get("model_version", "v1.0.0")
            self.imbalance_strategy = self.manifest.get("imbalance_strategy", "class_weight")

            # Extract calibrated thresholds from manifest
            thresholds = self.manifest.get("thresholds", {})
            self.threshold_high = float(thresholds.get("high", settings.FRAUD_THRESHOLD_HIGH))
            self.threshold_review = float(thresholds.get("review", settings.FRAUD_THRESHOLD_REVIEW))

            # Load replicates if present
            self.replicates = []
            replicates_dir = bundle_dir / "replicates"
            if replicates_dir.exists() and replicates_dir.is_dir():
                for rep_path in replicates_dir.glob("seed_*.joblib"):
                    try:
                        self.replicates.append(joblib.load(rep_path))
                    except Exception as ex:
                        logger.warning(f"Failed to load replicate at {rep_path}: {ex}")

            shap_top_k = int(self.manifest.get("shap_top_k", 3))
            stability_min_overlap = float(self.manifest.get("stability_min_overlap", 0.67))
            self.explainer = RealExplainer(
                top_k=shap_top_k,
                stability_min_overlap=stability_min_overlap,
            )
            self.mode = "trained"
            logger.info(f"Successfully loaded trained model '{self.model_version}' from {bundle_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ML artifact from {bundle_dir}: {e}") from e

    def _to_dataframe(self, tx: TransactionIn) -> pd.DataFrame:
        """Map transaction input to single-row DataFrame using snake_case API column names."""
        return pd.DataFrame([
            {
                "step": tx.step,
                "type": tx.type.value,
                "amount": tx.amount,
                "name_orig": tx.name_orig,
                "old_balance_orig": tx.old_balance_orig,
                "new_balance_orig": tx.new_balance_orig,
                "name_dest": tx.name_dest,
                "old_balance_dest": tx.old_balance_dest,
                "new_balance_dest": tx.new_balance_dest,
            }
        ])

    def route_decision(
        self,
        risk_score: float,
        rule_flags: List[str],
        explanation_status: ExplanationStatus,
    ) -> GatewayDecision:
        """Execute the deterministic gateway decision matrix.

        Matrix rules:
        - score >= high -> candidate High Risk.
        - rule flags present AND score >= review -> candidate High Risk.
        - rule flags present AND score < review -> Analyst Review.
        - candidate High Risk + Unstable -> Analyst Review; + Stable -> High Risk.
        - review <= score < high (with no rule flags) -> Analyst Review.
        - Otherwise Low Risk.
        """
        has_rules = len(rule_flags) > 0
        is_high_score = risk_score >= self.threshold_high
        is_review_score = risk_score >= self.threshold_review

        is_candidate_high_risk = is_high_score or (has_rules and is_review_score)

        if is_candidate_high_risk:
            if explanation_status == ExplanationStatus.UNSTABLE:
                return GatewayDecision.ANALYST_REVIEW
            else:
                return GatewayDecision.HIGH_RISK
        elif has_rules and not is_review_score:
            return GatewayDecision.ANALYST_REVIEW
        elif is_review_score and not is_high_score:
            return GatewayDecision.ANALYST_REVIEW
        else:
            return GatewayDecision.LOW_RISK

    def predict(self, req: TransactionIn) -> PredictionOut:
        """Score incoming transaction, compute SHAP attribution, and route gateway decision.

        Latency recorded captures both ML scoring and SHAP explanation duration.
        """
        start_time = time.perf_counter()
        tx_id = str(uuid.uuid4())

        # 1. Heuristic safety rules evaluation
        rule_flags = evaluate_rules(req)

        # 2. Score probability and compute explanations
        features_df = self._to_dataframe(req)

        if self.mode == "trained" and self.model is not None:
            proba = self.model.predict_proba(features_df)
            if hasattr(proba, "shape") and len(proba.shape) > 1 and proba.shape[1] >= 2:
                risk_score = float(proba[0][1])
            elif hasattr(proba, "shape") and len(proba.shape) > 1 and proba.shape[1] == 1:
                risk_score = float(proba[0][0])
            else:
                risk_score = float(proba[0]) if hasattr(proba, "__getitem__") else float(proba)
            top_features, explanation_status = self.explainer.explain(
                self.model,
                features_df,
                risk_score,
                self.replicates,
            )
        else:
            # Deterministic stub logic
            if "ORIGIN_ACCOUNT_DRAINED" in rule_flags or "LARGE_TRANSFER_AMOUNT" in rule_flags:
                risk_score = 0.92
            elif req.amount > 50_000.0 and req.type == TransactionType.TRANSFER:
                risk_score = 0.76
            elif req.amount > 10_000.0:
                risk_score = 0.45
            else:
                risk_score = 0.05

            top_features, explanation_status = self.explainer.explain(
                None,
                features_df,
                risk_score,
                None,
            )

        # 3. Route tri-state gateway decision
        gateway_decision = self.route_decision(
            risk_score=risk_score,
            rule_flags=rule_flags,
            explanation_status=explanation_status,
        )

        # Latency records scoring + explanation duration
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return PredictionOut(
            transaction_id=tx_id,
            risk_score=round(max(0.0, min(1.0, risk_score)), 4),
            top_features=top_features,
            explanation_status=explanation_status,
            gateway_decision=gateway_decision,
            rule_flags=rule_flags,
            imbalance_strategy=self.imbalance_strategy,
            model_version=self.model_version,
            latency_ms=round(max(latency_ms, 1.0), 2),
        )


scoring_service = ScoringService()
# Initialize default based on env
scoring_service.load_artifact(settings.MODEL_PATH)

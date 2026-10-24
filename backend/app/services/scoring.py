import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.core.config import settings
from app.models.schemas import (
    ExplanationStatus,
    GatewayDecision,
    PredictionOut,
    TransactionIn,
    TransactionType,
)
from app.services.explainer import ExplainerService, RealExplainer, StubExplainer
from app.services.rules import evaluate_rules

logger = logging.getLogger(__name__)


def validate_manifest(
    manifest: dict[str, Any],
    model: Any,
    replicate_files: list[Path],
) -> None:
    """Validate manifest.json against ML artifact contract specifications.

    Raises RuntimeError if any contract condition is violated:
    - feature_names must match model's expected features
    - thresholds must satisfy 0 < review < high < 1
    - n_replicates must match the count of replicate files found
    - shap_top_k >= 1 and 0 <= stability_min_overlap <= 1
    """
    # 1. Feature names validation
    feat_names = manifest.get("feature_names")
    if not isinstance(feat_names, list) or len(feat_names) == 0:
        raise RuntimeError("Manifest must define a non-empty 'feature_names' list.")

    # If pipeline/model declares expected input features, verify matching
    expected_model_features = None
    if hasattr(model, "feature_names_in_"):
        expected_model_features = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                expected_model_features = list(step.feature_names_in_)
                break

    if expected_model_features is not None and list(expected_model_features) != list(feat_names):
            raise RuntimeError(
                f"Manifest feature_names {feat_names} do not match pipeline expected features "
                f"{expected_model_features}."
            )

    # 2. Thresholds validation: strictly 0 < review < high < 1
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict) or "high" not in thresholds or "review" not in thresholds:
        raise RuntimeError("Manifest must contain 'thresholds' with both 'high' and 'review' keys.")

    high = float(thresholds["high"])
    review = float(thresholds["review"])
    if not (0.0 < review < high < 1.0):
        raise RuntimeError(
            f"Invalid thresholds in manifest: must satisfy 0 < review < high < 1. Got review={review}, high={high}."
        )

    # 3. n_replicates validation
    n_replicates = manifest.get("n_replicates")
    if n_replicates is None or not isinstance(n_replicates, int) or n_replicates < 0:
        raise RuntimeError(f"Manifest must define an integer 'n_replicates' >= 0, got {n_replicates}.")

    actual_file_count = len(replicate_files)
    if n_replicates != actual_file_count:
        raise RuntimeError(
            f"Manifest n_replicates ({n_replicates}) does not match number of replicate files found "
            f"({actual_file_count})."
        )

    # 4. SHAP parameters validation
    shap_top_k = manifest.get("shap_top_k", 3)
    if not isinstance(shap_top_k, int) or shap_top_k < 1:
        raise RuntimeError(f"Manifest shap_top_k must be an integer >= 1, got {shap_top_k}.")

    stability_min_overlap = manifest.get("stability_min_overlap", 0.67)
    if not isinstance(stability_min_overlap, (int, float)) or not (0.0 <= stability_min_overlap <= 1.0):
        raise RuntimeError(
            f"Manifest stability_min_overlap must be in range [0.0, 1.0], got {stability_min_overlap}."
        )


class ScoringService:
    """Production scoring service managing ML inference, explainability, and gateway routing."""

    def __init__(self) -> None:
        self.model: Any = None
        self.replicates: list[Any] = []
        self.manifest: dict[str, Any] = {}
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

            # Discover replicate files
            replicate_files: list[Path] = []
            replicates_dir = bundle_dir / "replicates"
            if replicates_dir.exists() and replicates_dir.is_dir():
                replicate_files = sorted(replicates_dir.glob("seed_*.joblib"))

            # Strict manifest validation against contract
            validate_manifest(self.manifest, self.model, replicate_files)

            self.model_version = self.manifest.get("model_version", "v1.0.0")
            self.imbalance_strategy = self.manifest.get("imbalance_strategy", "class_weight")

            # Extract calibrated thresholds from manifest
            thresholds = self.manifest["thresholds"]
            self.threshold_high = float(thresholds["high"])
            self.threshold_review = float(thresholds["review"])

            # Load replicate pipelines
            self.replicates = []
            for rep_path in replicate_files:
                try:
                    self.replicates.append(joblib.load(rep_path))
                except (OSError, KeyError, ValueError, TypeError) as ex:
                    logger.warning("Failed to load replicate at %s: %s", rep_path, ex)

            shap_top_k = int(self.manifest.get("shap_top_k", 3))
            stability_min_overlap = float(self.manifest.get("stability_min_overlap", 0.67))
            self.explainer = RealExplainer(
                top_k=shap_top_k,
                stability_min_overlap=stability_min_overlap,
            )
            self.mode = "trained"
            logger.info("Successfully loaded trained model '%s' from %s", self.model_version, bundle_dir)
        except RuntimeError:
            raise
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
        rule_flags: list[str],
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
        elif has_rules and not is_review_score or is_review_score and not is_high_score:
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

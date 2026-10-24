import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import ValidationError

from app.core.config import settings
from app.models.manifest import ManifestSchema
from app.models.schemas import (
    ExplanationStatus,
    FeatureExplanation,
    GatewayDecision,
    PredictionOut,
    TransactionIn,
)
from app.services.explainer import (
    ExplainerService,
    RealExplainer,
    StubExplainer,
    validate_model_supported,
)
from app.services.rules import evaluate_rules

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    """Raised when transaction scoring is attempted before a model artifact has been loaded."""


def extract_pipeline_feature_names(model: Any) -> list[str]:
    """Extract input feature column names expected by the model/pipeline.

    Fails startup with RuntimeError if pipeline input columns cannot be determined.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    raise RuntimeError(
        "Model pipeline does not expose input feature names ('feature_names_in_'). "
        "Pipeline input columns cannot be determined."
    )


def make_sample_row_from_features(feature_names: list[str]) -> pd.DataFrame:
    """Build a deterministic synthetic row conforming to the manifest feature names."""
    data = {}
    for col in feature_names:
        if col == "type":
            data[col] = ["TRANSFER"]
        elif "name" in col.lower():
            data[col] = ["C123456789"]
        else:
            data[col] = [0.0]
    return pd.DataFrame(data)


def run_startup_self_test(model: Any, feature_names: list[str]) -> None:
    """Score a single sample row built from the manifest and check probability is in [0, 1]."""
    sample_df = make_sample_row_from_features(feature_names)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample_df)
            if (
                hasattr(proba, "shape")
                and len(proba.shape) >= 2
                and proba.shape[1] >= 2
            ):
                score = float(proba[0][1])
            elif hasattr(proba, "shape") and len(proba.shape) == 1:
                score = float(proba[0])
            else:
                score = float(proba[0][0])
        else:
            pred = model.predict(sample_df)
            score = float(pred[0])

        if not (0.0 <= score <= 1.0):
            raise RuntimeError(
                f"Startup self-test produced score {score} outside [0.0, 1.0]."
            )
    except Exception as e:
        raise RuntimeError(
            f"Startup self-test failed: pipeline inference error: {e}"
        ) from e


def validate_manifest(
    manifest_data: dict[str, Any],
    model: Any,
    replicate_files: list[Path],
) -> ManifestSchema:
    """Validate manifest.json against ML artifact contract using ManifestSchema.

    Fails fast with RuntimeError on any contract violation.
    """
    try:
        parsed = ManifestSchema.model_validate(manifest_data)
    except ValidationError as ve:
        raise RuntimeError(f"Manifest schema validation failed: {ve}") from ve

    # 1. Supported model type check
    validate_model_supported(model)

    # 2. Pipeline feature column detection
    expected_model_features = extract_pipeline_feature_names(model)
    if list(expected_model_features) != list(parsed.feature_names):
        raise RuntimeError(
            f"Manifest feature_names {parsed.feature_names} do not match pipeline expected features "
            f"{expected_model_features}."
        )

    # 3. Replicate count matching
    actual_file_count = len(replicate_files)
    if parsed.n_replicates != actual_file_count:
        raise RuntimeError(
            f"Manifest n_replicates ({parsed.n_replicates}) does not match number of replicate files found "
            f"({actual_file_count})."
        )

    return parsed


class ScoringService:
    """Production scoring service managing ML inference, explainability, and gateway routing."""

    def __init__(self) -> None:
        self.model: Any = None
        self.replicates: list[Any] = []
        self.manifest: dict[str, Any] = {}
        self.mode: str = "unloaded"
        self.model_version: str = "none"
        self.imbalance_strategy: str = "class_weight"
        self.explainer: ExplainerService | None = None
        self.threshold_high: float = settings.FRAUD_THRESHOLD_HIGH
        self.threshold_review: float = settings.FRAUD_THRESHOLD_REVIEW

    def reset(self) -> None:
        """Reset service state back to uninitialized unloaded mode."""
        self.model = None
        self.replicates = []
        self.manifest = {}
        self.mode = "unloaded"
        self.model_version = "none"
        self.imbalance_strategy = "class_weight"
        self.explainer = None
        self.threshold_high = settings.FRAUD_THRESHOLD_HIGH
        self.threshold_review = settings.FRAUD_THRESHOLD_REVIEW

    def load_artifact(self, model_path_str: str | None) -> None:
        """Load trained model bundle or initialize deterministic stub per environment policy."""
        if not model_path_str:
            if settings.APP_ENV.lower() in ["dev", "development", "test"]:
                logger.info(
                    "MODEL_PATH is unset; initializing deterministic STUB model for dev/test."
                )
                self.mode = "stub"
                self.model_version = "stub"
                self.imbalance_strategy = "class_weight"
                self.explainer = StubExplainer()
                self.threshold_high = settings.FRAUD_THRESHOLD_HIGH
                self.threshold_review = settings.FRAUD_THRESHOLD_REVIEW
                return
            else:
                raise RuntimeError(
                    "MODEL_PATH must be configured in non-dev/production environments."
                )

        bundle_dir = Path(model_path_str)
        if bundle_dir.is_file():
            bundle_dir = bundle_dir.parent

        manifest_path = bundle_dir / "manifest.json"
        model_file = bundle_dir / "model.joblib"

        if (
            not bundle_dir.exists()
            or not manifest_path.exists()
            or not model_file.exists()
        ):
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
            parsed_manifest = validate_manifest(
                self.manifest, self.model, replicate_files
            )

            # Startup self-test
            run_startup_self_test(self.model, parsed_manifest.feature_names)

            self.model_version = parsed_manifest.model_version
            self.imbalance_strategy = parsed_manifest.imbalance_strategy.value

            # Extract calibrated thresholds from manifest
            self.threshold_high = parsed_manifest.thresholds.high
            self.threshold_review = parsed_manifest.thresholds.review

            # Load replicate pipelines fail-closed: missing/unloadable replicates stop the app
            self.replicates = []
            for rep_path in replicate_files:
                try:
                    rep_model = joblib.load(rep_path)
                    validate_model_supported(rep_model)
                    self.replicates.append(rep_model)
                except Exception as ex:
                    raise RuntimeError(
                        f"Failed to load replicate pipeline from {rep_path}: {ex}"
                    ) from ex

            if len(self.replicates) < 2:
                raise RuntimeError(
                    f"ML artifact requires at least 2 replicates, found {len(self.replicates)}."
                )

            self.explainer = RealExplainer(
                top_k=parsed_manifest.shap_top_k,
                stability_min_overlap=parsed_manifest.stability_min_overlap,
            )
            self.mode = "trained"
            logger.info(
                "Successfully loaded trained model '%s' from %s",
                self.model_version,
                bundle_dir,
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ML artifact from {bundle_dir}: {e}"
            ) from e

    def route_decision(
        self,
        risk_score: float,
        rule_flags: list[str],
        explanation_status: ExplanationStatus,
    ) -> GatewayDecision:
        """Route transaction using calibrated thresholds, heuristic flags, and explanation stability."""
        has_rules = len(rule_flags) > 0
        candidate_high_risk = False

        if (
            risk_score >= self.threshold_high
            or has_rules
            and risk_score >= self.threshold_review
        ):
            candidate_high_risk = True

        if candidate_high_risk:
            if explanation_status == ExplanationStatus.STABLE:
                return GatewayDecision.HIGH_RISK
            else:
                return GatewayDecision.ANALYST_REVIEW

        if has_rules and risk_score < self.threshold_review:
            return GatewayDecision.ANALYST_REVIEW

        if self.threshold_review <= risk_score < self.threshold_high:
            return GatewayDecision.ANALYST_REVIEW

        return GatewayDecision.LOW_RISK

    def _to_dataframe(self, tx: TransactionIn) -> pd.DataFrame:
        """Map API TransactionIn model to snake_case DataFrame input."""
        return pd.DataFrame(
            [
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
            ]
        )

    def _predict_stub(
        self,
        tx: TransactionIn,
        features_df: pd.DataFrame,
    ) -> tuple[float, list[FeatureExplanation], ExplanationStatus]:
        """Compute heuristic probability and deterministic explanations for stub mode."""
        score = 0.05
        if tx.type.value in ["TRANSFER", "CASH_OUT"]:
            score += 0.25
        if tx.amount > 200_000.0:
            score += 0.35
        if tx.old_balance_orig > 0 and tx.new_balance_orig == 0:
            score += 0.30

        score = min(max(round(score, 4), 0.0), 1.0)
        top_features, explanation_status = self.explainer.explain(
            model=None,
            features_df=features_df,
            risk_score=score,
            replicates=None,
        )
        return score, top_features, explanation_status

    def _predict_trained(
        self,
        tx: TransactionIn,
        features_df: pd.DataFrame,
    ) -> tuple[float, list[FeatureExplanation], ExplanationStatus]:
        """Score input DataFrame against the trained sklearn Pipeline and explain via SHAP."""
        proba = self.model.predict_proba(features_df)
        if hasattr(proba, "shape") and len(proba.shape) >= 2 and proba.shape[1] >= 2:
            score = float(proba[0][1])
        elif hasattr(proba, "shape") and len(proba.shape) == 1:
            score = float(proba[0])
        else:
            score = float(proba[0][0])
        score = min(max(round(score, 4), 0.0), 1.0)

        top_features, explanation_status = self.explainer.explain(
            model=self.model,
            features_df=features_df,
            risk_score=score,
            replicates=self.replicates,
        )
        return score, top_features, explanation_status

    def predict(
        self, tx: TransactionIn, transaction_id: str | None = None
    ) -> PredictionOut:
        """Execute full scoring lifecycle: heuristic rules, ML inference, explanation, and gateway routing."""
        if self.mode == "unloaded" or self.model_version == "none":
            raise ModelNotLoadedError(
                "Model artifact has not been loaded. Service is not ready to score transactions."
            )
        if not transaction_id:
            import uuid

            transaction_id = str(uuid.uuid4())
        rule_flags = evaluate_rules(tx)
        features_df = self._to_dataframe(tx)

        if self.mode == "trained" and self.model is not None:
            score, top_features, explanation_status = self._predict_trained(
                tx, features_df
            )
        else:
            score, top_features, explanation_status = self._predict_stub(
                tx, features_df
            )

        decision = self.route_decision(score, rule_flags, explanation_status)

        return PredictionOut(
            transaction_id=transaction_id,
            risk_score=score,
            top_features=top_features,
            explanation_status=explanation_status,
            gateway_decision=decision,
            rule_flags=rule_flags,
            imbalance_strategy=self.imbalance_strategy,
            model_version=self.model_version,
            latency_ms=0.0,
        )


scoring_service = ScoringService()

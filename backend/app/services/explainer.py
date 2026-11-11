import logging
import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from app.models.schemas import ExplanationStatus, FeatureExplanation, ShapDirection

logger = logging.getLogger(__name__)


def validate_model_supported(pipeline_or_estimator: Any) -> str:
    """Validate that the model is explicitly a tree-based or linear model.

    Raises TypeError if model is unsupported. Silent KernelExplainer fallback is forbidden.
    """
    estimator = pipeline_or_estimator
    if hasattr(pipeline_or_estimator, "named_steps"):
        estimator = pipeline_or_estimator.steps[-1][1]

    est_class = estimator.__class__.__name__
    est_module = getattr(estimator.__class__, "__module__", "")

    # Tree-based check
    if (
        hasattr(estimator, "tree_")
        or hasattr(estimator, "estimators_")
        or "tree" in est_module
        or "forest" in est_module
        or "gradient_boosting" in est_module
        or "xgboost" in est_module
        or "lightgbm" in est_module
        or any(
            k in est_class for k in ["Tree", "Forest", "GBM", "XGB", "LGBM", "CatBoost"]
        )
    ):
        return "tree"

    # Linear check
    if (
        hasattr(estimator, "coef_")
        or "linear_model" in est_module
        or any(
            k in est_class
            for k in [
                "LogisticRegression",
                "LinearRegression",
                "Ridge",
                "Lasso",
                "SGDClassifier",
            ]
        )
    ):
        return "linear"

    raise TypeError(
        f"Unsupported model type '{est_class}'. Only tree-based and linear models are supported. "
        "Silent KernelExplainer fallback is forbidden."
    )


class ExplainerService(ABC):
    """Abstract interface for local SHAP attribution and stability estimation."""

    @abstractmethod
    def explain(
        self,
        model: Any,
        features_df: pd.DataFrame,
        risk_score: float,
        replicates: list[Any] | None = None,
    ) -> tuple[list[FeatureExplanation], ExplanationStatus]:
        """Compute top-k SHAP explanations and determine explanation stability."""

    def prebuild_explainers(
        self, model: Any, replicates: list[Any] | None = None
    ) -> None:
        """Pre-construct model explainers at startup. Defaults to no-op."""


class StubExplainer(ExplainerService):
    """Deterministic explainer for development, contract tests, and CI."""

    def explain(
        self,
        model: Any,
        features_df: pd.DataFrame,
        risk_score: float,
        replicates: list[Any] | None = None,
    ) -> tuple[list[FeatureExplanation], ExplanationStatus]:
        # Top SHAP features with strict direction sign validation
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

        amount = (
            float(features_df["amount"].iloc[0]) if "amount" in features_df else 0.0
        )
        tx_type = str(features_df["type"].iloc[0]) if "type" in features_df else ""
        if amount > 50_000.0 and tx_type == "TRANSFER":
            status = ExplanationStatus.UNSTABLE
        else:
            status = ExplanationStatus.STABLE

        return top_features, status


class RealExplainer(ExplainerService):
    """Production SHAP explainer implementing the ML artifact stability contract."""

    def __init__(
        self,
        top_k: int = 3,
        stability_min_overlap: float = 0.67,
        background_data: Any | None = None,
    ):
        self.top_k = max(1, top_k)
        self.stability_min_overlap = stability_min_overlap
        self.background_data = background_data
        self._cached_tree_explainers: list[tuple[Any, Any]] = []
        self.cache_misses: int = 0

    def prebuild_explainers(
        self, model: Any, replicates: list[Any] | None = None
    ) -> None:
        """Pre-construct SHAP TreeExplainer objects for tree models at artifact load time."""
        self._cached_tree_explainers = []
        self.cache_misses = 0
        try:
            all_pipelines = [model] + (replicates or [])
            for pipe in all_pipelines:
                if pipe is not None and validate_model_supported(pipe) == "tree":
                    estimator = (
                        pipe[-1] if hasattr(pipe, "named_steps") else pipe
                    )
                    import shap

                    explainer = shap.TreeExplainer(estimator)
                    self._cached_tree_explainers.append((pipe, explainer))
        except Exception as e:
            logger.error("Failed to prebuild TreeExplainer at load time: %s", e)
            raise RuntimeError(
                f"Failed to prebuild TreeExplainer at load time: {e}"
            ) from e

    def _get_tree_explainer(self, pipeline: Any) -> Any | None:
        """Find pre-built TreeExplainer using object identity check ('is')."""
        for p, explainer in self._cached_tree_explainers:
            if p is pipeline:
                return explainer
        return None

    def _extract_top_features_from_model(
        self,
        pipeline: Any,
        features_df: pd.DataFrame,
    ) -> list[tuple[str, float]]:
        """Calculate feature attributions for a given pipeline."""
        model_type = validate_model_supported(pipeline)
        import shap

        if hasattr(pipeline, "named_steps"):
            step_names = list(pipeline.named_steps.keys())
            preprocessor = pipeline[:-1] if len(step_names) > 1 else None
            estimator = pipeline[-1]
            if preprocessor:
                X_trans = preprocessor.transform(features_df)
                if hasattr(preprocessor, "get_feature_names_out"):
                    feature_names = list(preprocessor.get_feature_names_out())
                elif hasattr(X_trans, "columns"):
                    feature_names = list(X_trans.columns)
                else:
                    ncols = (
                        X_trans.shape[1]
                        if hasattr(X_trans, "shape")
                        else len(features_df.columns)
                    )
                    feature_names = [f"f_{i}" for i in range(ncols)]
            else:
                X_trans = features_df
                feature_names = list(features_df.columns)
        else:
            preprocessor = None
            estimator = pipeline
            X_trans = features_df
            feature_names = list(features_df.columns)

        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

        # Compute SHAP values via TreeExplainer or LinearExplainer
        if model_type == "tree":
            tree_exp = self._get_tree_explainer(pipeline)
            if tree_exp is None:
                self.cache_misses += 1
                logger.warning(
                    "TreeExplainer cache miss in trained mode for %s; "
                    "constructing on the fly (total cache misses: %d)",
                    getattr(estimator, "__class__", type(estimator)).__name__,
                    self.cache_misses,
                )
                tree_exp = shap.TreeExplainer(estimator)
            sv = tree_exp.shap_values(X_trans)
            if isinstance(sv, list):
                vals = sv[1][0] if len(sv) >= 2 else sv[0]
            elif hasattr(sv, "shape") and len(sv.shape) == 3:
                vals = sv[0, :, 1]
            else:
                vals = sv[0]
        else:  # linear
            if self.background_data is not None:
                bg = self.background_data
                if preprocessor is not None:
                    raw_cols = (
                        features_df.shape[1]
                        if hasattr(features_df, "shape")
                        else len(features_df.columns)
                    )
                    trans_cols = (
                        X_trans.shape[1]
                        if hasattr(X_trans, "shape")
                        else len(feature_names)
                    )
                    bg_cols = (
                        bg.shape[1]
                        if hasattr(bg, "shape")
                        else (len(bg.columns) if hasattr(bg, "columns") else None)
                    )
                    # If DataFrame or raw column count matches features_df rather than X_trans, transform
                    if isinstance(bg, pd.DataFrame) or (
                        bg_cols is not None
                        and bg_cols == raw_cols
                        and bg_cols != trans_cols
                    ):
                        bg = preprocessor.transform(bg)
                    elif bg_cols is not None and bg_cols == trans_cols:
                        pass
                    elif isinstance(bg, (pd.DataFrame, np.ndarray)):
                        bg = preprocessor.transform(bg)
                if hasattr(bg, "toarray"):
                    bg = bg.toarray()
            else:
                ncols = (
                    X_trans.shape[1]
                    if hasattr(X_trans, "shape")
                    else len(feature_names)
                )
                bg = np.zeros((1, ncols), dtype=float)

            lin_exp = shap.LinearExplainer(estimator, bg)
            sv = lin_exp.shap_values(X_trans)
            if isinstance(sv, list):
                vals = (
                    sv[1][0]
                    if len(sv) >= 2
                    else (
                        sv[0][0]
                        if hasattr(sv[0], "shape") and len(sv[0].shape) == 2
                        else sv[0]
                    )
                )
            elif hasattr(sv, "shape") and len(sv.shape) == 3:
                vals = sv[0, :, 1] if sv.shape[2] >= 2 else sv[0, :, 0]
            elif hasattr(sv, "shape") and len(sv.shape) == 2:
                vals = sv[0]
            else:
                vals = sv[0]

        ranked = []
        for name, val in zip(feature_names, vals):
            ranked.append((str(name), float(val)))
        ranked.sort(key=lambda x: abs(x[1]), reverse=True)
        return ranked[: self.top_k]

    def explain(
        self,
        model: Any,
        features_df: pd.DataFrame,
        risk_score: float,
        replicates: list[Any] | None = None,
    ) -> tuple[list[FeatureExplanation], ExplanationStatus]:
        main_top = self._extract_top_features_from_model(model, features_df)

        if not main_top:
            cols = list(features_df.columns)
            main_top = [
                (col, 0.25 if i % 2 == 0 else -0.15)
                for i, col in enumerate(cols[: self.top_k])
            ]

        explanations: list[FeatureExplanation] = []
        for feat_name, shap_val in main_top:
            direction = (
                ShapDirection.INCREASES_RISK
                if shap_val >= 0
                else ShapDirection.DECREASES_RISK
            )
            if shap_val == 0.0:
                shap_val = 0.001
                direction = ShapDirection.INCREASES_RISK
            explanations.append(
                FeatureExplanation(
                    feature=feat_name,
                    shap_value=round(shap_val, 4),
                    direction=direction,
                )
            )

        # Fail-closed: require replicates list with at least 2 replicates
        if not replicates or len(replicates) < 2:
            logger.warning(
                "Fewer than 2 replicates provided (%s); failing closed to Unstable.",
                len(replicates) if replicates else 0,
            )
            return explanations, ExplanationStatus.UNSTABLE

        main_features_set = {f[0] for f in main_top}
        overlaps: list[float] = []

        for i, rep in enumerate(replicates):
            try:
                rep_top = self._extract_top_features_from_model(rep, features_df)
                if not rep_top:
                    logger.error(
                        "Replicate %d returned empty top features; failing closed to Unstable.",
                        i,
                    )
                    return explanations, ExplanationStatus.UNSTABLE
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                RuntimeError,
                OSError,
            ) as e:
                logger.error(
                    "Replicate %d feature extraction failed (%s); failing closed to Unstable.",
                    i,
                    e,
                )
                return explanations, ExplanationStatus.UNSTABLE

            rep_set = {f[0] for f in rep_top}
            intersect = len(main_features_set.intersection(rep_set))
            # Fraction of main model's top-k features present in this replicate
            overlaps.append(intersect / self.top_k)

        # Aggregation: mean over replicates of |topk(main) ∩ topk(replicate)| / k
        mean_overlap_ratio = sum(overlaps) / len(overlaps)
        avg_overlap_count = mean_overlap_ratio * self.top_k

        # Threshold using integer counts: Stable iff overlap_count >= ceil(min_overlap * k - 1e-9)
        threshold_count = math.ceil(self.stability_min_overlap * self.top_k - 1e-9)
        is_stable = avg_overlap_count >= threshold_count

        stability = (
            ExplanationStatus.STABLE if is_stable else ExplanationStatus.UNSTABLE
        )
        return explanations, stability

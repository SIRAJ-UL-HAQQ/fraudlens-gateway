import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app.models.schemas import (
    ExplanationStatus,
    GatewayDecision,
    TransactionIn,
    TransactionType,
)
from app.services.scoring import (
    ScoringService,
    extract_pipeline_feature_names,
    run_startup_self_test,
    validate_manifest,
)


@pytest.fixture
def scoring_svc():
    svc = ScoringService()
    svc.threshold_high = 0.70
    svc.threshold_review = 0.40
    return svc


def test_decision_matrix_all_cells(scoring_svc: ScoringService):
    """Verify all cells of the decision matrix:
    - score >= high -> candidate High Risk.
    - rule flags present AND score >= review -> candidate High Risk.
    - rule flags present AND score < review -> Analyst Review.
    - candidate High Risk + Unstable -> Analyst Review; + Stable -> High Risk.
    - review <= score < high (with no rule flags) -> Analyst Review.
    - Otherwise Low Risk.
    """
    # 1. High score (>=0.70), no rules, Stable -> High Risk
    assert (
        scoring_svc.route_decision(0.85, [], ExplanationStatus.STABLE)
        == GatewayDecision.HIGH_RISK
    )

    # 2. High score (>=0.70), no rules, Unstable -> Analyst Review
    assert (
        scoring_svc.route_decision(0.85, [], ExplanationStatus.UNSTABLE)
        == GatewayDecision.ANALYST_REVIEW
    )

    # 3. Rules present, score >= review (0.50 >= 0.40), Stable -> High Risk
    assert (
        scoring_svc.route_decision(
            0.50, ["LARGE_TRANSFER_AMOUNT"], ExplanationStatus.STABLE
        )
        == GatewayDecision.HIGH_RISK
    )

    # 4. Rules present, score >= review (0.50 >= 0.40), Unstable -> Analyst Review
    assert (
        scoring_svc.route_decision(
            0.50, ["LARGE_TRANSFER_AMOUNT"], ExplanationStatus.UNSTABLE
        )
        == GatewayDecision.ANALYST_REVIEW
    )

    # 5. Rules present, score < review (0.20 < 0.40), Stable -> Analyst Review
    assert (
        scoring_svc.route_decision(
            0.20, ["LARGE_TRANSFER_AMOUNT"], ExplanationStatus.STABLE
        )
        == GatewayDecision.ANALYST_REVIEW
    )

    # 6. Rules present, score < review (0.20 < 0.40), Unstable -> Analyst Review
    assert (
        scoring_svc.route_decision(
            0.20, ["LARGE_TRANSFER_AMOUNT"], ExplanationStatus.UNSTABLE
        )
        == GatewayDecision.ANALYST_REVIEW
    )

    # 7. Review <= score < high (0.40 <= 0.55 < 0.70), no rules, Stable -> Analyst Review
    assert (
        scoring_svc.route_decision(0.55, [], ExplanationStatus.STABLE)
        == GatewayDecision.ANALYST_REVIEW
    )

    # 8. Review <= score < high (0.40 <= 0.55 < 0.70), no rules, Unstable -> Analyst Review
    assert (
        scoring_svc.route_decision(0.55, [], ExplanationStatus.UNSTABLE)
        == GatewayDecision.ANALYST_REVIEW
    )

    # 9. Low score (<0.40), no rules, Stable -> Low Risk
    assert (
        scoring_svc.route_decision(0.15, [], ExplanationStatus.STABLE)
        == GatewayDecision.LOW_RISK
    )

    # 10. Low score (<0.40), no rules, Unstable -> Low Risk
    assert (
        scoring_svc.route_decision(0.15, [], ExplanationStatus.UNSTABLE)
        == GatewayDecision.LOW_RISK
    )


def test_invalid_model_path_raises_at_startup():
    svc = ScoringService()
    with pytest.raises(RuntimeError) as exc_info:
        svc.load_artifact("non_existent_path_to_model_bundle")
    assert "invalid or missing" in str(exc_info.value).lower()


# -------------------------------------------------------------------------
# Manifest Validation Tests
# -------------------------------------------------------------------------
def test_manifest_validation_thresholds():
    """Manifest thresholds must strictly satisfy 0 < review < high < 1."""
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = np.array([0, 1])
    clf = DecisionTreeClassifier().fit(X, y)

    manifest: dict[str, Any] = {
        "model_version": "v1.0.0",
        "imbalance_strategy": "class_weight",
        "feature_names": ["f1", "f2"],
        "thresholds": {"high": 0.50, "review": 0.60},  # review > high
        "n_replicates": 2,
        "shap_top_k": 2,
        "stability_min_overlap": 0.67,
    }
    with pytest.raises(RuntimeError, match="must satisfy 0 < review < high < 1"):
        validate_manifest(manifest, clf, [Path("seed_1.joblib"), Path("seed_2.joblib")])

    manifest["thresholds"] = {"high": 1.1, "review": 0.5}  # high >= 1
    with pytest.raises(RuntimeError, match="less than 1"):
        validate_manifest(manifest, clf, [Path("seed_1.joblib"), Path("seed_2.joblib")])

    manifest["thresholds"] = {"high": 0.8, "review": -0.1}  # review <= 0
    with pytest.raises(RuntimeError, match="greater than 0"):
        validate_manifest(manifest, clf, [Path("seed_1.joblib"), Path("seed_2.joblib")])


def test_manifest_validation_forbids_extra_fields():
    """ManifestSchema extra='forbid' rejects unexpected fields."""
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = np.array([0, 1])
    clf = DecisionTreeClassifier().fit(X, y)

    manifest: dict[str, Any] = {
        "model_version": "v1.0.0",
        "imbalance_strategy": "class_weight",
        "feature_names": ["f1", "f2"],
        "thresholds": {"high": 0.80, "review": 0.40},
        "n_replicates": 2,
        "shap_top_k": 2,
        "stability_min_overlap": 0.67,
        "unexpected_field": "disallowed",
    }
    with pytest.raises(RuntimeError, match="Extra inputs are not permitted"):
        validate_manifest(manifest, clf, [Path("seed_1.joblib"), Path("seed_2.joblib")])


def test_manifest_validation_strategy_enum():
    """Manifest imbalance_strategy must be in ('none', 'class_weight', 'smote_enn')."""
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = np.array([0, 1])
    clf = DecisionTreeClassifier().fit(X, y)

    manifest: dict[str, Any] = {
        "model_version": "v1.0.0",
        "imbalance_strategy": "invalid_strategy",
        "feature_names": ["f1", "f2"],
        "thresholds": {"high": 0.80, "review": 0.40},
        "n_replicates": 2,
        "shap_top_k": 2,
        "stability_min_overlap": 0.67,
    }
    with pytest.raises(
        RuntimeError, match="Input should be 'none', 'class_weight' or 'smote_enn'"
    ):
        validate_manifest(manifest, clf, [Path("seed_1.joblib"), Path("seed_2.joblib")])


def test_manifest_validation_n_replicates_requires_ge_two():
    """Manifest requires n_replicates >= 2."""
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = np.array([0, 1])
    clf = DecisionTreeClassifier().fit(X, y)

    manifest: dict[str, Any] = {
        "model_version": "v1.0.0",
        "imbalance_strategy": "class_weight",
        "feature_names": ["f1", "f2"],
        "thresholds": {"high": 0.80, "review": 0.40},
        "n_replicates": 1,  # < 2
        "shap_top_k": 2,
        "stability_min_overlap": 0.67,
    }
    with pytest.raises(RuntimeError, match="greater than or equal to 2"):
        validate_manifest(manifest, clf, [Path("seed_1.joblib")])


def test_manifest_validation_n_replicates_mismatch(tmp_path: Path):
    """Manifest n_replicates must match the count of replicate files found on disk."""
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = np.array([0, 1])
    clf = DecisionTreeClassifier().fit(X, y)

    manifest: dict[str, Any] = {
        "model_version": "v1.0.0",
        "imbalance_strategy": "class_weight",
        "feature_names": ["f1", "f2"],
        "thresholds": {"high": 0.80, "review": 0.40},
        "n_replicates": 3,
        "shap_top_k": 2,
        "stability_min_overlap": 0.67,
    }
    files = [tmp_path / "seed_42.joblib", tmp_path / "seed_43.joblib"]
    with pytest.raises(
        RuntimeError, match="does not match number of replicate files found"
    ):
        validate_manifest(manifest, clf, files)


def test_manifest_validation_feature_names_mismatch():
    """Manifest feature_names must match model's expected features when available."""
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = np.array([0, 1])
    clf = DecisionTreeClassifier().fit(X, y)

    manifest: dict[str, Any] = {
        "model_version": "v1.0.0",
        "imbalance_strategy": "class_weight",
        "feature_names": ["col_a", "col_b"],
        "thresholds": {"high": 0.80, "review": 0.40},
        "n_replicates": 2,
        "shap_top_k": 2,
        "stability_min_overlap": 0.67,
    }
    with pytest.raises(RuntimeError, match="do not match pipeline expected features"):
        validate_manifest(manifest, clf, [Path("seed_1.joblib"), Path("seed_2.joblib")])


def test_extract_pipeline_feature_names_fails_if_undetectable():
    """Fails startup if pipeline input columns cannot be determined."""

    class DummyModelWithoutFeatures:
        pass

    with pytest.raises(
        RuntimeError, match="Pipeline input columns cannot be determined"
    ):
        extract_pipeline_feature_names(DummyModelWithoutFeatures())


def test_startup_self_test():
    """Startup self-test scores one row from manifest and checks probability is in [0, 1]."""
    X = pd.DataFrame({"f1": [1.0, 0.0], "f2": [2.0, 0.0]})
    y = np.array([1, 0])
    clf = DecisionTreeClassifier().fit(X, y)

    # Valid model succeeds
    run_startup_self_test(clf, ["f1", "f2"])

    # Model returning invalid score fails
    class OutOfBoundsModel:
        def predict_proba(self, df):
            return np.array([[1.5, -0.5]])

    with pytest.raises(RuntimeError, match="outside \\[0.0, 1.0\\]"):
        run_startup_self_test(OutOfBoundsModel(), ["f1", "f2"])


def test_trained_model_pipeline_fixture(tmp_path: Path):
    """End-to-end test loading and predicting using a trained artifact fixture conforming to docs/ml-artifact-contract.md."""
    bundle_dir = tmp_path / "class_weight"
    bundle_dir.mkdir(parents=True)
    replicates_dir = bundle_dir / "replicates"
    replicates_dir.mkdir()

    feature_cols = [
        "step",
        "type",
        "amount",
        "name_orig",
        "old_balance_orig",
        "new_balance_orig",
        "name_dest",
        "old_balance_dest",
        "new_balance_dest",
    ]
    numeric_cols = [
        "step",
        "amount",
        "old_balance_orig",
        "new_balance_orig",
        "old_balance_dest",
        "new_balance_dest",
    ]

    preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", numeric_cols)],
        remainder="drop",
    )
    clf = DecisionTreeClassifier(random_state=42)
    pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])

    X_train = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "name_orig": "C1",
                "old_balance_orig": 100.0,
                "new_balance_orig": 0.0,
                "name_dest": "C2",
                "old_balance_dest": 0.0,
                "new_balance_dest": 100.0,
            },
            {
                "step": 2,
                "type": "PAYMENT",
                "amount": 20.0,
                "name_orig": "C3",
                "old_balance_orig": 50.0,
                "new_balance_orig": 30.0,
                "name_dest": "M4",
                "old_balance_dest": 0.0,
                "new_balance_dest": 0.0,
            },
        ]
    )
    y_train = np.array([1, 0])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, bundle_dir / "model.joblib")
    joblib.dump(pipeline, replicates_dir / "seed_42.joblib")
    joblib.dump(pipeline, replicates_dir / "seed_43.joblib")

    manifest = {
        "model_version": "v1.0.0-test",
        "imbalance_strategy": "class_weight",
        "feature_names": feature_cols,
        "thresholds": {"high": 0.75, "review": 0.35},
        "shap_top_k": 3,
        "stability_min_overlap": 0.60,
        "n_replicates": 2,
    }
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    svc = ScoringService()
    svc.load_artifact(str(bundle_dir))
    assert svc.mode == "trained"
    assert svc.model_version == "v1.0.0-test"
    assert svc.threshold_high == 0.75
    assert svc.threshold_review == 0.35

    tx = TransactionIn(
        step=1,
        type=TransactionType.TRANSFER,
        amount=500.0,
        name_orig="C111",
        old_balance_orig=1000.0,
        new_balance_orig=500.0,
        name_dest="C222",
        old_balance_dest=0.0,
        new_balance_dest=500.0,
    )
    pred = svc.predict(tx, transaction_id="test-tx-uuid-1234")
    assert pred.model_version == "v1.0.0-test"
    assert len(pred.top_features) > 0

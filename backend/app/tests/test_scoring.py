import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app.models.schemas import (
    ExplanationStatus,
    GatewayDecision,
    TransactionIn,
    TransactionType,
)
from app.services.explainer import RealExplainer
from app.services.scoring import ScoringService, validate_manifest


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
        scoring_svc.route_decision(0.50, ["LARGE_TRANSFER_AMOUNT"], ExplanationStatus.STABLE)
        == GatewayDecision.HIGH_RISK
    )

    # 4. Rules present, score >= review (0.50 >= 0.40), Unstable -> Analyst Review
    assert (
        scoring_svc.route_decision(0.50, ["LARGE_TRANSFER_AMOUNT"], ExplanationStatus.UNSTABLE)
        == GatewayDecision.ANALYST_REVIEW
    )

    # 5. Rules present, score < review (0.20 < 0.40), Stable -> Analyst Review
    assert (
        scoring_svc.route_decision(0.20, ["LARGE_TRANSFER_AMOUNT"], ExplanationStatus.STABLE)
        == GatewayDecision.ANALYST_REVIEW
    )

    # 6. Rules present, score < review (0.20 < 0.40), Unstable -> Analyst Review
    assert (
        scoring_svc.route_decision(0.20, ["LARGE_TRANSFER_AMOUNT"], ExplanationStatus.UNSTABLE)
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
    clf = DecisionTreeClassifier()
    manifest: dict[str, Any] = {
        "feature_names": ["f1", "f2"],
        "thresholds": {"high": 0.50, "review": 0.60},  # review > high
        "n_replicates": 0,
        "shap_top_k": 2,
        "stability_min_overlap": 0.67,
    }
    with pytest.raises(RuntimeError, match="must satisfy 0 < review < high < 1"):
        validate_manifest(manifest, clf, [])

    manifest["thresholds"] = {"high": 1.1, "review": 0.5}  # high >= 1
    with pytest.raises(RuntimeError, match="must satisfy 0 < review < high < 1"):
        validate_manifest(manifest, clf, [])

    manifest["thresholds"] = {"high": 0.8, "review": -0.1}  # review <= 0
    with pytest.raises(RuntimeError, match="must satisfy 0 < review < high < 1"):
        validate_manifest(manifest, clf, [])


def test_manifest_validation_n_replicates_mismatch(tmp_path: Path):
    """Manifest n_replicates must match the count of replicate files found."""
    clf = DecisionTreeClassifier()
    manifest: dict[str, Any] = {
        "feature_names": ["f1", "f2"],
        "thresholds": {"high": 0.80, "review": 0.40},
        "n_replicates": 3,
        "shap_top_k": 2,
        "stability_min_overlap": 0.67,
    }
    dummy_file = tmp_path / "seed_42.joblib"
    with pytest.raises(RuntimeError, match="does not match number of replicate files"):
        validate_manifest(manifest, clf, [dummy_file])


def test_manifest_validation_feature_names_mismatch():
    """Manifest feature_names must match model's expected features when available."""
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y = np.array([0, 1])
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    manifest: dict[str, Any] = {
        "feature_names": ["col_a", "col_b"],
        "thresholds": {"high": 0.80, "review": 0.40},
        "n_replicates": 0,
        "shap_top_k": 2,
        "stability_min_overlap": 0.67,
    }
    with pytest.raises(RuntimeError, match="do not match pipeline expected features"):
        validate_manifest(manifest, clf, [])


# -------------------------------------------------------------------------
# SHAP Explainer & Novelty Claim (Stability across Replicates) Tests
# -------------------------------------------------------------------------
def test_real_explainer_with_tree_pipeline_stable():
    """Verify RealExplainer computes SHAP via TreeExplainer and marks Stable when overlap >= threshold."""
    X = pd.DataFrame({
        "amount": [10.0, 50000.0, 100000.0, 20.0],
        "old_balance_orig": [100.0, 50000.0, 100000.0, 200.0],
        "new_balance_orig": [90.0, 0.0, 0.0, 180.0],
        "old_balance_dest": [0.0, 0.0, 1000.0, 50.0],
    })
    y = np.array([0, 1, 1, 0])

    clf1 = DecisionTreeClassifier(random_state=42, max_depth=3).fit(X, y)
    clf2 = DecisionTreeClassifier(random_state=42, max_depth=3).fit(X, y)

    explainer = RealExplainer(top_k=2, stability_min_overlap=0.67)
    features_df = X.iloc[[1]]
    top_feats, stability = explainer.explain(clf1, features_df, risk_score=0.90, replicates=[clf2])

    assert len(top_feats) == 2
    assert stability == ExplanationStatus.STABLE
    for feat in top_feats:
        assert feat.feature in X.columns
        assert feat.direction in ["increases_risk", "decreases_risk"]


def test_real_explainer_with_tree_pipeline_unstable():
    """Verify RealExplainer marks Unstable when replicate top-k features diverge."""
    # Main model learns from features f1, f2
    # Replicate model learns with inverse importance on f3, f4
    X = pd.DataFrame({
        "f1": [10.0, 100.0, 10.0, 100.0],
        "f2": [5.0, 50.0, 5.0, 50.0],
        "f3": [1.0, 1.0, 2.0, 2.0],
        "f4": [0.1, 0.1, 0.9, 0.9],
    })
    y1 = np.array([0, 1, 0, 1])  # Perfectly predicted by f1, f2
    y2 = np.array([0, 0, 1, 1])  # Perfectly predicted by f3, f4

    clf_main = DecisionTreeClassifier(random_state=42, max_depth=2).fit(X, y1)
    clf_rep = DecisionTreeClassifier(random_state=42, max_depth=2).fit(X, y2)

    explainer = RealExplainer(top_k=2, stability_min_overlap=0.67)
    features_df = X.iloc[[1]]
    _, stability = explainer.explain(clf_main, features_df, risk_score=0.85, replicates=[clf_rep])

    assert stability == ExplanationStatus.UNSTABLE


def test_real_explainer_mismatched_or_failing_replicates():
    """Verify RealExplainer handles failing replicates without crashing, falling back to Unstable."""
    X = pd.DataFrame({
        "amount": [10.0, 50000.0],
        "old_balance_orig": [100.0, 50000.0],
    })
    y = np.array([0, 1])
    clf = DecisionTreeClassifier().fit(X, y)

    class BrokenReplicate:
        def predict_proba(self, df):
            raise ValueError("Corrupt replicate weights")

    explainer = RealExplainer(top_k=2, stability_min_overlap=0.67)
    _, stability = explainer.explain(clf, X.iloc[[0]], risk_score=0.20, replicates=[BrokenReplicate()])
    assert stability == ExplanationStatus.UNSTABLE


def test_trained_model_pipeline_fixture(tmp_path: Path):
    """End-to-end test loading and predicting using a trained artifact fixture conforming to docs/ml-artifact-contract.md."""
    bundle_dir = tmp_path / "class_weight"
    bundle_dir.mkdir(parents=True)
    replicates_dir = bundle_dir / "replicates"
    replicates_dir.mkdir()

    # Pass the full snake_case columns as required by docs/ml-artifact-contract.md
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
    numeric_cols = ["step", "amount", "old_balance_orig", "new_balance_orig", "old_balance_dest", "new_balance_dest"]

    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", numeric_cols)],
        remainder="drop",
    )
    clf = DecisionTreeClassifier(random_state=42)
    pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])

    X_train = pd.DataFrame([
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
    ])
    y_train = np.array([1, 0])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, bundle_dir / "model.joblib")
    joblib.dump(pipeline, replicates_dir / "seed_42.joblib")

    manifest = {
        "model_version": "v1.0.0-test",
        "imbalance_strategy": "class_weight",
        "feature_names": feature_cols,
        "thresholds": {"high": 0.75, "review": 0.35},
        "shap_top_k": 3,
        "stability_min_overlap": 0.60,
        "n_replicates": 1,
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
    pred = svc.predict(tx)
    assert pred.model_version == "v1.0.0-test"
    assert pred.latency_ms > 0.0
    assert len(pred.top_features) > 0

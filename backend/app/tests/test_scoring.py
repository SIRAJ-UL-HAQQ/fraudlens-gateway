import json
from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from app.models.schemas import (
    ExplanationStatus,
    GatewayDecision,
    TransactionIn,
    TransactionType,
)
from app.services.scoring import ScoringService


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


def test_trained_model_pipeline_fixture(tmp_path: Path):
    """Test loading and predicting using a trained artifact fixture conforming to docs/ml-artifact-contract.md."""
    import joblib

    bundle_dir = tmp_path / "class_weight"
    bundle_dir.mkdir(parents=True)
    replicates_dir = bundle_dir / "replicates"
    replicates_dir.mkdir()

    # Create dummy pipeline that outputs calibrated probabilities
    clf = DummyClassifier(strategy="constant", constant=1)
    # Fit dummy on minimal mock data
    import numpy as np
    import pandas as pd
    dummy_df = pd.DataFrame([{
        "step": 1,
        "type": "TRANSFER",
        "amount": 100.0,
        "name_orig": "C1",
        "old_balance_orig": 100.0,
        "new_balance_orig": 0.0,
        "name_dest": "C2",
        "old_balance_dest": 0.0,
        "new_balance_dest": 100.0,
    }])
    clf.fit(dummy_df, np.array([1]))

    pipeline = Pipeline([("clf", clf)])
    joblib.dump(pipeline, bundle_dir / "model.joblib")
    joblib.dump(pipeline, replicates_dir / "seed_42.joblib")

    manifest = {
        "model_version": "v1.0.0-test",
        "imbalance_strategy": "class_weight",
        "feature_names": ["step", "type", "amount", "old_balance_orig", "new_balance_orig", "name_dest", "old_balance_dest", "new_balance_dest"],
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

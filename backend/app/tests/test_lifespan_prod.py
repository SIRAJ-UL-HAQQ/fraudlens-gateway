import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app.core.config import Settings, settings
from app.main import app, lifespan
from app.models.schemas import TransactionIn, TransactionType
from app.services.scoring import (
    ScoringService,
    extract_pipeline_feature_names,
    run_startup_self_test,
    scoring_service,
)


@pytest.fixture
def valid_bundle(tmp_path: Path):
    """Create a minimal valid model bundle conforming to docs/ml-artifact-contract.md."""
    bundle_dir = tmp_path / "valid_bundle"
    bundle_dir.mkdir(parents=True)
    reps_dir = bundle_dir / "replicates"
    reps_dir.mkdir()

    feature_cols = [
        "step",
        "type",
        "amount",
        "old_balance_orig",
        "new_balance_orig",
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

    ct = ColumnTransformer(
        transformers=[("num", "passthrough", numeric_cols)], remainder="drop"
    )
    clf = DecisionTreeClassifier(random_state=42)
    pipeline = Pipeline([("prep", ct), ("clf", clf)])

    X = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 1000.0,
                "old_balance_orig": 1000.0,
                "new_balance_orig": 0.0,
                "old_balance_dest": 0.0,
                "new_balance_dest": 1000.0,
            },
            {
                "step": 2,
                "type": "PAYMENT",
                "amount": 50.0,
                "old_balance_orig": 100.0,
                "new_balance_orig": 50.0,
                "old_balance_dest": 0.0,
                "new_balance_dest": 0.0,
            },
        ]
    )
    y = np.array([1, 0])
    pipeline.fit(X, y)

    joblib.dump(pipeline, bundle_dir / "model.joblib")
    joblib.dump(pipeline, reps_dir / "seed_42.joblib")
    joblib.dump(pipeline, reps_dir / "seed_43.joblib")

    manifest = {
        "model_version": "v1.0.0-prod",
        "imbalance_strategy": "class_weight",
        "feature_names": feature_cols,
        "thresholds": {"high": 0.70, "review": 0.40},
        "shap_top_k": 2,
        "stability_min_overlap": 0.50,
        "n_replicates": 2,
    }
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return bundle_dir


# -----------------------------------------------------------------------------
# 1. Production Secrets Checks
# -----------------------------------------------------------------------------
def test_lifespan_prod_placeholder_jwt_secret_fails_startup():
    """APP_ENV=prod rejects placeholder or empty JWT_SECRET_KEY."""
    with pytest.raises(
        ValueError, match="JWT_SECRET_KEY cannot use placeholder or default"
    ):
        Settings(
            APP_ENV="prod",
            JWT_SECRET_KEY="change_me_to_something_secure",
            ANALYST_PASSWORD="genuine_strong_analyst_password_2026",
        )

    with pytest.raises(ValueError, match="JWT_SECRET_KEY is required in production"):
        Settings(
            APP_ENV="prod",
            JWT_SECRET_KEY=None,
            ANALYST_PASSWORD="genuine_strong_analyst_password_2026",
        )


def test_lifespan_prod_placeholder_analyst_password_fails_startup():
    """APP_ENV=prod rejects placeholder or empty ANALYST_PASSWORD."""
    with pytest.raises(
        ValueError, match="ANALYST_PASSWORD cannot use placeholder or default"
    ):
        Settings(
            APP_ENV="prod",
            JWT_SECRET_KEY="genuine_custom_production_jwt_signing_key_443",
            ANALYST_PASSWORD="placeholder_analyst_password",
        )

    with pytest.raises(ValueError, match="ANALYST_PASSWORD is required in production"):
        Settings(
            APP_ENV="prod",
            JWT_SECRET_KEY="genuine_custom_production_jwt_signing_key_443",
            ANALYST_PASSWORD=None,
        )


# -----------------------------------------------------------------------------
# 2. Production Missing Artifact Checks
# -----------------------------------------------------------------------------
def test_lifespan_prod_missing_artifact_aborts_startup(monkeypatch):
    """When MODEL_PATH is unset or pointing to missing directory in prod, lifespan aborts."""
    import asyncio

    monkeypatch.setattr(settings, "APP_ENV", "prod")

    # Case A: MODEL_PATH is None
    monkeypatch.setattr(settings, "MODEL_PATH", None)

    async def run_missing_path():
        async with lifespan(app):
            pass

    with pytest.raises(
        RuntimeError, match="MODEL_PATH must be configured in non-dev/production"
    ):
        asyncio.run(run_missing_path())

    # Case B: MODEL_PATH directory does not exist
    monkeypatch.setattr(settings, "MODEL_PATH", "ml/models/non_existent_bundle_123")

    async def run_non_existent():
        async with lifespan(app):
            pass

    with pytest.raises(
        RuntimeError, match="Model artifact directory .* is invalid or missing"
    ):
        asyncio.run(run_non_existent())


# -----------------------------------------------------------------------------
# 3. Production Invalid Manifest Checks
# -----------------------------------------------------------------------------
def test_lifespan_prod_invalid_manifest_aborts_startup(valid_bundle: Path, monkeypatch):
    """Invalid manifest thresholds or schema aborts lifespan startup."""
    import asyncio

    monkeypatch.setattr(settings, "APP_ENV", "prod")
    monkeypatch.setattr(settings, "MODEL_PATH", str(valid_bundle))

    manifest_file = valid_bundle / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Invert thresholds (review >= high)
    data["thresholds"] = {"high": 0.40, "review": 0.80}
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    async def run_invalid_manifest():
        async with lifespan(app):
            pass

    with pytest.raises(RuntimeError, match="Manifest schema validation failed"):
        asyncio.run(run_invalid_manifest())


# -----------------------------------------------------------------------------
# 4. Production Replicates Mismatch Checks
# -----------------------------------------------------------------------------
def test_lifespan_prod_n_replicates_mismatch_aborts_startup(
    valid_bundle: Path, monkeypatch
):
    """When manifest specifies more replicates than files on disk, startup aborts."""
    import asyncio

    monkeypatch.setattr(settings, "APP_ENV", "prod")
    monkeypatch.setattr(settings, "MODEL_PATH", str(valid_bundle))

    manifest_file = valid_bundle / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["n_replicates"] = 5  # Files on disk is only 2
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    async def run_replicates_mismatch():
        async with lifespan(app):
            pass

    with pytest.raises(
        RuntimeError, match="does not match number of replicate files found"
    ):
        asyncio.run(run_replicates_mismatch())


# -----------------------------------------------------------------------------
# 5. Production Valid Artifact Lifespan Startup Success
# -----------------------------------------------------------------------------
def test_lifespan_prod_valid_artifact_starts_cleanly(valid_bundle: Path, monkeypatch):
    """When artifact and environment are valid, lifespan initializes trained model mode."""
    import asyncio

    monkeypatch.setattr(settings, "APP_ENV", "prod")
    monkeypatch.setattr(settings, "MODEL_PATH", str(valid_bundle))

    async def run_valid_bundle():
        async with lifespan(app):
            assert scoring_service.mode == "trained"
            assert scoring_service.model_version == "v1.0.0-prod"

    asyncio.run(run_valid_bundle())


# -----------------------------------------------------------------------------
# 6. Additional Tests to Cover ScoringService Branch Paths
# -----------------------------------------------------------------------------
def test_scoring_stub_predict_and_uuid_generation(monkeypatch):
    """Test stub scoring branches and default transaction_id generation."""
    monkeypatch.setattr(settings, "APP_ENV", "test")
    svc = ScoringService()
    svc.load_artifact(None)  # Stub mode

    assert svc.mode == "stub"

    # Transaction without explicit transaction_id generates UUID
    tx = TransactionIn(
        step=5,
        type=TransactionType.TRANSFER,
        amount=250000.0,
        name_orig="C111",
        old_balance_orig=250000.0,
        new_balance_orig=0.0,
        name_dest="C222",
        old_balance_dest=0.0,
        new_balance_dest=250000.0,
    )
    pred = svc.predict(tx, transaction_id=None)
    assert pred.transaction_id is not None
    assert len(pred.transaction_id) >= 32
    assert pred.risk_score >= 0.5
    assert len(pred.top_features) > 0


def test_scoring_load_artifact_via_direct_file_path(valid_bundle: Path):
    """Passing path to model.joblib directly resolves to parent directory."""
    svc = ScoringService()
    model_file_path = valid_bundle / "model.joblib"
    svc.load_artifact(str(model_file_path))
    assert svc.mode == "trained"


def test_scoring_load_artifact_corrupted_replicate(valid_bundle: Path):
    """Corrupted replicate file triggers startup failure."""
    corrupt_rep = valid_bundle / "replicates" / "seed_42.joblib"
    with open(corrupt_rep, "wb") as f:
        f.write(b"NOT_A_VALID_JOBLIB_FILE")

    svc = ScoringService()
    with pytest.raises(RuntimeError, match="Failed to load replicate pipeline"):
        svc.load_artifact(str(valid_bundle))


def test_scoring_load_artifact_less_than_two_replicates(valid_bundle: Path):
    """Fewer than 2 replicates on disk triggers startup abort."""
    for p in (valid_bundle / "replicates").glob("*.joblib"):
        p.unlink()

    svc = ScoringService()
    with pytest.raises(
        RuntimeError, match="does not match number of replicate files found"
    ):
        svc.load_artifact(str(valid_bundle))


def test_extract_feature_names_from_named_steps():
    """Extract input feature names when estimator is second step of pipeline."""
    clf = DecisionTreeClassifier()
    clf.feature_names_in_ = np.array(["col_a", "col_b"])
    pipe = Pipeline([("dummy", "passthrough"), ("clf", clf)])

    names = extract_pipeline_feature_names(pipe)
    assert names == ["col_a", "col_b"]


def test_run_startup_self_test_non_standard_probabilities():
    """Test 1D probability output and single column output in startup self-test."""

    class Mock1DModel:
        def predict_proba(self, X):
            return np.array([0.75])

    run_startup_self_test(Mock1DModel(), ["amount"])

    class Mock1ColModel:
        def predict_proba(self, X):
            return np.array([[0.85]])

    run_startup_self_test(Mock1ColModel(), ["amount"])

    class MockPredictOnlyModel:
        def predict(self, X):
            return np.array([0.45])

    run_startup_self_test(MockPredictOnlyModel(), ["amount"])

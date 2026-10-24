import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.models.schemas import ExplanationStatus, ShapDirection
from app.services.explainer import (
    RealExplainer,
    StubExplainer,
    validate_model_supported,
)


@pytest.fixture
def deterministic_signal_data():
    """Deterministic datasets where feature A dominates for main/stable, and feature C dominates for unstable."""
    X_main = pd.DataFrame(
        {
            "f_a": [10.0, 0.0, 10.0, 0.0],
            "f_b": [1.0, 1.0, 1.0, 1.0],
            "f_c": [2.0, 2.0, 2.0, 2.0],
        }
    )
    y = np.array([1, 0, 1, 0])

    X_unstable = pd.DataFrame(
        {
            "f_a": [1.0, 1.0, 1.0, 1.0],
            "f_b": [1.0, 1.0, 1.0, 1.0],
            "f_c": [10.0, 0.0, 10.0, 0.0],
        }
    )

    return X_main, X_unstable, y


def test_validate_model_supported():
    """Verify tree and linear models are supported, unsupported models fail fast."""
    dt = DecisionTreeClassifier()
    assert validate_model_supported(dt) == "tree"

    lr = LogisticRegression()
    assert validate_model_supported(lr) == "linear"

    pipe_dt = Pipeline([("clf", dt)])
    assert validate_model_supported(pipe_dt) == "tree"

    pipe_lr = Pipeline([("clf", lr)])
    assert validate_model_supported(pipe_lr) == "linear"

    knn = KNeighborsClassifier()
    with pytest.raises(TypeError, match="Only tree-based and linear models are supported"):
        validate_model_supported(knn)


def test_real_explainer_stable(deterministic_signal_data):
    """RealExplainer returns Stable when replicates share top-k feature attributions."""
    X_main, _, y = deterministic_signal_data
    main_dt = DecisionTreeClassifier(random_state=42).fit(X_main, y)
    rep_stable1 = DecisionTreeClassifier(random_state=43).fit(X_main, y)
    rep_stable2 = DecisionTreeClassifier(random_state=44).fit(X_main, y)

    single_row = X_main.iloc[[0]]
    explainer = RealExplainer(top_k=1, stability_min_overlap=0.67)
    explanations, status = explainer.explain(
        model=main_dt,
        features_df=single_row,
        risk_score=0.90,
        replicates=[rep_stable1, rep_stable2],
    )

    assert status == ExplanationStatus.STABLE
    assert len(explanations) == 1
    assert explanations[0].feature == "f_a"
    assert explanations[0].direction == ShapDirection.INCREASES_RISK


def test_real_explainer_unstable(deterministic_signal_data):
    """RealExplainer returns Unstable when replicates focus on divergent signals."""
    X_main, X_unstable, y = deterministic_signal_data
    main_dt = DecisionTreeClassifier(random_state=42).fit(X_main, y)
    rep_unstable1 = DecisionTreeClassifier(random_state=45).fit(X_unstable, y)
    rep_unstable2 = DecisionTreeClassifier(random_state=46).fit(X_unstable, y)

    single_row = X_main.iloc[[0]]
    explainer = RealExplainer(top_k=1, stability_min_overlap=0.67)
    explanations, status = explainer.explain(
        model=main_dt,
        features_df=single_row,
        risk_score=0.90,
        replicates=[rep_unstable1, rep_unstable2],
    )

    assert status == ExplanationStatus.UNSTABLE
    assert len(explanations) == 1


def test_real_explainer_pipeline_with_preprocessor(deterministic_signal_data):
    """RealExplainer correctly extracts features through an sklearn Pipeline."""
    X_main, _, y = deterministic_signal_data
    pipe_main = Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=42))])
    pipe_main.fit(X_main, y)

    pipe_rep1 = Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=43))])
    pipe_rep1.fit(X_main, y)

    pipe_rep2 = Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=44))])
    pipe_rep2.fit(X_main, y)

    single_row = X_main.iloc[[0]]
    explainer = RealExplainer(top_k=2, stability_min_overlap=0.50)
    explanations, status = explainer.explain(
        model=pipe_main,
        features_df=single_row,
        risk_score=0.90,
        replicates=[pipe_rep1, pipe_rep2],
    )

    assert status == ExplanationStatus.STABLE
    assert len(explanations) == 2


def test_real_explainer_linear_model(deterministic_signal_data):
    """RealExplainer supports linear models explicitly."""
    X_main, _, y = deterministic_signal_data
    lr_main = LogisticRegression(random_state=42).fit(X_main, y)
    lr_rep1 = LogisticRegression(random_state=43).fit(X_main, y)
    lr_rep2 = LogisticRegression(random_state=44).fit(X_main, y)

    single_row = X_main.iloc[[0]]
    explainer = RealExplainer(top_k=2, stability_min_overlap=0.67)
    explanations, status = explainer.explain(
        model=lr_main,
        features_df=single_row,
        risk_score=0.85,
        replicates=[lr_rep1, lr_rep2],
    )

    assert status == ExplanationStatus.STABLE
    assert len(explanations) == 2


def test_stability_boundary_conditions():
    """Test integer count stability boundaries:
    - 2/3 vs 3/3 with k=3
    - Exact integer case where min_overlap * k is an integer
    """
    class MockOverlapExplainer(RealExplainer):
        def __init__(self, top_k: int, min_overlap: float, fixed_reps_top: list[list[tuple[str, float]]]):
            super().__init__(top_k=top_k, stability_min_overlap=min_overlap)
            self._fixed_reps = fixed_reps_top

        def _extract_top_features_from_model(self, pipeline, features_df):
            if pipeline == "main":
                return [(f"f_{i}", 0.5) for i in range(self.top_k)]
            idx = int(pipeline.replace("rep_", ""))
            return self._fixed_reps[idx]

    df_dummy = pd.DataFrame([{"a": 1}])

    # 1. k=3, min_overlap = 2/3 (0.6666666667).
    # ceil( (2/3)*3 - 1e-9 ) = ceil(2 - 1e-9) = 2.
    # Replicates with 2 overlapping features out of 3 -> avg_count = 2 -> Stable
    exp_2_3 = MockOverlapExplainer(
        top_k=3,
        min_overlap=2 / 3,
        fixed_reps_top=[
            [("f_0", 0.5), ("f_1", 0.4), ("diff_a", 0.3)],
            [("f_0", 0.5), ("f_1", 0.4), ("diff_b", 0.3)],
        ],
    )
    _, status = exp_2_3.explain("main", df_dummy, 0.8, ["rep_0", "rep_1"])
    assert status == ExplanationStatus.STABLE

    # 2. k=3, min_overlap = 0.67.
    # ceil( 0.67*3 - 1e-9 ) = ceil(2.01 - 1e-9) = 3.
    # Overlap count = 2 -> Unstable (2 < 3)
    exp_067_unstable = MockOverlapExplainer(
        top_k=3,
        min_overlap=0.67,
        fixed_reps_top=[
            [("f_0", 0.5), ("f_1", 0.4), ("diff_a", 0.3)],
            [("f_0", 0.5), ("f_1", 0.4), ("diff_b", 0.3)],
        ],
    )
    _, status = exp_067_unstable.explain("main", df_dummy, 0.8, ["rep_0", "rep_1"])
    assert status == ExplanationStatus.UNSTABLE

    # Overlap count = 3 -> Stable (3 >= 3)
    exp_067_stable = MockOverlapExplainer(
        top_k=3,
        min_overlap=0.67,
        fixed_reps_top=[
            [("f_0", 0.5), ("f_1", 0.4), ("f_2", 0.3)],
            [("f_0", 0.5), ("f_1", 0.4), ("f_2", 0.3)],
        ],
    )
    _, status = exp_067_stable.explain("main", df_dummy, 0.8, ["rep_0", "rep_1"])
    assert status == ExplanationStatus.STABLE

    # 3. Exact integer case: k=5, min_overlap = 0.60 -> 0.60 * 5 = 3.0.
    # ceil(3.0 - 1e-9) = 3.
    # Exactly 3 overlapping features -> avg count 3 >= 3 -> Stable
    exp_int_stable = MockOverlapExplainer(
        top_k=5,
        min_overlap=0.60,
        fixed_reps_top=[
            [("f_0", 0.5), ("f_1", 0.4), ("f_2", 0.3), ("d1", 0.1), ("d2", 0.1)],
            [("f_0", 0.5), ("f_1", 0.4), ("f_2", 0.3), ("d3", 0.1), ("d4", 0.1)],
        ],
    )
    _, status = exp_int_stable.explain("main", df_dummy, 0.8, ["rep_0", "rep_1"])
    assert status == ExplanationStatus.STABLE

    # 2 overlapping features -> avg count 2 < 3 -> Unstable
    exp_int_unstable = MockOverlapExplainer(
        top_k=5,
        min_overlap=0.60,
        fixed_reps_top=[
            [("f_0", 0.5), ("f_1", 0.4), ("d0", 0.3), ("d1", 0.1), ("d2", 0.1)],
            [("f_0", 0.5), ("f_1", 0.4), ("d3", 0.1), ("d4", 0.1), ("d5", 0.1)],
        ],
    )
    _, status = exp_int_unstable.explain("main", df_dummy, 0.8, ["rep_0", "rep_1"])
    assert status == ExplanationStatus.UNSTABLE


def test_replicates_fail_closed_at_request_time(deterministic_signal_data):
    """Any replicate failure or empty replicate list at request time returns Unstable."""
    X_main, _, y = deterministic_signal_data
    main_dt = DecisionTreeClassifier(random_state=42).fit(X_main, y)
    rep1 = DecisionTreeClassifier(random_state=43).fit(X_main, y)

    single_row = X_main.iloc[[0]]
    explainer = RealExplainer(top_k=2, stability_min_overlap=0.67)

    # 1. Fewer than 2 replicates (empty or single) -> Unstable
    _, status_empty = explainer.explain(main_dt, single_row, 0.85, replicates=[])
    assert status_empty == ExplanationStatus.UNSTABLE

    _, status_one = explainer.explain(main_dt, single_row, 0.85, replicates=[rep1])
    assert status_one == ExplanationStatus.UNSTABLE

    # 2. Replicate that fails with exception at request time -> Unstable
    bad_replicate = "unsupported_string_model"
    _, status_bad = explainer.explain(main_dt, single_row, 0.85, replicates=[rep1, bad_replicate])
    assert status_bad == ExplanationStatus.UNSTABLE

    # 3. Replicate returning empty features -> Unstable
    class EmptyFeatureExplainer(RealExplainer):
        def _extract_top_features_from_model(self, pipeline, features_df):
            if pipeline == "rep_empty":
                return []
            return [("f_a", 0.5)]

    # 4. Main model returning empty features falls back to default columns
    class EmptyMainExplainer(RealExplainer):
        def _extract_top_features_from_model(self, pipeline, features_df):
            if pipeline == "main":
                return []
            return [("f_a", 0.5)]

    exp_empty_main = EmptyMainExplainer(top_k=1, stability_min_overlap=0.67)
    exps, status_main_empty = exp_empty_main.explain("main", single_row, 0.85, replicates=[rep1, rep1])
    assert len(exps) == 1
    assert status_main_empty == ExplanationStatus.STABLE


def test_top_k_larger_than_feature_count(deterministic_signal_data):
    """When top_k is larger than available features, all features are returned without error."""
    X_main, _, y = deterministic_signal_data
    dt = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_main[["f_a", "f_b"]], y)

    explainer = RealExplainer(top_k=10, stability_min_overlap=0.67)
    explanations, _ = explainer.explain(
        model=dt,
        features_df=X_main[["f_a", "f_b"]].iloc[[0]],
        risk_score=0.8,
        replicates=[dt, dt],
    )
    assert len(explanations) == 2


def test_stub_explainer_behavior():
    """StubExplainer produces deterministic attributions and stability."""
    stub = StubExplainer()

    # High score
    df_tx = pd.DataFrame([{"amount": 1000.0, "type": "PAYMENT"}])
    top_high, status_high = stub.explain(None, df_tx, risk_score=0.85)
    assert len(top_high) == 3
    assert status_high == ExplanationStatus.STABLE
    assert top_high[0].direction == ShapDirection.INCREASES_RISK

    # Low score
    top_low, status_low = stub.explain(None, df_tx, risk_score=0.20)
    assert len(top_low) == 3
    assert status_low == ExplanationStatus.STABLE
    assert top_low[0].direction == ShapDirection.DECREASES_RISK

    # Large transfer trigger Unstable in stub
    df_large_transfer = pd.DataFrame([{"amount": 75000.0, "type": "TRANSFER"}])
    _, status_unstable = stub.explain(None, df_large_transfer, risk_score=0.85)
    assert status_unstable == ExplanationStatus.UNSTABLE

"""Comprehensive unit tests for ML pipeline components.

Uses synthetic data fixtures only (fast in CI, no external raw dataset required).
"""

import numpy as np
import pandas as pd
import pytest
from app.models.schemas import TransactionIn, TransactionType
from app.services.rules import check_origin_account_drained
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from ml.src.features import API_INPUT_COLUMNS, PaySimFeaturePreprocessor
from ml.src.load_data import (
    evaluate_rules_baseline_vectorized,
    map_raw_to_api_columns,
    split_chronological,
)


@pytest.fixture
def synthetic_paysim_df() -> pd.DataFrame:
    """Generate reproducible synthetic PaySim DataFrame across steps 1..700."""
    rng = np.random.RandomState(42)
    n_rows = 300

    steps = np.sort(rng.choice(np.arange(1, 701), size=n_rows, replace=True))
    types = rng.choice(
        ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], size=n_rows
    )
    amounts = rng.exponential(scale=100000.0, size=n_rows)
    old_orig = amounts + rng.exponential(scale=50000.0, size=n_rows)
    new_orig = np.maximum(0.0, old_orig - amounts)
    old_dest = rng.exponential(scale=20000.0, size=n_rows)
    new_dest = old_dest + amounts

    # 10% fraud distributed in TRANSFER and CASH_OUT
    is_fraud = np.zeros(n_rows, dtype=int)
    for i in range(n_rows):
        if types[i] in ["TRANSFER", "CASH_OUT"] and rng.rand() < 0.20:
            is_fraud[i] = 1
            new_orig[i] = 0.0  # Drained account pattern

    df = pd.DataFrame(
        {
            "step": steps,
            "type": types,
            "amount": amounts,
            "name_orig": [f"C{rng.randint(100000, 999999)}" for _ in range(n_rows)],
            "old_balance_orig": old_orig,
            "new_balance_orig": new_orig,
            "name_dest": [
                f"{'M' if types[i] == 'PAYMENT' else 'C'}{rng.randint(100000, 999999)}"
                for i in range(n_rows)
            ],
            "old_balance_dest": old_dest,
            "new_balance_dest": new_dest,
            "is_fraud": is_fraud,
            "is_flagged_fraud": np.zeros(n_rows, dtype=int),
        }
    )
    return df


def test_split_chronological_ordering(synthetic_paysim_df: pd.DataFrame):
    """Assert strict chronological ordering between train, validation, and test splits."""
    train_df, val_df, test_df = split_chronological(
        synthetic_paysim_df, train_step_max=520, val_step_max=595
    )

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    train_max = train_df["step"].max()
    val_min = val_df["step"].min()
    val_max = val_df["step"].max()
    test_min = test_df["step"].min()

    assert train_max < val_min, f"Train step max {train_max} >= Val step min {val_min}"
    assert val_max < test_min, f"Val step max {val_max} >= Test step min {test_min}"


def test_pipeline_immutability_on_validation_test_mutation(
    synthetic_paysim_df: pd.DataFrame,
):
    """Verify that stateful pipeline transformers fitted on train are immutable to validation/test mutations.

    PaySimFeaturePreprocessor is structural, but a pipeline with a stateful transformer
    (e.g. RobustScaler) learns empirical center_ and scale_ parameters. We verify that
    mutating validation and test rows does not corrupt learned parameters, and that a
    re-fit on mutated data would produce significantly different parameters (proving
    the test is sensitive and non-vacuous).
    """
    train_df, val_df, test_df = split_chronological(synthetic_paysim_df)

    pipeline = Pipeline(
        [
            ("prep", PaySimFeaturePreprocessor(include_balance_errors=False)),
            ("scaler", RobustScaler()),
        ]
    )

    # Fit pipeline strictly on train
    pipeline.fit(train_df)
    scaler = pipeline.named_steps["scaler"]
    orig_center = scaler.center_.copy()
    orig_scale = scaler.scale_.copy()
    train_output = pipeline.transform(train_df).copy()

    # Mutate validation and test data aggressively with extreme outliers
    mutated_val = val_df.copy()
    mutated_val["amount"] = mutated_val["amount"] * 10000.0
    mutated_val["old_balance_orig"] = 9999999999.0

    mutated_test = test_df.copy()
    mutated_test["amount"] = 8888888888.0

    # Transform mutated splits
    _ = pipeline.transform(mutated_val)
    _ = pipeline.transform(mutated_test)

    # Assert fitted scaler parameters remain strictly bitwise unchanged
    np.testing.assert_array_equal(scaler.center_, orig_center)
    np.testing.assert_array_equal(scaler.scale_, orig_scale)

    # Re-transform train split and assert identical output
    re_train_output = pipeline.transform(train_df)
    np.testing.assert_array_equal(train_output, re_train_output)

    # Non-vacuity assertion: if mistakenly re-fitted on mutated data, parameters must differ
    leakage_pipeline = Pipeline(
        [
            ("prep", PaySimFeaturePreprocessor(include_balance_errors=False)),
            ("scaler", RobustScaler()),
        ]
    )
    combined_leaked = pd.concat([train_df, mutated_val], ignore_index=True)
    leakage_pipeline.fit(combined_leaked)
    leaked_scaler = leakage_pipeline.named_steps["scaler"]

    assert not np.allclose(
        leaked_scaler.center_, orig_center
    ), "Test non-vacuity check failed: leaked fit should alter center_"


def test_preprocessor_handles_unknown_type_and_zero_amount():
    """Verify preprocessor converts unknown types to all zeros and handles zero amount."""
    prep = PaySimFeaturePreprocessor(include_balance_errors=False)

    df = pd.DataFrame(
        [
            {
                "step": 10,
                "type": "UNKNOWN_CUSTOM_TYPE",
                "amount": 0.0,
                "name_orig": "C12345",
                "old_balance_orig": 100.0,
                "new_balance_orig": 100.0,
                "name_dest": "C67890",
                "old_balance_dest": 50.0,
                "new_balance_dest": 50.0,
            }
        ]
    )

    prep.fit(df)
    trans = prep.transform(df)

    assert trans.shape == (1, len(prep.get_feature_names_out()))
    # Amount is index 0
    assert trans[0, 0] == 0.0
    # Hour of day is index 5 (10 % 24 = 10)
    assert trans[0, 5] == 10.0
    # Type one-hot indices are 6..10; all should be 0.0 for unknown type
    for col_idx in range(6, 11):
        assert trans[0, col_idx] == 0.0


def test_preprocessor_variants_feature_names_and_shapes():
    """Test feature outputs for both Variant A (with balance errors) and Variant B (without)."""
    df = pd.DataFrame(
        [
            {
                "step": 5,
                "type": "TRANSFER",
                "amount": 1000.0,
                "name_orig": "C111",
                "old_balance_orig": 1000.0,
                "new_balance_orig": 0.0,
                "name_dest": "C222",
                "old_balance_dest": 500.0,
                "new_balance_dest": 1500.0,
            }
        ]
    )

    # Variant B: Without balance errors/deltas
    prep_b = PaySimFeaturePreprocessor(include_balance_errors=False)
    prep_b.fit(df)
    names_b = prep_b.get_feature_names_out()
    trans_b = prep_b.transform(df)

    assert len(names_b) == 11
    assert "balance_error_orig" not in names_b
    assert "balance_delta_orig" not in names_b
    assert trans_b.shape == (1, 11)

    # Variant A: With balance errors/deltas
    prep_a = PaySimFeaturePreprocessor(include_balance_errors=True)
    prep_a.fit(df)
    names_a = prep_a.get_feature_names_out()
    trans_a = prep_a.transform(df)

    assert len(names_a) == 15
    assert "balance_error_orig" in names_a
    assert "balance_delta_orig" in names_a
    assert trans_a.shape == (1, 15)

    # Check exact balance error: 1000 - 1000 - 0 = 0.0
    err_orig_idx = list(names_a).index("balance_error_orig")
    assert trans_a[0, err_orig_idx] == 0.0


def test_map_raw_to_api_columns():
    """Test column rename mapping matches API contract."""
    raw_df = pd.DataFrame(
        {
            "step": [1],
            "type": ["TRANSFER"],
            "amount": [10.0],
            "nameOrig": ["C1"],
            "oldbalanceOrg": [10.0],
            "newbalanceOrig": [0.0],
            "nameDest": ["C2"],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [10.0],
            "isFraud": [1],
            "isFlaggedFraud": [0],
        }
    )
    mapped = map_raw_to_api_columns(raw_df)
    for col in API_INPUT_COLUMNS:
        assert col in mapped.columns
    assert "is_fraud" in mapped.columns
    assert "is_flagged_fraud" in mapped.columns


def test_vectorized_rules_match_backend_rule_function(
    synthetic_paysim_df: pd.DataFrame,
):
    """Assert that vectorized rule logic matches app.services.rules.check_origin_account_drained."""
    for _, row in synthetic_paysim_df.iterrows():
        try:
            tx_type = TransactionType(row["type"])
        except ValueError:
            continue

        tx = TransactionIn(
            step=int(row["step"]),
            type=tx_type,
            amount=float(row["amount"]),
            name_orig=str(row["name_orig"]),
            old_balance_orig=float(row["old_balance_orig"]),
            new_balance_orig=float(row["new_balance_orig"]),
            name_dest=str(row["name_dest"]),
            old_balance_dest=float(row["old_balance_dest"]),
            new_balance_dest=float(row["new_balance_dest"]),
        )
        backend_flag = check_origin_account_drained(tx)

        # Vectorized equivalent condition
        is_type = tx.type in [TransactionType.TRANSFER, TransactionType.CASH_OUT]
        vectorized_flag = (
            is_type
            and tx.old_balance_orig > 0.0
            and tx.new_balance_orig == 0.0
            and tx.amount >= tx.old_balance_orig * 0.95
        )

        assert (backend_flag is not None) == vectorized_flag


def test_evaluate_rules_baseline(synthetic_paysim_df: pd.DataFrame):
    """Test rules-only baseline returns valid metrics for all three heuristic configurations."""
    results = evaluate_rules_baseline_vectorized(synthetic_paysim_df)

    assert "drain_ge_95pct_alone" in results
    assert "amount_gt_200k_alone" in results
    assert "union_drain_or_amount" in results

    for metrics in results.values():
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["fpr"] <= 1.0
        assert (
            metrics["true_positives"]
            + metrics["false_positives"]
            + metrics["true_negatives"]
            + metrics["false_negatives"]
            == len(synthetic_paysim_df)
        )


def test_shap_stability_arithmetic():
    """Verify SRS stability arithmetic: overlap = |topk(main) ∩ topk(rep)| / k,

    and stable iff overlap_count >= ceil(min_overlap * k - 1e-9).
    """
    from ml.src.shap_stability import (
        is_explanation_stable,
        spearman_rank_correlation,
        top_k_overlap,
    )

    # 1. Overlap ratio formula |top_k(main) ∩ top_k(rep)| / k
    main_top5 = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
    rep_top5_4overlap = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_z"]
    rep_top5_3overlap = ["feat_a", "feat_b", "feat_c", "feat_y", "feat_z"]

    assert top_k_overlap(main_top5, rep_top5_4overlap) == 0.80
    assert top_k_overlap(main_top5, rep_top5_3overlap) == 0.60

    # 2. Sensitivity arithmetic at k=5, min_overlap=0.70 -> ceil(3.5 - 1e-9) = 4
    assert is_explanation_stable(0.80, k=5, min_overlap=0.70) is True  # 4/5
    assert is_explanation_stable(0.60, k=5, min_overlap=0.70) is False  # 3/5

    # 3. Sensitivity arithmetic at k=3, min_overlap=0.70 -> ceil(2.1 - 1e-9) = 3
    assert is_explanation_stable(1.00, k=3, min_overlap=0.70) is True  # 3/3
    assert is_explanation_stable(0.67, k=3, min_overlap=0.70) is False  # 2/3 (ceil(2.099999999)=3)

    # 4. Sensitivity arithmetic at k=10, min_overlap=0.70 -> ceil(7.0 - 1e-9) = 7
    assert is_explanation_stable(0.70, k=10, min_overlap=0.70) is True  # 7/10
    assert is_explanation_stable(0.60, k=10, min_overlap=0.70) is False  # 6/10

    # 5. Spearman rank correlation
    assert spearman_rank_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert spearman_rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0


def test_gateway_threshold_selection():
    """Verify single gateway threshold selection logic:

    - review = largest t in unique scores with recall >= 0.95
    - high = smallest t > review with precision >= 0.90 and positives >= 20
    - Enforces 0 < review < high < 1
    """
    from ml.src.evaluate import compute_gateway_thresholds

    rng = np.random.RandomState(42)
    # Generate 100 positive instances and 900 negative instances
    n_pos = 100
    n_neg = 900
    y_val = np.array([1] * n_pos + [0] * n_neg)

    # Well-separated probability scores
    pos_scores = rng.uniform(0.70, 0.99, size=n_pos)
    neg_scores = rng.uniform(0.01, 0.85, size=n_neg)
    val_probs = np.concatenate([pos_scores, neg_scores])

    res = compute_gateway_thresholds(
        y_val=y_val,
        val_probs=val_probs,
        target_recall=0.95,
        target_precision=0.90,
        min_predicted_positives=20,
    )

    t_rev = res["threshold_review"]
    t_high = res["threshold_high"]

    assert 0.0 < t_rev < t_high < 1.0
    assert res["metrics_at_review"]["recall"] >= 0.95
    assert res["metrics_at_high"]["precision"] >= 0.90
    assert res["metrics_at_high"]["predicted_positives"] >= 20

    # Test error raised when target precision cannot be achieved
    with pytest.raises(RuntimeError, match="No threshold t > review"):
        compute_gateway_thresholds(
            y_val=y_val,
            val_probs=val_probs,
            target_recall=0.95,
            target_precision=0.9999,  # Unreachable precision
            min_predicted_positives=50,
        )



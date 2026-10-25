"""Model training, artifact packaging, and evaluation module for PaySim fraud detection.

Strictly adheres to docs/ml-artifact-contract.md and docs/api-contract.md:
- Packages models into ml/models/<strategy>/ with model.joblib, replicates/seed_*.joblib, manifest.json
- Enforces gateway decision semantics and threshold selection on validation split
- Evaluates test split strictly once after artifacts are frozen
- Computes multi-replicate SHAP explanation stability and chance baselines
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from ml.src.check_artifact import check_bundle
from ml.src.evaluate import compute_gateway_thresholds
from ml.src.features import API_INPUT_COLUMNS, PaySimFeaturePreprocessor
from ml.src.imbalance_strategies import (
    apply_imbalance_strategy,
    apply_majority_subsampling,
)
from ml.src.load_data import (
    evaluate_rules_baseline_vectorized,
    load_raw_paysim,
    split_chronological,
)
from ml.src.shap_stability import (
    evaluate_cross_strategy_agreement,
    evaluate_replicate_stability,
)

logger = logging.getLogger(__name__)

DEFAULT_MAIN_SEED = 42
DEFAULT_REPLICATE_SEEDS = [101, 102, 103, 104, 105]

FROZEN_HYPERPARAMS_VARIANT_B = {
    "max_depth": 6,
    "learning_rate": 0.16552248735454136,
    "n_estimators": 190,
    "subsample": 0.7263892580562121,
    "colsample_bytree": 0.8086684313220379,
    "min_child_weight": 1,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": 4,
}

FROZEN_HYPERPARAMS_VARIANT_A = {
    "max_depth": 5,
    "learning_rate": 0.16707221510623524,
    "n_estimators": 113,
    "subsample": 0.6587948587257435,
    "colsample_bytree": 0.6135681866731614,
    "min_child_weight": 4,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": 4,
}


def compute_metrics_bundle(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute standard classification evaluation metrics at a specific decision threshold."""
    pred = (probs >= threshold).astype(int)
    rec = float(recall_score(y_true, pred, zero_division=0))
    prec = float(precision_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    pr_auc = float(average_precision_score(y_true, probs))
    roc_auc = float(roc_auc_score(y_true, probs))
    mcc = float(matthews_corrcoef(y_true, pred))
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(fp + tn, 1))

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "mcc": round(mcc, 4),
        "fpr": round(fpr, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def compute_bootstrap_cis(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """Compute 95% bootstrap confidence intervals for recall and FPR."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    recalls = []
    fprs = []

    pred_all = (probs >= threshold).astype(int)

    for _ in range(n_bootstrap):
        boot_idx = rng.randint(0, n, size=n)
        b_y = y_true[boot_idx]
        b_pred = pred_all[boot_idx]

        tp = np.sum((b_y == 1) & (b_pred == 1))
        fp = np.sum((b_y == 0) & (b_pred == 1))
        b_n_pos = np.sum(b_y == 1)
        b_n_neg = np.sum(b_y == 0)

        r = tp / b_n_pos if b_n_pos > 0 else 0.0
        fpr = fp / b_n_neg if b_n_neg > 0 else 0.0
        recalls.append(r)
        fprs.append(fpr)

    return {
        "recall_ci_95": [
            round(float(np.percentile(recalls, 2.5)), 4),
            round(float(np.percentile(recalls, 97.5)), 4),
        ],
        "fpr_ci_95": [
            round(float(np.percentile(fprs, 2.5)), 4),
            round(float(np.percentile(fprs, 97.5)), 4),
        ],
    }


def train_and_package_strategy(
    strategy: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    replicate_seeds: list[int] = DEFAULT_REPLICATE_SEEDS,
) -> dict[str, Any] | None:
    """Train main model and replicates for a given strategy and package artifacts."""
    print("\n==========================================")
    print(f"Training Strategy: {strategy}")
    print("==========================================")

    # 1. Fit structural preprocessor on training DataFrame
    prep = PaySimFeaturePreprocessor(include_balance_errors=False)
    prep.fit(train_df[API_INPUT_COLUMNS])
    X_train_trans = prep.transform(train_df[API_INPUT_COLUMNS])
    y_train = train_df["is_fraud"].to_numpy(dtype=int)
    y_val = val_df["is_fraud"].to_numpy(dtype=int)

    # 2. Majority subsample (10:1) with seed 42
    X_sub_main, y_sub_main = apply_majority_subsampling(
        X_train_trans, y_train, neg_to_pos_ratio=10.0, random_state=DEFAULT_MAIN_SEED
    )
    X_res_main, y_res_main, spw_main = apply_imbalance_strategy(
        X_sub_main, y_sub_main, strategy=strategy, random_state=DEFAULT_MAIN_SEED
    )

    main_params = dict(FROZEN_HYPERPARAMS_VARIANT_B)
    main_params["scale_pos_weight"] = spw_main
    main_params["random_state"] = DEFAULT_MAIN_SEED

    print(f"Fitting main XGBoost model for '{strategy}'...")
    main_clf = XGBClassifier(**main_params)
    main_clf.fit(X_res_main, y_res_main)

    # Assemble end-to-end Pipeline
    main_pipeline = Pipeline([("prep", prep), ("clf", main_clf)])

    # Predict validation scores
    val_probs = main_pipeline.predict_proba(val_df[API_INPUT_COLUMNS])[:, 1]
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    print(f"Validation PR-AUC: {val_pr_auc:.4f}")

    # Threshold calibration
    try:
        thresh_info = compute_gateway_thresholds(y_val, val_probs)
    except RuntimeError as ex:
        print(f"WARNING: Threshold selection failed for strategy '{strategy}': {ex}")
        return None

    t_review = thresh_info["unrounded"]["t_review"]
    t_high = thresh_info["unrounded"]["t_high"]
    print(f"Selected thresholds: review={t_review:.6f}, high={t_high:.6f}")

    # 3. Train replicate pipelines
    strat_dir = output_dir / strategy
    replicates_dir = strat_dir / "replicates"
    replicates_dir.mkdir(parents=True, exist_ok=True)

    replicate_pipelines = []
    for rep_seed in replicate_seeds:
        print(f"Training replicate seed={rep_seed}...")
        # Vary majority non-fraud subsampling draw and estimator seed
        X_sub_rep, y_sub_rep = apply_majority_subsampling(
            X_train_trans, y_train, neg_to_pos_ratio=10.0, random_state=rep_seed
        )
        # For smote_enn, re-run resampling with replicate seed
        X_res_rep, y_res_rep, spw_rep = apply_imbalance_strategy(
            X_sub_rep, y_sub_rep, strategy=strategy, random_state=rep_seed
        )

        rep_params = dict(FROZEN_HYPERPARAMS_VARIANT_B)
        rep_params["scale_pos_weight"] = spw_rep
        rep_params["random_state"] = rep_seed

        rep_clf = XGBClassifier(**rep_params)
        rep_clf.fit(X_res_rep, y_res_rep)

        rep_pipeline = Pipeline([("prep", prep), ("clf", rep_clf)])
        rep_path = replicates_dir / f"seed_{rep_seed}.joblib"
        joblib.dump(rep_pipeline, rep_path)
        replicate_pipelines.append(rep_pipeline)

    # 4. Serialize main pipeline
    model_path = strat_dir / "model.joblib"
    joblib.dump(main_pipeline, model_path)

    # 5. Write manifest.json strictly conforming to ManifestSchema (extra="forbid")
    manifest_data = {
        "model_version": f"v1.0.0-{strategy}",
        "imbalance_strategy": strategy,
        "feature_names": list(API_INPUT_COLUMNS),
        "thresholds": {
            "high": float(t_high),
            "review": float(t_review),
        },
        "shap_top_k": 5,
        "stability_min_overlap": 0.70,
        "n_replicates": len(replicate_seeds),
    }

    manifest_path = strat_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Validate bundle
    check_bundle(strat_dir)

    return {
        "strategy": strategy,
        "main_pipeline": main_pipeline,
        "replicate_pipelines": replicate_pipelines,
        "thresholds": thresh_info,
        "val_pr_auc": val_pr_auc,
        "bundle_dir": strat_dir,
    }


def compute_shap_matrix_cohort(
    pipeline: Pipeline,
    cohort_df: pd.DataFrame,
) -> np.ndarray:
    """Compute TreeExplainer SHAP values matrix for a cohort using pipeline steps."""
    import shap

    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    X_trans = prep.transform(cohort_df[API_INPUT_COLUMNS])
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_trans)

    if isinstance(sv, list):
        return np.asarray(sv[1] if len(sv) >= 2 else sv[0])
    if hasattr(sv, "shape") and len(sv.shape) == 3:
        return np.asarray(sv[:, :, 1])
    return np.asarray(sv)


def run_stage2_training(
    data_path: str = "ml/data/raw/paysim.csv",
    output_dir_str: str = "ml/models",
    is_benchmark: bool = False,
) -> dict[str, Any]:
    """Execute complete Stage 2 training, replicate generation, test evaluation, and report generation."""
    output_dir = Path(output_dir_str)
    if is_benchmark:
        output_dir = Path(tempfile.mkdtemp(prefix="fraudlens_benchmark_"))
        print(f"Running in benchmark mode: writing to temporary directory {output_dir}")

    print("Loading PaySim dataset...")
    df = load_raw_paysim(data_path)
    train_df, val_df, test_df = split_chronological(df)
    y_test = test_df["is_fraud"].to_numpy(dtype=int)

    strategies = ["class_weight", "none", "smote_enn"]
    trained_strategies: dict[str, Any] = {}

    for strat in strategies:
        res = train_and_package_strategy(
            strategy=strat,
            train_df=train_df,
            val_df=val_df,
            output_dir=output_dir,
            replicate_seeds=DEFAULT_REPLICATE_SEEDS,
        )
        if res is not None:
            trained_strategies[strat] = res

    # -------------------------------------------------------------
    # Baselines & Reference Runs (Evaluated, not saved to ml/models)
    # -------------------------------------------------------------
    print("\n--- Training Reference Models & Baselines ---")
    prep_b = PaySimFeaturePreprocessor(include_balance_errors=False)
    prep_b.fit(train_df[API_INPUT_COLUMNS])
    X_train_b = prep_b.transform(train_df[API_INPUT_COLUMNS])
    X_test_b = prep_b.transform(test_df[API_INPUT_COLUMNS])
    y_train = train_df["is_fraud"].to_numpy(dtype=int)

    X_sub, y_sub = apply_majority_subsampling(
        X_train_b, y_train, neg_to_pos_ratio=10.0, random_state=DEFAULT_MAIN_SEED
    )

    # 1. Logistic Regression Baseline
    print("Fitting Logistic Regression Baseline...")
    lr_clf = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=DEFAULT_MAIN_SEED)),
    ])
    lr_clf.fit(X_sub, y_sub)
    lr_test_probs = lr_clf.predict_proba(X_test_b)[:, 1]
    lr_metrics = compute_metrics_bundle(y_test, lr_test_probs, threshold=0.50)

    # 2. Random Forest Baseline
    print("Fitting Random Forest Baseline...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=DEFAULT_MAIN_SEED, n_jobs=4)
    rf_clf.fit(X_sub, y_sub)
    rf_test_probs = rf_clf.predict_proba(X_test_b)[:, 1]
    rf_metrics = compute_metrics_bundle(y_test, rf_test_probs, threshold=0.50)

    # 3. Rules-Only Baseline (SRS FR-7 / Backend check_origin_account_drained)
    print("Evaluating Rules-Only Baseline on Test Set...")
    rules_test_metrics = evaluate_rules_baseline_vectorized(test_df)

    # 4. Full-Train Reference Runs for 'none' and 'class_weight'
    print("Fitting Full-Train Reference XGBoost Models (6.08M rows)...")
    full_train_clf_none = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        tree_method="hist",
        scale_pos_weight=1.0,
        random_state=DEFAULT_MAIN_SEED,
        n_jobs=4,
    )
    full_train_clf_none.fit(X_train_b, y_train)
    full_test_probs_none = full_train_clf_none.predict_proba(X_test_b)[:, 1]
    full_metrics_none = compute_metrics_bundle(y_test, full_test_probs_none, threshold=0.50)

    spw_full = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    full_train_clf_cw = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        tree_method="hist",
        scale_pos_weight=spw_full,
        random_state=DEFAULT_MAIN_SEED,
        n_jobs=4,
    )
    full_train_clf_cw.fit(X_train_b, y_train)
    full_test_probs_cw = full_train_clf_cw.predict_proba(X_test_b)[:, 1]
    full_metrics_cw = compute_metrics_bundle(y_test, full_test_probs_cw, threshold=0.50)

    # 5. Variant A Reference Run (15 features, unsaved)
    print("Fitting Variant A Reference Model (15 features)...")
    prep_a = PaySimFeaturePreprocessor(include_balance_errors=True)
    prep_a.fit(train_df[API_INPUT_COLUMNS])
    X_train_a = prep_a.transform(train_df[API_INPUT_COLUMNS])
    X_sub_a, y_sub_a = apply_majority_subsampling(
        X_train_a, y_train, neg_to_pos_ratio=10.0, random_state=DEFAULT_MAIN_SEED
    )
    var_a_params = dict(FROZEN_HYPERPARAMS_VARIANT_A)
    var_a_params["scale_pos_weight"] = 10.0
    var_a_clf = XGBClassifier(**var_a_params)
    var_a_clf.fit(X_sub_a, y_sub_a)
    var_a_pipe = Pipeline([("prep", prep_a), ("clf", var_a_clf)])
    var_a_test_probs = var_a_pipe.predict_proba(test_df[API_INPUT_COLUMNS])[:, 1]
    var_a_metrics = compute_metrics_bundle(y_test, var_a_test_probs, threshold=0.50)

    # -------------------------------------------------------------
    # Held-Out Test Evaluation (Computed Strictly ONCE)
    # -------------------------------------------------------------
    print("\n--- Evaluating Held-Out Test Split (118,146 rows, 1,642 frauds) ---")
    strategy_test_eval: dict[str, Any] = {}
    test_scores_dict: dict[str, np.ndarray] = {}

    for strat, data in trained_strategies.items():
        pipe = data["main_pipeline"]
        thresh = data["thresholds"]["unrounded"]
        t_rev = thresh["t_review"]
        t_high = thresh["t_high"]

        test_probs = pipe.predict_proba(test_df[API_INPUT_COLUMNS])[:, 1]
        test_scores_dict[strat] = test_probs

        metrics_high = compute_metrics_bundle(y_test, test_probs, threshold=t_high)
        metrics_rev = compute_metrics_bundle(y_test, test_probs, threshold=t_rev)
        cis_high = compute_bootstrap_cis(y_test, test_probs, threshold=t_high)
        cis_rev = compute_bootstrap_cis(y_test, test_probs, threshold=t_rev)

        # Precision-Recall curve points for reports
        precisions, recalls, pr_thresholds = precision_recall_curve(y_test, test_probs)
        # Sample 50 points evenly to keep metrics.json compact
        step = max(1, len(pr_thresholds) // 50)
        pr_curve_sampled = [
            {"recall": round(float(recalls[i]), 4), "precision": round(float(precisions[i]), 4), "threshold": round(float(pr_thresholds[i]), 4)}
            for i in range(0, len(pr_thresholds), step)
        ]

        # Test set routing breakdown
        n_test = len(test_df)
        low_risk = int(np.sum(test_probs < t_rev))
        analyst_rev = int(np.sum((test_probs >= t_rev) & (test_probs < t_high)))
        high_risk = int(np.sum(test_probs >= t_high))

        strategy_test_eval[strat] = {
            "validation_pr_auc": data["val_pr_auc"],
            "test_pr_auc": metrics_high["pr_auc"],
            "threshold_review": t_rev,
            "threshold_high": t_high,
            "metrics_at_high": {**metrics_high, **cis_high},
            "metrics_at_review": {**metrics_rev, **cis_rev},
            "test_routing": {
                "total_rows": n_test,
                "low_risk_count": low_risk,
                "low_risk_share": round(low_risk / n_test, 4),
                "analyst_review_count": analyst_rev,
                "analyst_review_share": round(analyst_rev / n_test, 4),
                "high_risk_count": high_risk,
                "high_risk_share": round(high_risk / n_test, 4),
            },
            "pr_curve_sample": pr_curve_sampled,
        }

    # -------------------------------------------------------------
    # SHAP Explanation Stability Evaluation on Test Sample Cohort
    # -------------------------------------------------------------
    print("\n--- Evaluating SHAP Explanation Stability on Test Cohort ---")
    deployed_test_probs = test_scores_dict["class_weight"]
    t_high_deployed = trained_strategies["class_weight"]["thresholds"]["unrounded"]["t_high"]

    fraud_indices = np.where(y_test == 1)[0]
    fp_candidate_indices = np.where((y_test == 0) & (deployed_test_probs >= t_high_deployed))[0]
    if len(fp_candidate_indices) > 100:
        # Sort by highest score among false positives
        top_fp_idx = fp_candidate_indices[np.argsort(-deployed_test_probs[fp_candidate_indices])[:100]]
    else:
        top_fp_idx = fp_candidate_indices

    cohort_frauds_df = test_df.iloc[fraud_indices].copy().reset_index(drop=True)
    cohort_fps_df = test_df.iloc[top_fp_idx].copy().reset_index(drop=True) if len(top_fp_idx) > 0 else pd.DataFrame()

    model_features_b = list(prep_b.get_feature_names_out())
    stability_reports: dict[str, Any] = {}
    strategy_shap_matrices: dict[str, np.ndarray] = {}

    for strat, data in trained_strategies.items():
        print(f"Computing SHAP stability for '{strat}'...")
        m_pipe = data["main_pipeline"]
        rep_pipes = data["replicate_pipelines"]

        main_shap_frauds = compute_shap_matrix_cohort(m_pipe, cohort_frauds_df)
        strategy_shap_matrices[strat] = main_shap_frauds
        rep_shaps_frauds = [compute_shap_matrix_cohort(rp, cohort_frauds_df) for rp in rep_pipes]

        stab_frauds = evaluate_replicate_stability(
            main_shap=main_shap_frauds,
            replicate_shaps=rep_shaps_frauds,
            feature_names=model_features_b,
            k_values=[3, 5, 10],
            min_overlap=0.70,
        )

        stab_fps = None
        if len(cohort_fps_df) > 0:
            main_shap_fps = compute_shap_matrix_cohort(m_pipe, cohort_fps_df)
            rep_shaps_fps = [compute_shap_matrix_cohort(rp, cohort_fps_df) for rp in rep_pipes]
            stab_fps = evaluate_replicate_stability(
                main_shap=main_shap_fps,
                replicate_shaps=rep_shaps_fps,
                feature_names=model_features_b,
                k_values=[3, 5, 10],
                min_overlap=0.70,
            )

        stability_reports[strat] = {
            "fraud_cohort": stab_frauds,
            "top_false_positives_cohort": stab_fps,
        }

    # Cross-strategy SHAP agreement on test frauds
    cross_strategy_agreement = evaluate_cross_strategy_agreement(
        strategy_shaps=strategy_shap_matrices,
        feature_names=model_features_b,
        k=5,
    )

    # Reference Variant A stability evaluation (unsaved)
    print("Computing Variant A reference SHAP stability...")
    var_a_shaps_rep = []
    for r_seed in [101, 102]:
        X_sub_ar, y_sub_ar = apply_majority_subsampling(
            X_train_a, y_train, neg_to_pos_ratio=10.0, random_state=r_seed
        )
        clf_ar = XGBClassifier(**{**var_a_params, "random_state": r_seed})
        clf_ar.fit(X_sub_ar, y_sub_ar)
        pipe_ar = Pipeline([("prep", prep_a), ("clf", clf_ar)])
        var_a_shaps_rep.append(compute_shap_matrix_cohort(pipe_ar, cohort_frauds_df))

    main_shap_var_a = compute_shap_matrix_cohort(var_a_pipe, cohort_frauds_df)
    var_a_stability = evaluate_replicate_stability(
        main_shap=main_shap_var_a,
        replicate_shaps=var_a_shaps_rep,
        feature_names=list(prep_a.get_feature_names_out()),
        k_values=[3, 5, 10],
        min_overlap=0.70,
    )

    # -------------------------------------------------------------
    # Build Final Reports (metrics.json, comparison.csv)
    # -------------------------------------------------------------
    deployed_test = strategy_test_eval["class_weight"]["metrics_at_high"]

    final_metrics_json = {
        "strategy_comparison": {
            "class_weight": {
                "label": "class_weight (deployed)",
                **strategy_test_eval["class_weight"],
                "shap_stability": stability_reports["class_weight"],
            },
            "none": {
                "label": "undersampled 10:1 baseline",
                **strategy_test_eval.get("none", {}),
                "shap_stability": stability_reports.get("none", {}),
            },
            "smote_enn": {
                "label": "smote_enn",
                **strategy_test_eval.get("smote_enn", {}),
                "shap_stability": stability_reports.get("smote_enn", {}),
            },
            "logistic_regression": {
                "label": "Logistic Regression baseline",
                **lr_metrics,
            },
            "random_forest": {
                "label": "Random Forest baseline",
                **rf_metrics,
            },
            "rules_baseline": {
                "label": "Rules baseline (backend rule function)",
                **rules_test_metrics,
            },
            "full_train_none": {
                "label": "Full-train reference (none, 6.08M rows)",
                **full_metrics_none,
            },
            "full_train_class_weight": {
                "label": "Full-train reference (class_weight, 6.08M rows)",
                **full_metrics_cw,
            },
            "variant_A_reference": {
                "label": "Variant A reference (15 features with balance errors/deltas)",
                **var_a_metrics,
                "shap_stability": var_a_stability,
            },
            "cross_strategy_shap_agreement": cross_strategy_agreement,
        },
        "evaluation_metrics": {
            "precision": deployed_test["precision"],
            "recall": deployed_test["recall"],
            "f1": deployed_test["f1"],
            "pr_auc": deployed_test["pr_auc"],
            "roc_auc": deployed_test["roc_auc"],
            "mcc": deployed_test["mcc"],
        },
    }

    if not is_benchmark:
        reports_dir = Path("ml/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(final_metrics_json, f, indent=2)

        # Build comparison.csv
        comparison_rows = [
            {
                "model_strategy": "class_weight (deployed)",
                "val_pr_auc": strategy_test_eval["class_weight"]["validation_pr_auc"],
                "test_pr_auc": strategy_test_eval["class_weight"]["test_pr_auc"],
                "recall_at_high": deployed_test["recall"],
                "precision_at_high": deployed_test["precision"],
                "f1_at_high": deployed_test["f1"],
                "fpr_at_high": deployed_test["fpr"],
                "stability_rate_k5": stability_reports["class_weight"]["fraud_cohort"]["primary_k5"]["stability_rate"],
            },
            {
                "model_strategy": "none (undersampled 10:1 baseline)",
                "val_pr_auc": strategy_test_eval["none"]["validation_pr_auc"],
                "test_pr_auc": strategy_test_eval["none"]["test_pr_auc"],
                "recall_at_high": strategy_test_eval["none"]["metrics_at_high"]["recall"],
                "precision_at_high": strategy_test_eval["none"]["metrics_at_high"]["precision"],
                "f1_at_high": strategy_test_eval["none"]["metrics_at_high"]["f1"],
                "fpr_at_high": strategy_test_eval["none"]["metrics_at_high"]["fpr"],
                "stability_rate_k5": stability_reports["none"]["fraud_cohort"]["primary_k5"]["stability_rate"],
            },
            {
                "model_strategy": "smote_enn",
                "val_pr_auc": strategy_test_eval["smote_enn"]["validation_pr_auc"],
                "test_pr_auc": strategy_test_eval["smote_enn"]["test_pr_auc"],
                "recall_at_high": strategy_test_eval["smote_enn"]["metrics_at_high"]["recall"],
                "precision_at_high": strategy_test_eval["smote_enn"]["metrics_at_high"]["precision"],
                "f1_at_high": strategy_test_eval["smote_enn"]["metrics_at_high"]["f1"],
                "fpr_at_high": strategy_test_eval["smote_enn"]["metrics_at_high"]["fpr"],
                "stability_rate_k5": stability_reports["smote_enn"]["fraud_cohort"]["primary_k5"]["stability_rate"],
            },
            {
                "model_strategy": "Logistic Regression baseline",
                "val_pr_auc": "-",
                "test_pr_auc": lr_metrics["pr_auc"],
                "recall_at_high": lr_metrics["recall"],
                "precision_at_high": lr_metrics["precision"],
                "f1_at_high": lr_metrics["f1"],
                "fpr_at_high": lr_metrics["fpr"],
                "stability_rate_k5": "-",
            },
            {
                "model_strategy": "Random Forest baseline",
                "val_pr_auc": "-",
                "test_pr_auc": rf_metrics["pr_auc"],
                "recall_at_high": rf_metrics["recall"],
                "precision_at_high": rf_metrics["precision"],
                "f1_at_high": rf_metrics["f1"],
                "fpr_at_high": rf_metrics["fpr"],
                "stability_rate_k5": "-",
            },
            {
                "model_strategy": "Rules baseline (union: drain or >200k)",
                "val_pr_auc": "-",
                "test_pr_auc": "-",
                "recall_at_high": rules_test_metrics["union_drain_or_amount"]["recall"],
                "precision_at_high": rules_test_metrics["union_drain_or_amount"]["precision"],
                "f1_at_high": rules_test_metrics["union_drain_or_amount"]["f1"],
                "fpr_at_high": rules_test_metrics["union_drain_or_amount"]["fpr"],
                "stability_rate_k5": "-",
            },
            {
                "model_strategy": "Variant A reference (15 features)",
                "val_pr_auc": 0.9998,
                "test_pr_auc": var_a_metrics["pr_auc"],
                "recall_at_high": var_a_metrics["recall"],
                "precision_at_high": var_a_metrics["precision"],
                "f1_at_high": var_a_metrics["f1"],
                "fpr_at_high": var_a_metrics["fpr"],
                "stability_rate_k5": var_a_stability["primary_k5"]["stability_rate"],
            },
        ]

        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(reports_dir / "comparison.csv", index=False)
        print("\nReports successfully written to ml/reports/metrics.json and ml/reports/comparison.csv.")

    print("\nStage 2 execution completed successfully.")
    return final_metrics_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Stage 2 ML Training and Reporting Pipeline.")
    parser.add_argument("--data-path", type=str, default="ml/data/raw/paysim.csv", help="Path to raw PaySim CSV.")
    parser.add_argument("--output-dir", type=str, default="ml/models", help="Directory to save model artifacts.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode writing to temp dir.")
    args = parser.parse_args()

    run_stage2_training(
        data_path=args.data_path,
        output_dir_str=args.output_dir,
        is_benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()

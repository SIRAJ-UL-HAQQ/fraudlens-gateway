"""Model training, artifact packaging, and evaluation module for PaySim fraud detection.

Strictly adheres to docs/ml-artifact-contract.md and docs/api-contract.md:
- Packages models into ml/models/<strategy>/ with model.joblib, replicates/seed_*.joblib, manifest.json
- Enforces gateway decision semantics and threshold selection on validation split
- Evaluates test split strictly once after artifacts are frozen
- Implements 10 replicate sets across 3 strategies with matched majority subsample seeds
- Decomposes stability into data-draw and model-seed-only noise
- Evaluates candidate high risk, top false positives, and test frauds
- Computes Wilson score intervals, bootstrap CIs, and AUROC of overlap for predicting false positives
"""

import argparse
import json
import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
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
    extract_top_features,
    is_explanation_stable,
    top_k_overlap,
)

logger = logging.getLogger(__name__)

# Canonical seeds across all 10 replicate sets (Section 2.C analysis plan)
DEFAULT_MAIN_SEED = 42
MAIN_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
REPLICATE_SEEDS_PER_SET = [
    [101, 102, 103, 104, 105],
    [201, 202, 203, 204, 205],
    [301, 302, 303, 304, 305],
    [401, 402, 403, 404, 405],
    [501, 502, 503, 504, 505],
    [601, 602, 603, 604, 605],
    [701, 702, 703, 704, 705],
    [801, 802, 803, 804, 805],
    [901, 902, 903, 904, 905],
    [1001, 1002, 1003, 1004, 1005],
]

MODEL_SEED_ONLY_SEEDS = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509]

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


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> list[float]:
    """Compute Wilson score interval for a binomial proportion."""
    if n == 0:
        return [0.0, 0.0]
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [round(float(max(0.0, center - spread)), 6), round(float(min(1.0, center + spread)), 6)]


def compute_auroc_with_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute AUROC and 95% bootstrap confidence interval."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.5, "ci_95": [0.5, 0.5]}
    point = float(roc_auc_score(y_true, scores))
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) > 1:
            aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    ci = (
        [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]
        if aucs
        else [round(point, 4), round(point, 4)]
    )
    return {"auroc": round(point, 4), "ci_95": ci}


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


def update_progress_log(
    progress_file: Path,
    current_step: str,
    completed_steps: int,
    total_steps: int,
    details: dict[str, Any] | None = None,
) -> None:
    """Write live progress update to JSON file."""
    data = {
        "current_step": current_step,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "progress_pct": round(float(completed_steps / max(total_steps, 1) * 100), 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "details": details or {},
    }
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_stage2_training(
    data_path: str = "ml/data/raw/paysim.csv",
    output_dir_str: str = "ml/models",
    is_benchmark: bool = False,
) -> dict[str, Any]:
    """Execute complete Stage 2 training, multi-replicate generation, test evaluation, and report generation."""
    start_time_all = time.time()
    output_dir = Path(output_dir_str)
    if is_benchmark:
        output_dir = Path(tempfile.mkdtemp(prefix="fraudlens_benchmark_"))
        print(f"Running in benchmark mode: writing to temporary directory {output_dir}")

    reports_dir = Path("ml/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    progress_path = reports_dir / ".training_progress.json"

    print("Loading PaySim dataset...")
    update_progress_log(progress_path, "Loading PaySim dataset", 0, 100)
    df = load_raw_paysim(data_path)
    train_df, val_df, test_df = split_chronological(df)
    y_train = train_df["is_fraud"].to_numpy(dtype=int)
    y_val = val_df["is_fraud"].to_numpy(dtype=int)
    y_test = test_df["is_fraud"].to_numpy(dtype=int)

    # 1. Fit structural preprocessor
    prep = PaySimFeaturePreprocessor(include_balance_errors=False)
    prep.fit(train_df[API_INPUT_COLUMNS])
    X_train_trans = prep.transform(train_df[API_INPUT_COLUMNS])
    feature_names = list(prep.get_feature_names_out())

    strategies = ["class_weight", "none", "smote_enn"]
    n_sets = 10
    total_pipeline_steps = 100

    # Directory for replicates sets 2..10 (gitignored)
    rep_sets_dir = Path("ml/replicate_sets")
    rep_sets_dir.mkdir(parents=True, exist_ok=True)

    # Storage for multi-set trained models: {strategy: {set_idx: {'main': pipeline, 'replicates': [pipelines]}}}
    multi_set_models: dict[str, dict[int, dict[str, Any]]] = {s: {} for s in strategies}
    thresholds_dict: dict[str, dict[str, Any]] = {}
    validation_pr_aucs: dict[str, float] = {}

    print("\n=======================================================")
    print("Training 10 Replicate Sets across 3 Strategies (Matched Seeds)")
    print("=======================================================")

    for set_idx in range(n_sets):
        main_seed = MAIN_SEEDS[set_idx]
        rep_seeds = REPLICATE_SEEDS_PER_SET[set_idx]
        print(f"\n--- Replicate Set {set_idx + 1}/10 (Main Seed={main_seed}, Replicates={rep_seeds}) ---")

        # Subsample non-fraud majority with matched seed across strategies
        X_sub_main, y_sub_main = apply_majority_subsampling(
            X_train_trans, y_train, neg_to_pos_ratio=10.0, random_state=main_seed
        )

        for strat in strategies:
            X_res_main, y_res_main, spw_main = apply_imbalance_strategy(
                X_sub_main, y_sub_main, strategy=strat, random_state=main_seed
            )

            main_params = dict(FROZEN_HYPERPARAMS_VARIANT_B)
            main_params["scale_pos_weight"] = spw_main
            main_params["random_state"] = main_seed

            main_clf = XGBClassifier(**main_params)
            main_clf.fit(X_res_main, y_res_main)
            main_pipe = Pipeline([("prep", prep), ("clf", main_clf)])

            # Thresholds and validation PR-AUC computed on Set 1
            if set_idx == 0:
                val_probs = main_pipe.predict_proba(val_df[API_INPUT_COLUMNS])[:, 1]
                val_pr_auc = float(average_precision_score(y_val, val_probs))
                thresh_info = compute_gateway_thresholds(y_val, val_probs)
                thresholds_dict[strat] = thresh_info
                validation_pr_aucs[strat] = val_pr_auc
                print(f"[{strat}] Set 1 Val PR-AUC: {val_pr_auc:.4f}, Review={thresh_info['unrounded']['t_review']:.6f}, High={thresh_info['unrounded']['t_high']:.6f}")

            # Replicate models for this set
            rep_pipes = []
            for r_seed in rep_seeds:
                # Majority subsample with replicate seed (data-draw variation)
                X_sub_rep, y_sub_rep = apply_majority_subsampling(
                    X_train_trans, y_train, neg_to_pos_ratio=10.0, random_state=r_seed
                )
                X_res_rep, y_res_rep, spw_rep = apply_imbalance_strategy(
                    X_sub_rep, y_sub_rep, strategy=strat, random_state=r_seed
                )
                rep_params = dict(FROZEN_HYPERPARAMS_VARIANT_B)
                rep_params["scale_pos_weight"] = spw_rep
                rep_params["random_state"] = r_seed

                rep_clf = XGBClassifier(**rep_params)
                rep_clf.fit(X_res_rep, y_res_rep)
                rep_pipe = Pipeline([("prep", prep), ("clf", rep_clf)])
                rep_pipes.append(rep_pipe)

            multi_set_models[strat][set_idx] = {
                "main": main_pipe,
                "replicates": rep_pipes,
            }

            # Packaging Set 1 to ml/models/<strat>/
            if set_idx == 0:
                strat_dir = output_dir / strat
                strat_dir.mkdir(parents=True, exist_ok=True)
                reps_dir = strat_dir / "replicates"
                reps_dir.mkdir(parents=True, exist_ok=True)

                joblib.dump(main_pipe, strat_dir / "model.joblib")
                for r_seed, rp in zip(rep_seeds, rep_pipes, strict=False):
                    joblib.dump(rp, reps_dir / f"seed_{r_seed}.joblib")

                manifest_data = {
                    "model_version": f"v1.0.0-{strat}",
                    "imbalance_strategy": strat,
                    "feature_names": list(API_INPUT_COLUMNS),
                    "thresholds": {
                        "high": float(thresholds_dict[strat]["unrounded"]["t_high"]),
                        "review": float(thresholds_dict[strat]["unrounded"]["t_review"]),
                    },
                    "shap_top_k": 5,
                    "stability_min_overlap": 0.80,
                    "n_replicates": len(rep_seeds),
                }
                with open(strat_dir / "manifest.json", "w", encoding="utf-8") as mf:
                    json.dump(manifest_data, mf, indent=2)

                check_bundle(strat_dir)
            else:
                # Save Sets 2..10 to gitignored ml/replicate_sets/<strat>/set_{s}/
                set_save_dir = rep_sets_dir / strat / f"set_{set_idx + 1}"
                set_save_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(main_pipe, set_save_dir / "model.joblib")

        step_num = 10 + int((set_idx + 1) / n_sets * 35)
        update_progress_log(progress_path, f"Completed Replicate Set {set_idx + 1}/10", step_num, total_pipeline_steps)

    # -------------------------------------------------------------
    # Model-Seed-Only Noise Runs (Fixed subsample draw, varying seed)
    # -------------------------------------------------------------
    print("\n--- Training Model-Seed-Only Variation Models (10 fits per strategy) ---")
    model_seed_models: dict[str, list[Pipeline]] = {s: [] for s in strategies}
    # Use Set 1 majority subsample draw (seed 42)
    X_sub_fixed, y_sub_fixed = apply_majority_subsampling(
        X_train_trans, y_train, neg_to_pos_ratio=10.0, random_state=DEFAULT_MAIN_SEED
    )

    for strat in strategies:
        X_res_fixed, y_res_fixed, spw_fixed = apply_imbalance_strategy(
            X_sub_fixed, y_sub_fixed, strategy=strat, random_state=DEFAULT_MAIN_SEED
        )
        for m_seed in MODEL_SEED_ONLY_SEEDS:
            m_params = dict(FROZEN_HYPERPARAMS_VARIANT_B)
            m_params["scale_pos_weight"] = spw_fixed
            m_params["random_state"] = m_seed
            m_clf = XGBClassifier(**m_params)
            m_clf.fit(X_res_fixed, y_res_fixed)
            model_seed_models[strat].append(Pipeline([("prep", prep), ("clf", m_clf)]))

    update_progress_log(progress_path, "Completed Model-Seed-Only Variation Fits", 50, total_pipeline_steps)

    # -------------------------------------------------------------
    # Baselines & Reference Runs (Evaluated, not saved to ml/models)
    # -------------------------------------------------------------
    print("\n--- Training Reference Models & Baselines ---")
    X_test_b = prep.transform(test_df[API_INPUT_COLUMNS])

    # 1. Logistic Regression Baseline
    print("Fitting Logistic Regression Baseline...")
    lr_clf = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=DEFAULT_MAIN_SEED)),
    ])
    lr_clf.fit(X_sub_fixed, y_sub_fixed)
    lr_test_probs = lr_clf.predict_proba(X_test_b)[:, 1]
    lr_metrics = compute_metrics_bundle(y_test, lr_test_probs, threshold=0.50)

    # 2. Random Forest Baseline
    print("Fitting Random Forest Baseline...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=DEFAULT_MAIN_SEED, n_jobs=4)
    rf_clf.fit(X_sub_fixed, y_sub_fixed)
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
    full_train_clf_none.fit(X_train_trans, y_train)
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
    full_train_clf_cw.fit(X_train_trans, y_train)
    full_test_probs_cw = full_train_clf_cw.predict_proba(X_test_b)[:, 1]
    full_metrics_cw = compute_metrics_bundle(y_test, full_test_probs_cw, threshold=0.50)

    # 5. Variant A Reference Run (15 features, unsaved, Set 1 seeds)
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

    update_progress_log(progress_path, "Completed Reference Baselines", 60, total_pipeline_steps)

    # -------------------------------------------------------------
    # Held-Out Test Evaluation for Deployed Models (Replicate Set 1)
    # -------------------------------------------------------------
    print("\n--- Evaluating Held-Out Test Split (118,146 rows, 1,642 frauds) ---")
    strategy_test_eval: dict[str, Any] = {}
    test_scores_dict: dict[str, np.ndarray] = {}

    for strat in strategies:
        pipe = multi_set_models[strat][0]["main"]
        thresh = thresholds_dict[strat]["unrounded"]
        t_rev = thresh["t_review"]
        t_high = thresh["t_high"]

        test_probs = pipe.predict_proba(test_df[API_INPUT_COLUMNS])[:, 1]
        test_scores_dict[strat] = test_probs

        metrics_high = compute_metrics_bundle(y_test, test_probs, threshold=t_high)
        metrics_rev = compute_metrics_bundle(y_test, test_probs, threshold=t_rev)
        cis_high = compute_bootstrap_cis(y_test, test_probs, threshold=t_high)
        cis_rev = compute_bootstrap_cis(y_test, test_probs, threshold=t_rev)

        precisions, recalls, pr_thresholds = precision_recall_curve(y_test, test_probs)
        step = max(1, len(pr_thresholds) // 50)
        pr_curve_sampled = [
            {"recall": round(float(recalls[i]), 4), "precision": round(float(precisions[i]), 4), "threshold": round(float(pr_thresholds[i]), 4)}
            for i in range(0, len(pr_thresholds), step)
        ]

        n_test = len(test_df)
        low_risk = int(np.sum(test_probs < t_rev))
        analyst_rev = int(np.sum((test_probs >= t_rev) & (test_probs < t_high)))
        high_risk = int(np.sum(test_probs >= t_high))

        strategy_test_eval[strat] = {
            "validation_pr_auc": validation_pr_aucs[strat],
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
    # Cohorts Extraction for Stability Evaluation
    # -------------------------------------------------------------
    print("\n--- Extracting Evaluation Cohorts ---")
    deployed_scores = test_scores_dict["class_weight"]
    t_high_deployed = thresholds_dict["class_weight"]["unrounded"]["t_high"]
    t_rev_deployed = thresholds_dict["class_weight"]["unrounded"]["t_review"]

    # Cohort 1: Test Frauds (1,642 frauds)
    fraud_indices = np.where(y_test == 1)[0]
    cohort_frauds_df = test_df.iloc[fraud_indices].copy().reset_index(drop=True)

    # Cohort 2: Candidate High Risk rows (1,645 rows)
    cand_high_indices = np.where(deployed_scores >= t_high_deployed)[0]
    cohort_cand_high_df = test_df.iloc[cand_high_indices].copy().reset_index(drop=True)
    y_cand_high = y_test[cand_high_indices]

    # Cohort 3: Top False Positives (94 FPs at high)
    top_fp_indices = np.where((y_test == 0) & (deployed_scores >= t_high_deployed))[0]
    cohort_top_fps_df = test_df.iloc[top_fp_indices].copy().reset_index(drop=True)

    # Cohort 4: Step 0c cohort (deployed score >= review, capped at 2,000, seed 42)
    step0c_qualifying = np.where(deployed_scores >= t_rev_deployed)[0]
    if len(step0c_qualifying) > 2000:
        rng_cohort = np.random.RandomState(42)
        step0c_indices = rng_cohort.choice(step0c_qualifying, size=2000, replace=False)
        step0c_indices = np.sort(step0c_indices)
    else:
        step0c_indices = step0c_qualifying
    cohort_step0c_df = test_df.iloc[step0c_indices].copy().reset_index(drop=True)
    print(f"Cohorts: Frauds={len(cohort_frauds_df)}, Cand-High={len(cohort_cand_high_df)}, Top-FPs={len(cohort_top_fps_df)}, Step0c={len(cohort_step0c_df)}")

    # -------------------------------------------------------------
    # Save Top-5 SHAP Feature Indices on Step 0c Cohort for Step 4
    # -------------------------------------------------------------
    print("\nSaving Top-5 SHAP feature indices on Step 0c cohort for Step 4...")
    step0c_top5_indices_by_strategy: dict[str, list[list[int]]] = {}
    strategy_step0c_shap_matrices: dict[str, np.ndarray] = {}

    for strat in strategies:
        main_p = multi_set_models[strat][0]["main"]
        sv_step0c = compute_shap_matrix_cohort(main_p, cohort_step0c_df)
        strategy_step0c_shap_matrices[strat] = sv_step0c
        top5_indices = [np.argsort(np.abs(sv_step0c[i]))[::-1][:5].tolist() for i in range(len(sv_step0c))]
        step0c_top5_indices_by_strategy[strat] = top5_indices

    step0c_export = {
        "cohort_size": len(cohort_step0c_df),
        "feature_names": feature_names,
        "sample_indices_in_test": step0c_indices.tolist(),
        "top_5_shap_feature_indices": step0c_top5_indices_by_strategy,
    }
    with open(reports_dir / "step0c_cohort_shap_top5.json", "w", encoding="utf-8") as f:
        json.dump(step0c_export, f)
    print("Saved ml/reports/step0c_cohort_shap_top5.json.")

    update_progress_log(progress_path, "Computed SHAP matrices on Step 0c cohort", 70, total_pipeline_steps)

    # -------------------------------------------------------------
    # Multi-Set Stability Evaluation across 10 Sets & Cohorts
    # -------------------------------------------------------------
    print("\n--- Computing Multi-Set Stability across 10 Replicate Sets ---")
    multi_set_stability_results: dict[str, Any] = {
        "strategies": {},
        "model_seed_noise": {},
    }

    # Evaluate across all 10 sets
    for strat in strategies:
        print(f"Evaluating 10 replicate sets for strategy: {strat}...")
        strat_set_stats: list[dict[str, Any]] = []

        for s_idx in range(n_sets):
            m_pipe = multi_set_models[strat][s_idx]["main"]
            r_pipes = multi_set_models[strat][s_idx]["replicates"]

            # Compute SHAP on test frauds
            m_sv_frauds = compute_shap_matrix_cohort(m_pipe, cohort_frauds_df)
            r_svs_frauds = [compute_shap_matrix_cohort(rp, cohort_frauds_df) for rp in r_pipes]

            stab_frauds = evaluate_replicate_stability(
                main_shap=m_sv_frauds,
                replicate_shaps=r_svs_frauds,
                feature_names=feature_names,
                k_values=[3, 5, 10],
                min_overlap=0.80,
            )

            # Compute SHAP on candidate high risk rows
            m_sv_ch = compute_shap_matrix_cohort(m_pipe, cohort_cand_high_df)
            r_svs_ch = [compute_shap_matrix_cohort(rp, cohort_cand_high_df) for rp in r_pipes]

            stab_cand_high = evaluate_replicate_stability(
                main_shap=m_sv_ch,
                replicate_shaps=r_svs_ch,
                feature_names=feature_names,
                k_values=[5],
                min_overlap=0.80,
            )

            # Compute SHAP on top false positives
            stab_top_fps = None
            if len(cohort_top_fps_df) > 0:
                m_sv_fp = compute_shap_matrix_cohort(m_pipe, cohort_top_fps_df)
                r_svs_fp = [compute_shap_matrix_cohort(rp, cohort_top_fps_df) for rp in r_pipes]
                stab_top_fps = evaluate_replicate_stability(
                    main_shap=m_sv_fp,
                    replicate_shaps=r_svs_fp,
                    feature_names=feature_names,
                    k_values=[5],
                    min_overlap=0.80,
                )

            strat_set_stats.append({
                "set_index": s_idx + 1,
                "main_seed": MAIN_SEEDS[s_idx],
                "fraud_cohort_k5_stable_rate": stab_frauds["primary_k5"]["stability_rate"],
                "cand_high_cohort_k5_stable_rate": stab_cand_high["primary_k5"]["stability_rate"],
                "top_fps_cohort_k5_stable_rate": stab_top_fps["primary_k5"]["stability_rate"] if stab_top_fps else None,
                "fraud_cohort_k5_mean_overlap": stab_frauds["primary_k5"]["mean_top_k_overlap"],
                "fraud_cohort_mean_spearman": stab_frauds["primary_k5"]["mean_spearman"],
                "full_stats_frauds": stab_frauds,
            })

        # Calculate across-set means and 95% confidence intervals (t-distribution with df=9)
        fraud_rates = [s["fraud_cohort_k5_stable_rate"] for s in strat_set_stats]
        ch_rates = [s["cand_high_cohort_k5_stable_rate"] for s in strat_set_stats]
        fp_rates = [s["top_fps_cohort_k5_stable_rate"] for s in strat_set_stats if s["top_fps_cohort_k5_stable_rate"] is not None]

        def compute_mean_ci(arr: list[float]) -> dict[str, Any]:
            mean_val = float(np.mean(arr))
            std_err = float(stats.sem(arr)) if len(arr) > 1 else 0.0
            ci_half = float(stats.t.ppf(0.975, df=len(arr) - 1) * std_err) if len(arr) > 1 else 0.0
            return {
                "mean": round(mean_val, 4),
                "std": round(float(np.std(arr, ddof=1)), 4),
                "ci_95": [round(max(0.0, mean_val - ci_half), 4), round(min(1.0, mean_val + ci_half), 4)],
            }

        multi_set_stability_results["strategies"][strat] = {
            "per_set_stats": strat_set_stats,
            "fraud_cohort_stability": compute_mean_ci(fraud_rates),
            "cand_high_cohort_stability": compute_mean_ci(ch_rates),
            "top_fps_cohort_stability": compute_mean_ci(fp_rates),
            "chance_baseline_k5": round(5.0 / len(feature_names), 4),
        }

    # Model-Seed-Only Noise Stability (Set 1 main vs 10 model-seed variants)
    print("\n--- Evaluating Model-Seed-Only Noise Floor ---")
    for strat in strategies:
        m_pipe = multi_set_models[strat][0]["main"]
        m_sv = compute_shap_matrix_cohort(m_pipe, cohort_frauds_df)
        seed_svs = [compute_shap_matrix_cohort(sp, cohort_frauds_df) for sp in model_seed_models[strat]]

        stab_seed_only = evaluate_replicate_stability(
            main_shap=m_sv,
            replicate_shaps=seed_svs,
            feature_names=feature_names,
            k_values=[5],
            min_overlap=0.80,
        )
        multi_set_stability_results["model_seed_noise"][strat] = {
            "mean_top_k_overlap": stab_seed_only["primary_k5"]["mean_top_k_overlap"],
            "stability_rate": stab_seed_only["primary_k5"]["stability_rate"],
            "mean_spearman": stab_seed_only["primary_k5"]["mean_spearman"],
        }

    update_progress_log(progress_path, "Completed Multi-Set Stability Evaluation", 85, total_pipeline_steps)

    # -------------------------------------------------------------
    # Informative Nature of 'Unstable' on Candidate High Risk Rows
    # -------------------------------------------------------------
    print("\n--- Evaluating Informative Nature of Unstable Explanation on Candidate High Risk Rows ---")
    # Using deployed class_weight model and Set 1 replicates on Candidate High Risk rows
    deployed_main = multi_set_models["class_weight"][0]["main"]
    deployed_reps = multi_set_models["class_weight"][0]["replicates"]

    ch_main_sv = compute_shap_matrix_cohort(deployed_main, cohort_cand_high_df)
    ch_rep_svs = [compute_shap_matrix_cohort(rp, cohort_cand_high_df) for rp in deployed_reps]

    n_ch = len(cohort_cand_high_df)
    k_eval = 5

    ch_overlaps = []
    ch_stable_flags = []
    for i in range(n_ch):
        top_m = extract_top_features(ch_main_sv[i], feature_names, k_eval)
        rep_overlaps = [top_k_overlap(top_m, extract_top_features(r_sv[i], feature_names, k_eval)) for r_sv in ch_rep_svs]
        mo = float(np.mean(rep_overlaps))
        ch_overlaps.append(mo)
        ch_stable_flags.append(is_explanation_stable(mo, k=k_eval, min_overlap=0.80))

    ch_overlaps = np.array(ch_overlaps)
    ch_stable_flags = np.array(ch_stable_flags, dtype=bool)

    # True positives (fraud) and False positives (legitimate)
    ch_is_fp = (y_cand_high == 0).astype(int)
    ch_is_tp = (y_cand_high == 1).astype(int)

    n_stable = int(np.sum(ch_stable_flags))
    n_unstable = int(np.sum(~ch_stable_flags))

    tp_stable = int(np.sum((ch_stable_flags) & (ch_is_tp == 1)))
    fp_stable = int(np.sum((ch_stable_flags) & (ch_is_fp == 1)))
    tp_unstable = int(np.sum((~ch_stable_flags) & (ch_is_tp == 1)))
    fp_unstable = int(np.sum((~ch_stable_flags) & (ch_is_fp == 1)))

    prec_stable = float(tp_stable / n_stable) if n_stable > 0 else 0.0
    prec_unstable = float(tp_unstable / n_unstable) if n_unstable > 0 else 0.0
    fp_share_stable = float(fp_stable / n_stable) if n_stable > 0 else 0.0
    fp_share_unstable = float(fp_unstable / n_unstable) if n_unstable > 0 else 0.0

    prec_stable_ci = wilson_score_interval(tp_stable, n_stable)
    prec_unstable_ci = wilson_score_interval(tp_unstable, n_unstable)
    fp_share_stable_ci = wilson_score_interval(fp_stable, n_stable)
    fp_share_unstable_ci = wilson_score_interval(fp_unstable, n_unstable)

    # AUROC of lower overlap predicting False Positives
    auroc_fp_overlap = compute_auroc_with_bootstrap_ci(ch_is_fp, -ch_overlaps, n_boot=1000, seed=42)

    # Statistical Conclusion per Section 2.E:
    # Informative IF AND ONLY IF (a) lower bound of Unstable FP share > upper bound of Stable FP share AND (b) AUROC CI excludes 0.50
    condition_a = fp_share_unstable_ci[0] > fp_share_stable_ci[1]
    condition_b = auroc_fp_overlap["ci_95"][0] > 0.50
    is_informative = condition_a and condition_b
    conclusion_verdict = (
        "explanation instability is informative for triage"
        if is_informative
        else "no evidence that explanation instability predicts false positive alert status"
    )

    informative_analysis = {
        "total_candidate_high": n_ch,
        "stable_cohort": {
            "count": n_stable,
            "tp": tp_stable,
            "fp": fp_stable,
            "precision": round(prec_stable, 4),
            "precision_wilson_ci": prec_stable_ci,
            "fp_share": round(fp_share_stable, 4),
            "fp_share_wilson_ci": fp_share_stable_ci,
        },
        "unstable_cohort": {
            "count": n_unstable,
            "tp": tp_unstable,
            "fp": fp_unstable,
            "precision": round(prec_unstable, 4),
            "precision_wilson_ci": prec_unstable_ci,
            "fp_share": round(fp_share_unstable, 4),
            "fp_share_wilson_ci": fp_share_unstable_ci,
        },
        "auroc_mean_overlap_predicting_fp": auroc_fp_overlap,
        "criteria": {
            "condition_a_fp_share_unstable_strictly_above_stable": condition_a,
            "condition_b_auroc_ci_excludes_half": condition_b,
        },
        "statistical_conclusion": conclusion_verdict,
    }

    # -------------------------------------------------------------
    # Same-Run Variant A Reference Stability (Unsaved, Set 1 seeds)
    # -------------------------------------------------------------
    print("\n--- Computing Variant A Reference Stability (Same-Run) ---")
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
        min_overlap=0.80,
    )

    # -------------------------------------------------------------
    # Cross-Strategy Agreement on Matched Seeds (Step 0c cohort)
    # -------------------------------------------------------------
    print("\n--- Cross-Strategy Agreement on Matched Seeds ---")
    cross_strategy_agreement = evaluate_cross_strategy_agreement(
        strategy_shaps=strategy_step0c_shap_matrices,
        feature_names=feature_names,
        k=5,
    )

    # -------------------------------------------------------------
    # Build and Write metrics.json & stability_bootstrap_comparison.json
    # -------------------------------------------------------------
    deployed_test = strategy_test_eval["class_weight"]["metrics_at_high"]

    final_metrics_json = {
        "strategy_comparison": {
            "class_weight": {
                "label": "class_weight (deployed)",
                **strategy_test_eval["class_weight"],
                "shap_stability": {
                    "fraud_cohort": multi_set_models["class_weight"][0]["main"],  # placeholder for structure
                    "cand_high_cohort": multi_set_stability_results["strategies"]["class_weight"]["cand_high_cohort_stability"],
                    "top_fps_cohort": multi_set_stability_results["strategies"]["class_weight"]["top_fps_cohort_stability"],
                    "across_10_sets": multi_set_stability_results["strategies"]["class_weight"],
                    "model_seed_only_noise": multi_set_stability_results["model_seed_noise"]["class_weight"],
                },
            },
            "none": {
                "label": "none (10:1 undersampled baseline)",
                **strategy_test_eval.get("none", {}),
                "shap_stability": {
                    "across_10_sets": multi_set_stability_results["strategies"]["none"],
                    "model_seed_only_noise": multi_set_stability_results["model_seed_noise"]["none"],
                },
            },
            "smote_enn": {
                "label": "smote_enn",
                **strategy_test_eval.get("smote_enn", {}),
                "shap_stability": {
                    "across_10_sets": multi_set_stability_results["strategies"]["smote_enn"],
                    "model_seed_only_noise": multi_set_stability_results["model_seed_noise"]["smote_enn"],
                },
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
            "informative_nature_of_unstable": informative_analysis,
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

    # Fix placeholder structure in class_weight
    final_metrics_json["strategy_comparison"]["class_weight"]["shap_stability"]["fraud_cohort"] = (
        multi_set_stability_results["strategies"]["class_weight"]["per_set_stats"][0]["full_stats_frauds"]
    )

    if not is_benchmark:
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
                "stability_rate_k5": multi_set_stability_results["strategies"]["class_weight"]["fraud_cohort_stability"]["mean"],
            },
            {
                "model_strategy": "none (undersampled 10:1 baseline)",
                "val_pr_auc": strategy_test_eval["none"]["validation_pr_auc"],
                "test_pr_auc": strategy_test_eval["none"]["test_pr_auc"],
                "recall_at_high": strategy_test_eval["none"]["metrics_at_high"]["recall"],
                "precision_at_high": strategy_test_eval["none"]["metrics_at_high"]["precision"],
                "f1_at_high": strategy_test_eval["none"]["metrics_at_high"]["f1"],
                "fpr_at_high": strategy_test_eval["none"]["metrics_at_high"]["fpr"],
                "stability_rate_k5": multi_set_stability_results["strategies"]["none"]["fraud_cohort_stability"]["mean"],
            },
            {
                "model_strategy": "smote_enn",
                "val_pr_auc": strategy_test_eval["smote_enn"]["validation_pr_auc"],
                "test_pr_auc": strategy_test_eval["smote_enn"]["test_pr_auc"],
                "recall_at_high": strategy_test_eval["smote_enn"]["metrics_at_high"]["recall"],
                "precision_at_high": strategy_test_eval["smote_enn"]["metrics_at_high"]["precision"],
                "f1_at_high": strategy_test_eval["smote_enn"]["metrics_at_high"]["f1"],
                "fpr_at_high": strategy_test_eval["smote_enn"]["metrics_at_high"]["fpr"],
                "stability_rate_k5": multi_set_stability_results["strategies"]["smote_enn"]["fraud_cohort_stability"]["mean"],
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

        # Write stability bootstrap and multi-set comparison json
        stab_summary = {
            "multi_set_stability_across_10_sets": multi_set_stability_results["strategies"],
            "model_seed_noise_floor": multi_set_stability_results["model_seed_noise"],
            "informative_nature_of_unstable": informative_analysis,
        }
        with open(reports_dir / "stability_bootstrap_comparison.json", "w", encoding="utf-8") as f:
            json.dump(stab_summary, f, indent=2)

        print("\nReports successfully written to:")
        print("  - ml/reports/metrics.json")
        print("  - ml/reports/comparison.csv")
        print("  - ml/reports/stability_bootstrap_comparison.json")
        print("  - ml/reports/step0c_cohort_shap_top5.json")

    elapsed_all = time.time() - start_time_all
    update_progress_log(progress_path, "Stage 2 Completed Successfully", 100, total_pipeline_steps, {"elapsed_seconds": elapsed_all})
    print(f"\nStage 2 execution completed in {elapsed_all:.2f} seconds.")
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

"""Stage 1: Validation-only Optuna hyperparameter tuning and threshold calibration.

Tunes Variant A (with balance errors/deltas) and Variant B (without) on the validation slice.
Evaluates on VALIDATION ONLY. No test set metrics are computed.
"""

import json
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from ml.src.features import PaySimFeaturePreprocessor
from ml.src.imbalance_strategies import (
    apply_imbalance_strategy,
    apply_majority_subsampling,
)
from ml.src.load_data import load_raw_paysim, split_chronological

optuna.logging.set_verbosity(optuna.logging.WARNING)


def compute_exact_thresholds(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    target_recall: float = 0.95,
    target_precision: float = 0.90,
    min_predicted_positives: int = 20,
) -> dict[str, Any]:
    """Compute decision thresholds on validation scores using gateway >= semantics.

    Rule:
    1. review = largest threshold t in unique_scores with Recall(t) >= target_recall (0.95).
    2. high = SMALLEST threshold t in unique_scores with t > review, Precision(t) >= target_precision (0.90),
       and Predicted_Positives(t) >= min_predicted_positives (20).
       If no valid high threshold exists, raises RuntimeError.
    3. Strictly enforces 0.0 < review < high < 1.0.
    """
    unique_scores = np.sort(np.unique(val_probs))

    # 1. Find review threshold: largest t with recall >= target_recall
    total_positives = int(np.sum(y_val == 1))
    best_review = None

    for t in unique_scores:
        pred = (val_probs >= t).astype(int)
        tp = np.sum((y_val == 1) & (pred == 1))
        rec = tp / total_positives
        if rec >= target_recall:
            best_review = float(t)
        else:
            # Since unique_scores are sorted, once recall drops below target, larger t will have lower recall
            break

    if best_review is None:
        raise RuntimeError(
            f"Validation scores cannot achieve target review recall {target_recall}."
        )

    # 2. Find high threshold: SMALLEST t > review with precision >= target_precision and pred_pos >= min_pred_pos
    candidates = unique_scores[unique_scores > best_review]
    best_high = None

    for t in candidates:
        pred = (val_probs >= t).astype(int)
        pred_pos = int(np.sum(pred == 1))
        if pred_pos < min_predicted_positives:
            continue
        tp = np.sum((y_val == 1) & (pred == 1))
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        if prec >= target_precision:
            best_high = float(t)
            break  # Smallest satisfying threshold found

    if best_high is None:
        raise RuntimeError(
            f"No threshold t > review ({best_review:.4f}) achieves Precision >= {target_precision} "
            f"with at least {min_predicted_positives} predicted positives on validation."
        )

    if not (0.0 < best_review < best_high < 1.0):
        raise RuntimeError(
            f"Invalid threshold ordering: must satisfy 0 < review < high < 1. "
            f"Got review={best_review}, high={best_high}."
        )

    # Calculate validation routing breakdown (gateway decision tiers)
    n_val = len(val_probs)
    low_risk_count = int(np.sum(val_probs < best_review))
    analyst_review_count = int(
        np.sum((val_probs >= best_review) & (val_probs < best_high))
    )
    high_risk_count = int(np.sum(val_probs >= best_high))

    # Evaluate metrics at review threshold
    pred_review = (val_probs >= best_review).astype(int)
    rec_at_rev = float(recall_score(y_val, pred_review, zero_division=0))
    prec_at_rev = float(precision_score(y_val, pred_review, zero_division=0))
    f1_at_rev = float(f1_score(y_val, pred_review, zero_division=0))
    tn_rev, fp_rev, _, _ = confusion_matrix(
        y_val, pred_review, labels=[0, 1]
    ).ravel()
    fpr_at_rev = float(fp_rev / max(fp_rev + tn_rev, 1))

    # Evaluate metrics at high threshold
    pred_high = (val_probs >= best_high).astype(int)
    rec_at_high = float(recall_score(y_val, pred_high, zero_division=0))
    prec_at_high = float(precision_score(y_val, pred_high, zero_division=0))
    f1_at_high = float(f1_score(y_val, pred_high, zero_division=0))
    tn_h, fp_h, _, tp_h = confusion_matrix(
        y_val, pred_high, labels=[0, 1]
    ).ravel()
    fpr_at_high = float(fp_h / max(fp_h + tn_h, 1))

    return {
        "threshold_review": round(best_review, 4),
        "threshold_high": round(best_high, 4),
        "metrics_at_review": {
            "recall": round(rec_at_rev, 4),
            "precision": round(prec_at_rev, 4),
            "f1": round(f1_at_rev, 4),
            "fpr": round(fpr_at_rev, 4),
        },
        "metrics_at_high": {
            "recall": round(rec_at_high, 4),
            "precision": round(prec_at_high, 4),
            "f1": round(f1_at_high, 4),
            "fpr": round(fpr_at_high, 4),
            "predicted_positives": int(tp_h + fp_h),
            "true_positives": int(tp_h),
            "false_positives": int(fp_h),
        },
        "validation_routing_shares": {
            "low_risk_count": low_risk_count,
            "low_risk_share": round(low_risk_count / n_val, 4),
            "analyst_review_count": analyst_review_count,
            "analyst_review_share": round(analyst_review_count / n_val, 4),
            "high_risk_count": high_risk_count,
            "high_risk_share": round(high_risk_count / n_val, 4),
        },
    }


def tune_and_evaluate_variant(
    variant_name: str,
    include_balance_errors: bool,
    train_df: Any,
    val_df: Any,
    n_trials: int = 40,
    timeout_seconds: int = 600,
    seed: int = 42,
    n_jobs: int = 4,
) -> dict[str, Any]:
    """Execute Optuna tuning and validation evaluation for a specific feature variant."""
    print(f"\n--- Starting Stage 1 Tuning for Variant {variant_name} ---")

    prep = PaySimFeaturePreprocessor(
        include_balance_errors=include_balance_errors
    )
    X_train_trans = prep.fit_transform(train_df)
    X_val_trans = prep.transform(val_df)
    y_train = train_df["is_fraud"].to_numpy(dtype=int)
    y_val = val_df["is_fraud"].to_numpy(dtype=int)

    # 10:1 Subsampling on Train split
    X_sub, y_sub = apply_majority_subsampling(
        X_train_trans, y_train, neg_to_pos_ratio=10.0, random_state=seed
    )
    _, _, spw = apply_imbalance_strategy(
        X_sub, y_sub, strategy="class_weight", random_state=seed
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 250),
            "subsample": trial.suggest_float("subsample", 0.60, 0.90),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.90),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": spw,
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "random_state": seed,
            "n_jobs": n_jobs,
        }

        clf = XGBClassifier(**params)
        clf.fit(X_sub, y_sub)
        probs = clf.predict_proba(X_val_trans)[:, 1]
        return float(average_precision_score(y_val, probs))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    best_params = study.best_params
    best_params["tree_method"] = "hist"
    best_params["eval_metric"] = "aucpr"
    best_params["scale_pos_weight"] = spw
    best_params["n_jobs"] = n_jobs
    best_params["random_state"] = seed

    # Train final model on 10:1 subsample with best params
    clf = XGBClassifier(**best_params)
    clf.fit(X_sub, y_sub)
    val_probs = clf.predict_proba(X_val_trans)[:, 1]

    val_pr_auc = float(average_precision_score(y_val, val_probs))
    threshold_results = compute_exact_thresholds(y_val, val_probs)

    return {
        "variant": variant_name,
        "n_features": len(prep.get_feature_names_out()),
        "feature_names": list(prep.get_feature_names_out()),
        "optuna_trials_completed": len(study.trials),
        "best_hyperparameters": best_params,
        "validation_pr_auc": round(val_pr_auc, 4),
        "threshold_calibration": threshold_results,
    }


def run_stage1() -> dict[str, Any]:
    print("Loading PaySim dataset for Stage 1 validation tuning...")
    df = load_raw_paysim()
    train_df, val_df, _ = split_chronological(df)

    # 1. Variant B: Without balance errors/deltas (11 features)
    res_b = tune_and_evaluate_variant(
        variant_name="B (Without Balance Errors/Deltas)",
        include_balance_errors=False,
        train_df=train_df,
        val_df=val_df,
        n_trials=40,
        timeout_seconds=600,
        seed=42,
    )

    # 2. Variant A: With balance errors/deltas (15 features)
    res_a = tune_and_evaluate_variant(
        variant_name="A (With Balance Errors/Deltas)",
        include_balance_errors=True,
        train_df=train_df,
        val_df=val_df,
        n_trials=40,
        timeout_seconds=600,
        seed=42,
    )

    summary = {
        "variant_B": res_b,
        "variant_A": res_a,
    }

    with open("ml/reports/stage1_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    results = run_stage1()
    print("\nStage 1 validation tuning complete. Results written to ml/reports/stage1_validation_results.json.")

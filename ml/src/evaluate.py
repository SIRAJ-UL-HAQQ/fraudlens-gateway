"""Model prediction evaluation metrics and threshold calibration module.

Computes Precision, Recall, F1, PR-AUC, ROC-AUC, MCC, and Confusion Matrix.
Contains single authoritative threshold selection function compute_gateway_thresholds.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Computes Precision, Recall, F1, PR-AUC, ROC-AUC, and MCC."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "mcc": round(float(mcc), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def compute_gateway_thresholds(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    target_recall: float = 0.95,
    target_precision: float = 0.90,
    min_predicted_positives: int = 20,
) -> dict[str, Any]:
    """Compute deterministic decision thresholds on validation scores using gateway >= semantics.

    Rule:
    1. review = largest threshold t in unique_scores with Recall(t) >= target_recall (0.95).
    2. high = SMALLEST threshold t in unique_scores with t > review, Precision(t) >= target_precision (0.90),
       and Predicted_Positives(t) >= min_predicted_positives (20).
    3. Strictly enforces 0.0 < review < high < 1.0.

    Returns dict containing full float precision thresholds, 4-decimal rounded thresholds,
    and validation routing breakdown.
    Raises RuntimeError if target recall or valid high threshold is not achievable.
    """
    n_total = len(val_probs)
    total_pos = int(np.sum(y_val == 1))

    sorted_idx = np.argsort(val_probs)
    sorted_scores = val_probs[sorted_idx]
    sorted_y = y_val[sorted_idx]

    cum_tp = np.cumsum(sorted_y[::-1])[::-1]

    unique_scores = np.unique(sorted_scores)
    left_idx = np.searchsorted(sorted_scores, unique_scores, side="left")

    pos = n_total - left_idx
    tp = cum_tp[left_idx]
    rec = tp / total_pos
    prec = tp / np.maximum(pos, 1)

    # 1. Largest t with recall >= target_recall
    valid_review_mask = rec >= target_recall
    if not np.any(valid_review_mask):
        raise RuntimeError(f"Cannot achieve target recall {target_recall} on validation.")
    best_review = float(unique_scores[valid_review_mask][-1])

    # 2. SMALLEST t > review with precision >= target_precision and pred_pos >= min_pred_pos
    valid_high_mask = (
        (unique_scores > best_review)
        & (prec >= target_precision)
        & (pos >= min_predicted_positives)
    )
    if not np.any(valid_high_mask):
        raise RuntimeError(
            f"No threshold t > review ({best_review}) achieves precision >= {target_precision} "
            f"with at least {min_predicted_positives} positives."
        )
    best_high = float(unique_scores[valid_high_mask][0])

    # Strict check: review < high
    assert best_review < best_high, (
        f"Threshold violation: review ({best_review}) must be strictly < high ({best_high})"
    )
    assert 0.0 < best_review < 1.0, f"review threshold {best_review} not in (0, 1)"
    assert 0.0 < best_high < 1.0, f"high threshold {best_high} not in (0, 1)"

    low_risk = int(np.sum(val_probs < best_review))
    analyst_review = int(np.sum((val_probs >= best_review) & (val_probs < best_high)))
    high_risk = int(np.sum(val_probs >= best_high))

    pred_rev = (val_probs >= best_review).astype(int)
    pred_h = (val_probs >= best_high).astype(int)
    _tn_h, fp_h, _, tp_h = confusion_matrix(y_val, pred_h, labels=[0, 1]).ravel()

    return {
        "threshold_review": best_review,
        "threshold_high": best_high,
        "unrounded": {
            "t_review": best_review,
            "t_high": best_high,
            "difference": best_high - best_review,
        },
        "rounded_4dec": {
            "t_review": round(best_review, 4),
            "t_high": round(best_high, 4),
        },
        "validation_routing": {
            "total_rows": n_total,
            "low_risk_count": low_risk,
            "low_risk_share": low_risk / n_total,
            "analyst_review_count": analyst_review,
            "analyst_review_share": analyst_review / n_total,
            "high_risk_count": high_risk,
            "high_risk_share": high_risk / n_total,
        },
        "metrics_at_review": {
            "recall": float(recall_score(y_val, pred_rev, zero_division=0)),
            "precision": float(precision_score(y_val, pred_rev, zero_division=0)),
            "f1": float(f1_score(y_val, pred_rev, zero_division=0)),
        },
        "metrics_at_high": {
            "recall": float(recall_score(y_val, pred_h, zero_division=0)),
            "precision": float(precision_score(y_val, pred_h, zero_division=0)),
            "f1": float(f1_score(y_val, pred_h, zero_division=0)),
            "predicted_positives": int(tp_h + fp_h),
            "true_positives": int(tp_h),
            "false_positives": int(fp_h),
        },
    }

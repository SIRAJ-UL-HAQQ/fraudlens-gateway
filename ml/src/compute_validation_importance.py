"""Computes Variant B validation feature importance and unrounded threshold calibration.

Validation-only script: does NOT evaluate or touch test split.
- Gain importance from XGBoost booster
- Permutation importance on validation slice (PR-AUC)
- Uses shared compute_gateway_thresholds
"""

import json
from typing import Any

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from ml.src.evaluate import compute_gateway_thresholds
from ml.src.features import PaySimFeaturePreprocessor
from ml.src.imbalance_strategies import (
    apply_imbalance_strategy,
    apply_majority_subsampling,
)
from ml.src.load_data import load_raw_paysim, split_chronological


def compute_validation_importance() -> dict[str, Any]:
    print("Loading PaySim dataset for validation importance evaluation...", flush=True)
    df = load_raw_paysim()
    train_df, val_df, _ = split_chronological(df)

    prep = PaySimFeaturePreprocessor(include_balance_errors=False)
    X_train_trans = prep.fit_transform(train_df)
    X_val_trans = prep.transform(val_df)
    y_train = train_df["is_fraud"].to_numpy(dtype=int)
    y_val = val_df["is_fraud"].to_numpy(dtype=int)
    feature_names = list(prep.get_feature_names_out())

    # 10:1 Subsample
    X_sub, y_sub = apply_majority_subsampling(
        X_train_trans, y_train, neg_to_pos_ratio=10.0, random_state=42
    )
    _, _, spw = apply_imbalance_strategy(
        X_sub, y_sub, strategy="class_weight", random_state=42
    )

    # Tuned hyperparameters for Variant B
    params = {
        "max_depth": 6,
        "learning_rate": 0.16552248735454136,
        "n_estimators": 190,
        "subsample": 0.7263892580562121,
        "colsample_bytree": 0.8086684313220379,
        "min_child_weight": 1,
        "scale_pos_weight": spw,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": 4,
    }

    print("Fitting Variant B XGBoost on 10:1 subsample...", flush=True)
    clf = XGBClassifier(**params)
    clf.fit(X_sub, y_sub)

    val_probs = clf.predict_proba(X_val_trans)[:, 1]
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    print(f"Variant B Validation PR-AUC: {val_pr_auc:.4f}", flush=True)

    # Threshold calibration
    thresh_info = compute_gateway_thresholds(y_val, val_probs)
    print(f"Unrounded review: {thresh_info['unrounded']['t_review']:.8f}", flush=True)
    print(f"Unrounded high:   {thresh_info['unrounded']['t_high']:.8f}", flush=True)

    # 1. Gain Importance from booster
    booster = clf.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    total_gain = sum(gain_dict.values()) if gain_dict else 1.0
    gain_importance = {}
    for idx, f_name in enumerate(feature_names):
        key = f"f{idx}"
        raw_gain = gain_dict.get(key, 0.0)
        gain_importance[f_name] = {
            "raw_gain": float(raw_gain),
            "normalized_gain": float(raw_gain / total_gain),
        }

    sorted_gain = sorted(
        gain_importance.items(), key=lambda x: x[1]["normalized_gain"], reverse=True
    )

    # 2. Permutation Importance on validation split (scoring=average_precision)
    print("Computing Permutation Importance on validation set (5 repeats)...", flush=True)
    perm = permutation_importance(
        clf,
        X_val_trans,
        y_val,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    print("Permutation Importance computation completed.", flush=True)

    perm_importance = {}
    for idx, f_name in enumerate(feature_names):
        perm_importance[f_name] = {
            "mean_pr_auc_decrease": float(perm.importances_mean[idx]),
            "std_pr_auc_decrease": float(perm.importances_std[idx]),
        }

    sorted_perm = sorted(
        perm_importance.items(),
        key=lambda x: x[1]["mean_pr_auc_decrease"],
        reverse=True,
    )

    results = {
        "variant": "Variant B (Reduced-Shortcut)",
        "features": feature_names,
        "validation_pr_auc": val_pr_auc,
        "thresholds": thresh_info,
        "gain_importance_ranked": [
            {
                "feature": k,
                "normalized_gain": round(v["normalized_gain"], 4),
                "raw_gain": round(v["raw_gain"], 2),
            }
            for k, v in sorted_gain
        ],
        "permutation_importance_ranked": [
            {
                "feature": k,
                "mean_pr_auc_decrease": round(v["mean_pr_auc_decrease"], 6),
                "std_pr_auc_decrease": round(v["std_pr_auc_decrease"], 6),
            }
            for k, v in sorted_perm
        ],
    }

    with open("ml/reports/stage2_validation_importance.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    compute_validation_importance()

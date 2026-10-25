"""PaySim data ingestion, chronological splitting, and EDA generation module.

Source of Truth: docs/ml-artifact-contract.md and docs/api-contract.md.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

COLUMN_MAPPING: dict[str, str] = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}


def map_raw_to_api_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw PaySim column names to the snake_case API schema.

    Reused across training, evaluation, and backend scoring so column
    definitions cannot drift.
    """
    rename_dict = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    return df.rename(columns=rename_dict)


def load_raw_paysim(raw_path: str | Path = "ml/data/raw/paysim.csv") -> pd.DataFrame:
    """Load raw PaySim dataset from disk and apply standard column mapping.

    Fails with FileNotFoundError if dataset is missing (real data is required for
    all training and reporting; synthetic fixtures are restricted to unit tests).
    """
    path = Path(raw_path)
    if not path.exists():
        # Fallback check for paysim.csv.csv double-extension
        alt_path = path.parent / f"{path.name}.csv"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(
                f"Raw PaySim dataset not found at '{path}'. "
                "Real PaySim data is required for training and reporting. "
                "Please place paysim.csv in ml/data/raw/."
            )

    df = pd.read_csv(path)
    return map_raw_to_api_columns(df)


def split_chronological(
    df: pd.DataFrame,
    train_step_max: int = 520,
    val_step_max: int = 595,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split PaySim data on simulation step.

    Split boundaries:
    - Train: step <= train_step_max (steps 1..520, ~70% time horizon)
    - Validation: train_step_max < step <= val_step_max (steps 521..595, ~10% tail)
    - Test: step > val_step_max (steps 596..743, ~20% future horizon)

    Asserts strict chronological ordering with zero overlap.
    """
    train_mask = df["step"] <= train_step_max
    val_mask = (df["step"] > train_step_max) & (df["step"] <= val_step_max)
    test_mask = df["step"] > val_step_max

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Strict chronological ordering assertion
    train_max = train_df["step"].max()
    val_min = val_df["step"].min()
    val_max = val_df["step"].max()
    test_min = test_df["step"].min()

    assert train_max < val_min, f"Train step max {train_max} >= Val step min {val_min}"
    assert val_max < test_min, f"Val step max {val_max} >= Test step min {test_min}"

    return train_df, val_df, test_df


def evaluate_rules_baseline_vectorized(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate SRS FR-7 rules-only baseline heuristics using backend rule semantics.

    Matches app.services.rules definitions:
    1. check_origin_account_drained:
       - type in ['TRANSFER', 'CASH_OUT']
       - old_balance_orig > 0.0 (old_balance_orig == 0 is strictly NOT drained)
       - new_balance_orig == 0.0
       - amount >= old_balance_orig * 0.95
    2. check_large_transfer_amount:
       - type in ['TRANSFER', 'CASH_OUT']
       - amount >= 200,000.0
    3. Union: drain >= 95% OR amount >= 200,000.0
    """
    y_true = df["is_fraud"].to_numpy(dtype=int)

    is_transfer_or_cashout = df["type"].isin(["TRANSFER", "CASH_OUT"]).to_numpy()
    old_orig = df["old_balance_orig"].to_numpy(dtype=float)
    new_orig = df["new_balance_orig"].to_numpy(dtype=float)
    amount = df["amount"].to_numpy(dtype=float)

    # Backend rule 1: ORIGIN_ACCOUNT_DRAINED
    # If old_balance_orig == 0, returns None (not drained). Requires new_balance_orig == 0.
    drain_pred = (
        is_transfer_or_cashout
        & (old_orig > 0.0)
        & (new_orig == 0.0)
        & (amount >= old_orig * 0.95)
    ).astype(int)

    # Backend rule 2: LARGE_TRANSFER_AMOUNT
    amount_pred = (is_transfer_or_cashout & (amount >= 200_000.0)).astype(int)

    # Union rule: Drain OR Amount per SRS FR-7
    union_pred = (drain_pred | amount_pred).astype(int)

    def calc_metrics(y_pred: np.ndarray) -> dict[str, float]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        pr_auc = float(average_precision_score(y_true, y_pred))
        fpr = float(fp / max(fp + tn, 1))
        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "pr_auc": round(pr_auc, 4),
            "fpr": round(fpr, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    return {
        "drain_ge_95pct_alone": calc_metrics(drain_pred),
        "amount_gt_200k_alone": calc_metrics(amount_pred),
        "union_drain_or_amount": calc_metrics(union_pred),
    }


def compute_balance_and_drain_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute balance error and account drain statistics for fraud vs legit rows.

    Uses backend rule definition of ORIGIN_ACCOUNT_DRAINED:
    - type in ['TRANSFER', 'CASH_OUT']
    - old_balance_orig > 0.0 (when old_balance_orig == 0, drain is False)
    - new_balance_orig == 0.0
    - amount >= old_balance_orig * 0.95
    """
    fraud_df = df[df["is_fraud"] == 1]
    legit_df = df[df["is_fraud"] == 0]

    def stats_for_subset(sub: pd.DataFrame) -> dict[str, Any]:
        amount = sub["amount"].to_numpy(dtype=float)
        old_orig = sub["old_balance_orig"].to_numpy(dtype=float)
        new_orig = sub["new_balance_orig"].to_numpy(dtype=float)
        old_dest = sub["old_balance_dest"].to_numpy(dtype=float)
        new_dest = sub["new_balance_dest"].to_numpy(dtype=float)
        is_tx_type = sub["type"].isin(["TRANSFER", "CASH_OUT"]).to_numpy()

        err_orig = old_orig - amount - new_orig
        err_dest = old_dest + amount - new_dest

        # Backend rule logic for account drain
        is_drained = (
            is_tx_type
            & (old_orig > 0.0)
            & (new_orig == 0.0)
            & (amount >= old_orig * 0.95)
        )
        drained_count = int(np.sum(is_drained))
        drained_ratio = float(drained_count / max(len(sub), 1))

        # Destination balance error statistics (means and medians)
        return {
            "count": len(sub),
            "balance_error_orig": {
                "mean": round(float(np.mean(err_orig)), 2),
                "std": round(float(np.std(err_orig)), 2),
                "median": round(float(np.median(err_orig)), 2),
                "pct_zero_error_within_1e4": round(
                    float(np.mean(np.abs(err_orig) < 1e-4)), 4
                ),
            },
            "balance_error_dest": {
                "mean": round(float(np.mean(err_dest)), 2),
                "std": round(float(np.std(err_dest)), 2),
                "median": round(float(np.median(err_dest)), 2),
            },
            "origin_drain_backend_rule": {
                "drained_count": drained_count,
                "drained_ratio": round(drained_ratio, 4),
                "definition": (
                    "type in ['TRANSFER', 'CASH_OUT'] AND old_balance_orig > 0 "
                    "AND new_balance_orig == 0 AND amount >= old_balance_orig * 0.95. "
                    "When old_balance_orig == 0, drain is strictly False."
                ),
            },
        }

    return {
        "fraud_transactions": stats_for_subset(fraud_df),
        "legitimate_transactions": stats_for_subset(legit_df),
    }


def compute_single_feature_pr_auc_split(
    df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute single-feature PR-AUC and correlation direction on a given split.

    Direction handling:
    Computes pr_pos = average_precision_score(y, +x) and pr_neg = average_precision_score(y, -x).
    If pr_pos >= pr_neg: direction is '+' (higher feature value increases fraud likelihood).
    If pr_neg > pr_pos: direction is '-' (lower feature value increases fraud likelihood).
    PR-AUC score reported is max(pr_pos, pr_neg).
    """
    y_true = df["is_fraud"].to_numpy(dtype=int)
    scores: dict[str, dict[str, Any]] = {}

    numeric_cols = [
        "amount",
        "old_balance_orig",
        "new_balance_orig",
        "old_balance_dest",
        "new_balance_dest",
    ]

    for col in numeric_cols:
        vals = df[col].to_numpy(dtype=float)
        pr_pos = float(average_precision_score(y_true, vals))
        pr_neg = float(average_precision_score(y_true, -vals))
        direction = "+" if pr_pos >= pr_neg else "-"
        best_pr = max(pr_pos, pr_neg)
        scores[col] = {
            "pr_auc": round(best_pr, 4),
            "direction": direction,
        }

    # Engineered features
    amount = df["amount"].to_numpy(dtype=float)
    old_orig = df["old_balance_orig"].to_numpy(dtype=float)
    new_orig = df["new_balance_orig"].to_numpy(dtype=float)
    old_dest = df["old_balance_dest"].to_numpy(dtype=float)
    new_dest = df["new_balance_dest"].to_numpy(dtype=float)
    step = df["step"].to_numpy(dtype=int)

    err_orig = old_orig - amount - new_orig
    err_dest = old_dest + amount - new_dest
    delta_orig = new_orig - old_orig
    delta_dest = new_dest - old_dest
    hour = (step % 24).astype(float)

    engineered = {
        "balance_error_orig": err_orig,
        "balance_error_dest": err_dest,
        "balance_delta_orig": delta_orig,
        "balance_delta_dest": delta_dest,
        "hour_of_day": hour,
    }

    for name, arr in engineered.items():
        pr_pos = float(average_precision_score(y_true, arr))
        pr_neg = float(average_precision_score(y_true, -arr))
        direction = "+" if pr_pos >= pr_neg else "-"
        best_pr = max(pr_pos, pr_neg)
        scores[name] = {
            "pr_auc": round(best_pr, 4),
            "direction": direction,
        }

    # One-hot type indicators (always '+' direction)
    for t in ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]:
        arr = (df["type"] == t).to_numpy(dtype=float)
        pr = float(average_precision_score(y_true, arr))
        scores[f"type_{t}"] = {
            "pr_auc": round(pr, 4),
            "direction": "+",
        }

    return scores


def build_split_breakdown(
    split_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build detailed per-type and per-step breakdown for a split."""
    step_min = int(split_df["step"].min()) if len(split_df) > 0 else 0
    step_max = int(split_df["step"].max()) if len(split_df) > 0 else 0

    # Per-type counts
    per_type: dict[str, dict[str, Any]] = {}
    for t in ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]:
        sub = split_df[split_df["type"] == t]
        total_t = len(sub)
        fraud_t = int(sub["is_fraud"].sum()) if total_t > 0 else 0
        per_type[t] = {
            "total_rows": total_t,
            "fraud_count": fraud_t,
            "fraud_prevalence": round(float(fraud_t / max(total_t, 1)), 6),
        }

    # Per-step volume
    per_step: dict[str, dict[str, int]] = {}
    grouped = split_df.groupby("step")["is_fraud"].agg(["count", "sum"])
    for step_num, row in grouped.iterrows():
        per_step[str(step_num)] = {
            "total_rows": int(row["count"]),
            "fraud_count": int(row["sum"]),
        }

    return {
        "step_range": [step_min, step_max],
        "total_rows": len(split_df),
        "fraud_count": int(split_df["is_fraud"].sum()),
        "fraud_prevalence": round(float(split_df["is_fraud"].mean()), 6),
        "per_type": per_type,
        "per_step": per_step,
    }


def generate_eda_report(
    raw_path: str | Path = "ml/data/raw/paysim.csv",
    output_path: str | Path = "ml/reports/eda.json",
) -> dict[str, Any]:
    """Generate comprehensive exploratory data analysis report from raw PaySim data.

    Evaluates single-feature PR-AUC on TRAIN and VALIDATION splits (never on test).
    """
    df = load_raw_paysim(raw_path)
    train_df, val_df, test_df = split_chronological(df)

    total_rows = len(df)
    total_fraud = int(df["is_fraud"].sum())
    overall_fraud_ratio = float(df["is_fraud"].mean())

    # 1. Per-split statistics with per-type and per-step breakdowns
    split_stats = {
        "train": build_split_breakdown(train_df),
        "validation": build_split_breakdown(val_df),
        "test": build_split_breakdown(test_df),
        "prevalence_drift_ratio_test_vs_train": round(
            float(test_df["is_fraud"].mean() / max(train_df["is_fraud"].mean(), 1e-9)),
            2,
        ),
    }

    # 2. Overall per-type distribution
    overall_per_type: dict[str, Any] = {}
    for t, count in df["type"].value_counts().items():
        sub = df[df["type"] == t]
        f_count = int(sub["is_fraud"].sum())
        overall_per_type[str(t)] = {
            "total_transactions": int(count),
            "fraud_count": f_count,
            "fraud_ratio": round(float(f_count / count), 6),
        }

    # 3. Merchant destination analysis
    is_merchant = df["name_dest"].astype(str).str.startswith("M")
    merchant_count = int(is_merchant.sum())
    merchant_fraud = int(df.loc[is_merchant, "is_fraud"].sum())

    # 4. Balance error and drain statistics (fraud vs legitimate)
    balance_and_drain_stats = compute_balance_and_drain_stats(df)

    # 5. Rules baseline evaluated on the test set
    rules_baseline_test = evaluate_rules_baseline_vectorized(test_df)

    # 6. Single-feature PR-AUC on TRAIN and VALIDATION (NOT test)
    single_feature_pr_auc_train = compute_single_feature_pr_auc_split(train_df)
    single_feature_pr_auc_val = compute_single_feature_pr_auc_split(val_df)

    eda_data: dict[str, Any] = {
        "data_provenance": {
            "raw_dataset_path": str(raw_path),
            "total_rows": total_rows,
            "overall_fraud_count": total_fraud,
            "overall_fraud_ratio": round(overall_fraud_ratio, 8),
            "step_min": int(df["step"].min()),
            "step_max": int(df["step"].max()),
        },
        "per_type_distribution": overall_per_type,
        "merchant_destination_analysis": {
            "merchant_transactions": merchant_count,
            "merchant_fraud_count": merchant_fraud,
            "merchant_equals_type_payment": bool(
                merchant_count
                == overall_per_type.get("PAYMENT", {}).get(
                    "total_transactions", 0
                )
            ),
        },
        "split_statistics": split_stats,
        "balance_error_and_drain_statistics": balance_and_drain_stats,
        "rules_baseline_test_evaluation": rules_baseline_test,
        "single_feature_pr_auc": {
            "train": single_feature_pr_auc_train,
            "validation": single_feature_pr_auc_val,
            "sign_direction_methodology": (
                "Evaluated strictly on TRAIN and VALIDATION (test numbers excluded). "
                "Direction sign ('+' / '-') indicates whether larger (+) or smaller (-) "
                "values correlate with fraud risk: pr_pos = PR-AUC(+x), pr_neg = PR-AUC(-x). "
                "Reported pr_auc = max(pr_pos, pr_neg)."
            ),
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(eda_data, f, indent=2)

    return eda_data


if __name__ == "__main__":
    report = generate_eda_report()
    print("Successfully generated updated EDA report at ml/reports/eda.json")

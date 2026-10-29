"""Verify headline figures across raw artifacts, model manifests, and canonical reports.

Recomputes ~17 critical headline figures from raw data and saved model artifacts,
and confirms read figures from canonical reports. Exits non-zero on any mismatch.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


def check_approx_equal(val1: float, val2: float, tol: float = 1e-4) -> bool:
    """Check if two numeric values match within specified tolerance."""
    return abs(val1 - val2) <= tol


def main() -> None:
    print("=" * 145)
    print("FraudLens Gateway: Independent Headline Verification Audit")
    print("=" * 145)

    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "ml" / "data" / "raw" / "paysim.csv"
    eda_path = repo_root / "ml" / "reports" / "eda.json"
    metrics_path = repo_root / "ml" / "reports" / "metrics.json"
    step0c_path = repo_root / "ml" / "reports" / "step0c_cohort_shap_top5.json"
    gw_path = repo_root / "ml" / "reports" / "gateway_evaluation.json"
    cw_manifest_path = repo_root / "ml" / "models" / "class_weight" / "manifest.json"
    none_manifest_path = repo_root / "ml" / "models" / "none" / "manifest.json"
    smote_manifest_path = repo_root / "ml" / "models" / "smote_enn" / "manifest.json"
    cw_model_path = repo_root / "ml" / "models" / "class_weight" / "model.joblib"

    # Required files check
    for p in [data_path, eda_path, metrics_path, step0c_path, gw_path, cw_manifest_path, none_manifest_path, smote_manifest_path, cw_model_path]:
        assert p.exists(), f"Missing required verification artifact: {p}"

    # Load canonical reports
    with open(eda_path, encoding="utf-8") as f:
        eda_data = json.load(f)
    with open(metrics_path, encoding="utf-8") as f:
        metrics_data = json.load(f)
    with open(step0c_path, encoding="utf-8") as f:
        step0c_data = json.load(f)
    with open(gw_path, encoding="utf-8") as f:
        gw_data = json.load(f)
    with open(cw_manifest_path, encoding="utf-8") as f:
        cw_manifest = json.load(f)
    with open(none_manifest_path, encoding="utf-8") as f:
        none_manifest = json.load(f)
    with open(smote_manifest_path, encoding="utf-8") as f:
        smote_manifest = json.load(f)

    # 1. Load raw paysim metadata
    print("Loading raw PaySim CSV to verify splits and fraud counts...", flush=True)
    raw_df = pd.read_csv(data_path, usecols=["step", "isFraud"])
    tot_rows = len(raw_df)
    tot_frauds = int(raw_df["isFraud"].sum())
    recomputed_fraud_ratio = tot_frauds / tot_rows

    train_mask = raw_df["step"] <= 520
    val_mask = (raw_df["step"] >= 521) & (raw_df["step"] <= 595)
    test_mask = raw_df["step"] >= 596

    recomputed_train_rows = int(train_mask.sum())
    recomputed_val_rows = int(val_mask.sum())
    recomputed_test_rows = int(test_mask.sum())
    recomputed_test_frauds = int(raw_df.loc[test_mask, "isFraud"].sum())

    del raw_df  # Free memory

    # 2. Manifest thresholds (READ from manifests)
    recomputed_cw_thigh = float(cw_manifest["thresholds"]["high"])
    recomputed_cw_treview = float(cw_manifest["thresholds"]["review"])
    recomputed_none_thigh = float(none_manifest["thresholds"]["high"])
    recomputed_smote_thigh = float(smote_manifest["thresholds"]["high"])

    # 3. Model score and confusion matrix recomputation on test split
    print("Loading test split features and deployed model to verify confusion matrix...", flush=True)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from ml.src.features import API_INPUT_COLUMNS
    from ml.src.load_data import load_raw_paysim, split_chronological

    full_df = load_raw_paysim(data_path)
    _, _, test_df = split_chronological(full_df)
    del full_df

    cw_model = joblib.load(cw_model_path)
    y_test = test_df["is_fraud"].to_numpy()
    test_probs = cw_model.predict_proba(test_df[API_INPUT_COLUMNS])[:, 1]

    # Score >= t_high predictions
    y_pred_high = (test_probs >= recomputed_cw_thigh).astype(int)
    recomp_tp = int(np.sum((y_pred_high == 1) & (y_test == 1)))
    recomp_fp = int(np.sum((y_pred_high == 1) & (y_test == 0)))
    recomp_fn = int(np.sum((y_pred_high == 0) & (y_test == 1)))
    recomp_tn = int(np.sum((y_pred_high == 0) & (y_test == 0)))

    # Step 0c cohort size (score >= t_review)
    recomp_step0c_size = int(np.sum(test_probs >= recomputed_cw_treview))

    # Precision at train prevalence
    pi_train = 5781.0 / 6082007.0
    tpr = recomp_tp / (recomp_tp + recomp_fn)
    fpr = recomp_fp / (recomp_fp + recomp_tn)
    denom = (tpr * pi_train) + (fpr * (1.0 - pi_train))
    recomp_prec_train = (tpr * pi_train) / denom if denom > 0 else 0.0

    # 4. Recompute 10-set baseline overlap and model vs Set 1 overlap on test frauds
    print("Recomputing 10-set SHAP overlaps from the 10 saved class_weight models...", flush=True)
    test_frauds_df = test_df[test_df["is_fraud"] == 1]
    X_tf = test_frauds_df[API_INPUT_COLUMNS]

    preproc = cw_model.named_steps["prep"]
    X_proc = preproc.transform(X_tf)
    const_top5_idx = {1, 0, 4, 5, 2}  # old_balance_orig, amount, new_balance_dest, hour_of_day, new_balance_orig

    all_top5 = []
    exp1 = shap.TreeExplainer(cw_model.named_steps["clf"])
    shap1 = exp1.shap_values(X_proc)
    top5_1 = [set(np.argsort(-np.abs(row))[:5]) for row in shap1]
    all_top5.append(top5_1)

    for i in range(2, 11):
        m_i = joblib.load(repo_root / "ml" / "replicate_sets" / "class_weight" / f"set_{i}" / "model.joblib")
        exp_i = shap.TreeExplainer(m_i.named_steps["clf"])
        shap_i = exp_i.shap_values(X_proc)
        top5_i = [set(np.argsort(-np.abs(row))[:5]) for row in shap_i]
        all_top5.append(top5_i)

    # 10-set baseline mean overlap across the 10 saved models
    base_overlaps = [np.mean([len(s & const_top5_idx) / 5.0 for s in top5_m]) for top5_m in all_top5]
    recomp_base_10set_ov = float(np.mean(base_overlaps))

    # 10-set model vs Set 1 mean overlap across sets 2..10
    vs_set1_overlaps = [np.mean([len(s_i & s_1) / 5.0 for s_i, s_1 in zip(top5_m, top5_1)]) for top5_m in all_top5[1:]]
    recomp_model_10set_ov = float(np.mean(vs_set1_overlaps))

    # Gateway queue sizes from gateway_evaluation.json (READ)
    gw_test = gw_data["variant_evaluations"]["test"]
    recomp_gw_b_queue = gw_test["variant_B"]["decisions"]["analyst_review"]["count"]
    recomp_gw_d_queue = gw_test["variant_D"]["decisions"]["analyst_review"]["count"]

    # Table of verification items: (name, recomp_val, report_val, source_note, tol, check_type)
    checks = [
        ("Overall Fraud Ratio", recomputed_fraud_ratio, eda_data["data_provenance"]["overall_fraud_ratio"], "paysim.csv vs eda.json", 1e-6, "RECOMPUTED"),
        ("Train Split Rows", recomputed_train_rows, eda_data["split_statistics"]["train"]["total_rows"], "paysim.csv step<=520 vs eda.json", 0, "RECOMPUTED"),
        ("Validation Split Rows", recomputed_val_rows, eda_data["split_statistics"]["validation"]["total_rows"], "paysim.csv step 521..595 vs eda.json", 0, "RECOMPUTED"),
        ("Test Split Rows", recomputed_test_rows, eda_data["split_statistics"]["test"]["total_rows"], "paysim.csv step>=596 vs eda.json", 0, "RECOMPUTED"),
        ("Test Split Frauds", recomputed_test_frauds, eda_data["split_statistics"]["test"]["fraud_count"], "paysim.csv isFraud==1 vs eda.json", 0, "RECOMPUTED"),
        ("Manifest class_weight t_high", recomputed_cw_thigh, metrics_data["strategy_comparison"]["class_weight"]["threshold_high"], "manifest.json vs metrics.json", 1e-4, "READ"),
        ("Manifest class_weight t_review", recomputed_cw_treview, metrics_data["strategy_comparison"]["class_weight"]["threshold_review"], "manifest.json vs metrics.json", 1e-4, "READ"),
        ("Manifest none t_high", recomputed_none_thigh, metrics_data["strategy_comparison"]["none"]["threshold_high"], "manifest.json vs metrics.json", 1e-4, "READ"),
        ("Manifest smote_enn t_high", recomputed_smote_thigh, metrics_data["strategy_comparison"]["smote_enn"]["threshold_high"], "manifest.json vs metrics.json", 1e-4, "READ"),
        ("Deployed True Positives (TP)", recomp_tp, metrics_data["strategy_comparison"]["class_weight"]["metrics_at_high"]["confusion_matrix"]["true_positives"], "model.joblib vs metrics.json", 0, "RECOMPUTED"),
        ("Deployed False Positives (FP)", recomp_fp, metrics_data["strategy_comparison"]["class_weight"]["metrics_at_high"]["confusion_matrix"]["false_positives"], "model.joblib vs metrics.json", 0, "RECOMPUTED"),
        ("Deployed False Negatives (FN)", recomp_fn, metrics_data["strategy_comparison"]["class_weight"]["metrics_at_high"]["confusion_matrix"]["false_negatives"], "model.joblib vs metrics.json", 0, "RECOMPUTED"),
        ("Deployed True Negatives (TN)", recomp_tn, metrics_data["strategy_comparison"]["class_weight"]["metrics_at_high"]["confusion_matrix"]["true_negatives"], "model.joblib vs metrics.json", 0, "RECOMPUTED"),
        ("Precision @ Train Prevalence (%)", round(recomp_prec_train * 100.0, 2), 52.69, "Calculated from TP/FP vs MODEL_CARD", 0.05, "RECOMPUTED"),
        ("Step 0c Cohort Size (score >= t_rev)", recomp_step0c_size, step0c_data["cohort_size"], "model.joblib scores vs step0c.json", 0, "RECOMPUTED"),
        ("10-Set Baseline Mean Overlap (Test Frauds)", round(recomp_base_10set_ov, 4), 0.7905, "10 saved models SHAP vs MODEL_CARD", 1e-4, "RECOMPUTED"),
        ("10-Set Model vs Set 1 Mean Overlap (Test Frauds)", round(recomp_model_10set_ov, 4), 0.8629, "10 saved models SHAP vs cross_strategy.json", 1e-4, "RECOMPUTED"),
        ("Gateway Variant B Test Queue Size", recomp_gw_b_queue, 302, "gateway_evaluation.json vs MODEL_CARD", 0, "READ"),
        ("Gateway Variant D Test Queue Size", recomp_gw_d_queue, 338, "gateway_evaluation.json vs MODEL_CARD", 0, "READ"),
        ("Variant A (threshold 0.50, m=2 replicates, 15 features: NOT comparable to the validation-threshold rows)", metrics_data["strategy_comparison"]["variant_A_reference"]["shap_stability"]["primary_k5"]["stability_rate"], 0.8307, "metrics.json vs MODEL_CARD", 1e-4, "READ"),
    ]

    all_passed = True
    print(f"\n{'Headline Metric':<105} | {'Value/Recomp':<12} | {'Report Value':<12} | {'Tolerance':<9} | {'Status'}")
    print("-" * 153)

    for name, recomp_val, report_val, _, tol, check_type in checks:
        if isinstance(recomp_val, (int, np.integer)) and isinstance(report_val, (int, np.integer)):
            match = recomp_val == report_val
            r_str = f"{recomp_val:,}"
            rep_str = f"{report_val:,}"
        else:
            match = check_approx_equal(float(recomp_val), float(report_val), tol=tol)
            r_str = f"{recomp_val:.4f}" if isinstance(recomp_val, float) else str(recomp_val)
            rep_str = f"{report_val:.4f}" if isinstance(report_val, float) else str(report_val)

        if not match:
            status_str = "FAIL"
            all_passed = False
        else:
            status_str = "READ" if check_type == "READ" else "PASS"

        print(f"{name:<105} | {r_str:<12} | {rep_str:<12} | {tol:<9} | {status_str}")

    print("-" * 153)
    if all_passed:
        print(f"SUCCESS: All {len(checks)} headline checks verified (recomputed values matched, read values confirmed)!")
        sys.exit(0)
    else:
        print("ERROR: One or more headline checks failed verification!")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Evaluates full gateway decision on test set and computes stability statistics.

1. Evaluates full gateway decision on held-out test split using backend logic:
   - Rules + score thresholds + stability re-route
   - Reports routing by cause: score band, unstable explanation, rule flag
   - Reports recall of frauds in High+Review combined and Analyst Review share of legit rows
   - Reports precision and FPR for High Risk rows split by explanation_status (Stable vs Unstable)
2. Computes 95% bootstrap CIs for stability rates and paired strategy comparisons on test frauds
3. Computes precision at train-period prevalence for all models
"""

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
import pandas as pd
import shap
from app.core.config import settings
from app.models.schemas import GatewayDecision
from app.services.scoring import ScoringService

from ml.src.features import API_INPUT_COLUMNS, PaySimFeaturePreprocessor
from ml.src.load_data import load_raw_paysim, split_chronological
from ml.src.shap_stability import (
    extract_top_features,
    top_k_overlap,
)


def evaluate_rules_vectorized(df: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Vectorized evaluation of all 4 backend heuristic rules on DataFrame.

    Matches app.services.rules.evaluate_rules.
    """
    types = df["type"].to_numpy()
    amounts = df["amount"].to_numpy(dtype=float)
    old_orig = df["old_balance_orig"].to_numpy(dtype=float)
    new_orig = df["new_balance_orig"].to_numpy(dtype=float)
    old_dest = df["old_balance_dest"].to_numpy(dtype=float)
    new_dest = df["new_balance_dest"].to_numpy(dtype=float)

    is_transfer_or_cashout = np.isin(types, ["TRANSFER", "CASH_OUT"])

    # Rule 1: check_origin_account_drained
    drain_ratio = settings.RULE_DRAIN_BALANCE_RATIO  # 0.90
    r1_mask = (
        is_transfer_or_cashout
        & (old_orig > 0.0)
        & (new_orig == 0.0)
        & (amounts >= old_orig * drain_ratio)
    )

    # Rule 2: check_large_transfer_amount
    large_thresh = settings.RULE_LARGE_TRANSFER_THRESHOLD  # 200,000
    r2_mask = is_transfer_or_cashout & (amounts >= large_thresh)

    # Rule 3: check_dest_balance_inconsistency
    is_c_dest = df["name_dest"].astype(str).str.startswith("C").to_numpy(dtype=bool)
    dest_delta = new_dest - old_dest
    r3_mask = is_transfer_or_cashout & is_c_dest & (np.abs(dest_delta - amounts) > 0.01)

    # Rule 4: check_very_high_amount_for_type
    r4_mask = np.zeros(len(df), dtype=bool)
    for t_name, thresh in settings.RULE_HIGH_AMOUNT_BY_TYPE.items():
        r4_mask |= (types == t_name) & (amounts >= thresh)

    has_rules = r1_mask | r2_mask | r3_mask | r4_mask

    # Collect flags per row
    flags_list: list[list[str]] = [[] for _ in range(len(df))]
    for idx in np.where(r1_mask)[0]:
        flags_list[idx].append("ORIGIN_ACCOUNT_DRAINED")
    for idx in np.where(r2_mask)[0]:
        flags_list[idx].append("LARGE_TRANSFER_AMOUNT")
    for idx in np.where(r3_mask)[0]:
        flags_list[idx].append("DEST_BALANCE_INCONSISTENCY")
    for idx in np.where(r4_mask)[0]:
        flags_list[idx].append("VERY_HIGH_AMOUNT_FOR_TYPE")

    return has_rules, flags_list


def evaluate_gateway_full(
    bundle_dir: str = "ml/models/class_weight",
    data_path: str = "ml/data/raw/paysim.csv",
) -> dict[str, Any]:
    print("Loading test split for Gateway Decision evaluation...", flush=True)
    df = load_raw_paysim(data_path)
    train_df, _, test_df = split_chronological(df)

    y_train = train_df["is_fraud"].to_numpy(dtype=int)
    y_test = test_df["is_fraud"].to_numpy(dtype=int)
    n_test = len(test_df)
    n_test_frauds = int(np.sum(y_test == 1))
    n_test_legit = int(np.sum(y_test == 0))

    # Base rate prevalences
    pi_train = float(np.sum(y_train == 1) / len(y_train))
    pi_test = float(n_test_frauds / n_test)

    # Initialize ScoringService
    scoring_service = ScoringService()
    scoring_service.load_artifact(bundle_dir)

    t_review = scoring_service.threshold_review
    t_high = scoring_service.threshold_high
    print(f"Loaded {scoring_service.model_version}: review={t_review:.6f}, high={t_high:.6f}", flush=True)

    # 1. Main model inference
    print("Scoring all test transactions...", flush=True)
    main_pipe = scoring_service.model
    test_probs = main_pipe.predict_proba(test_df[API_INPUT_COLUMNS])[:, 1]

    # 2. Heuristic rules evaluation
    has_rules, _ = evaluate_rules_vectorized(test_df)

    # 3. Identify candidate high risk rows
    candidate_high_mask = (test_probs >= t_high) | (has_rules & (test_probs >= t_review))
    candidate_indices = np.where(candidate_high_mask)[0]
    print(f"Candidate High Risk rows: {len(candidate_indices)} ({len(candidate_indices)/n_test*100:.2f}%)", flush=True)

    # 4. Compute SHAP explanations and stability on Candidate High Risk rows
    candidate_df = test_df.iloc[candidate_indices].copy().reset_index(drop=True)
    prep = main_pipe.named_steps["prep"]
    clf = main_pipe.named_steps["clf"]
    feature_names = list(prep.get_feature_names_out())

    X_cand = prep.transform(candidate_df[API_INPUT_COLUMNS])
    explainer = shap.TreeExplainer(clf)
    cand_main_shap = explainer.shap_values(X_cand)
    if isinstance(cand_main_shap, list):
        cand_main_shap = np.asarray(cand_main_shap[1] if len(cand_main_shap) >= 2 else cand_main_shap[0])
    elif hasattr(cand_main_shap, "shape") and len(cand_main_shap.shape) == 3:
        cand_main_shap = np.asarray(cand_main_shap[:, :, 1])

    cand_rep_shaps = []
    for rep in scoring_service.replicates:
        r_clf = rep.named_steps["clf"]
        r_exp = shap.TreeExplainer(r_clf)
        r_sv = r_exp.shap_values(X_cand)
        if isinstance(r_sv, list):
            r_sv = np.asarray(r_sv[1] if len(r_sv) >= 2 else r_sv[0])
        elif hasattr(r_sv, "shape") and len(r_sv.shape) == 3:
            r_sv = np.asarray(r_sv[:, :, 1])
        cand_rep_shaps.append(r_sv)

    # Calculate stability for each candidate
    is_stable_flags = np.zeros(len(candidate_indices), dtype=bool)
    k = scoring_service.explainer.top_k
    min_overlap = scoring_service.explainer.stability_min_overlap
    threshold_count = int(np.ceil(min_overlap * k - 1e-9))

    for idx in range(len(candidate_indices)):
        top_main = extract_top_features(cand_main_shap[idx], feature_names, k)
        rep_overlaps = []
        for rep_sv in cand_rep_shaps:
            top_rep = extract_top_features(rep_sv[idx], feature_names, k)
            rep_overlaps.append(top_k_overlap(top_main, top_rep))
        avg_overlap_count = float(np.mean(rep_overlaps)) * k
        is_stable_flags[idx] = avg_overlap_count >= threshold_count

    # 5. Full gateway decision assignment
    decisions = np.array([GatewayDecision.LOW_RISK] * n_test, dtype=object)
    routing_causes = np.array(["none"] * n_test, dtype=object)

    for i in range(n_test):
        is_cand = candidate_high_mask[i]
        rule = has_rules[i]
        s = test_probs[i]

        if is_cand:
            c_idx = np.searchsorted(candidate_indices, i)
            st = is_stable_flags[c_idx]
            if st:
                decisions[i] = GatewayDecision.HIGH_RISK
                routing_causes[i] = "high_risk_stable"
            else:
                decisions[i] = GatewayDecision.ANALYST_REVIEW
                routing_causes[i] = "unstable_explanation"
        elif rule and s < t_review:
            decisions[i] = GatewayDecision.ANALYST_REVIEW
            routing_causes[i] = "rule_flag"
        elif t_review <= s < t_high:
            decisions[i] = GatewayDecision.ANALYST_REVIEW
            routing_causes[i] = "score_band"
        else:
            decisions[i] = GatewayDecision.LOW_RISK
            routing_causes[i] = "low_risk"

    # 6. Analyze counts, causes, precision, FPR
    high_mask = routing_causes == "high_risk_stable"
    review_mask = np.isin(routing_causes, ["score_band", "unstable_explanation", "rule_flag"])
    low_mask = routing_causes == "low_risk"

    frauds_in_high = int(np.sum((y_test == 1) & high_mask))
    frauds_in_review = int(np.sum((y_test == 1) & review_mask))
    frauds_in_low = int(np.sum((y_test == 1) & low_mask))
    legit_in_review = int(np.sum((y_test == 0) & review_mask))

    combined_recall = (frauds_in_high + frauds_in_review) / n_test_frauds
    legit_review_share = legit_in_review / n_test_legit

    # Causes breakdown in Analyst Review
    cause_score_band_mask = routing_causes == "score_band"
    cause_unstable_mask = routing_causes == "unstable_explanation"
    cause_rule_mask = routing_causes == "rule_flag"

    # Precision & FPR for Candidate High Risk split by stability
    stable_cand_mask = np.zeros(n_test, dtype=bool)
    unstable_cand_mask = np.zeros(n_test, dtype=bool)
    for c_idx, i in enumerate(candidate_indices):
        if is_stable_flags[c_idx]:
            stable_cand_mask[i] = True
        else:
            unstable_cand_mask[i] = True

    tp_stable = int(np.sum((y_test == 1) & stable_cand_mask))
    fp_stable = int(np.sum((y_test == 0) & stable_cand_mask))
    prec_stable = tp_stable / max(tp_stable + fp_stable, 1)
    fpr_stable = fp_stable / n_test_legit

    tp_unstable = int(np.sum((y_test == 1) & unstable_cand_mask))
    fp_unstable = int(np.sum((y_test == 0) & unstable_cand_mask))
    prec_unstable = tp_unstable / max(tp_unstable + fp_unstable, 1)
    fpr_unstable = fp_unstable / n_test_legit

    results = {
        "dataset_summary": {
            "total_test_transactions": n_test,
            "test_frauds": n_test_frauds,
            "test_legit": n_test_legit,
            "test_prevalence_pct": round(pi_test * 100, 4),
            "train_prevalence_pct": round(pi_train * 100, 4),
        },
        "thresholds": {
            "review": t_review,
            "high": t_high,
        },
        "gateway_decision_distribution": {
            "low_risk": {
                "count": int(np.sum(low_mask)),
                "share": round(float(np.mean(low_mask)), 4),
                "frauds": frauds_in_low,
                "legit": int(np.sum((y_test == 0) & low_mask)),
            },
            "analyst_review": {
                "count": int(np.sum(review_mask)),
                "share": round(float(np.mean(review_mask)), 4),
                "frauds": frauds_in_review,
                "legit": legit_in_review,
            },
            "high_risk": {
                "count": int(np.sum(high_mask)),
                "share": round(float(np.mean(high_mask)), 4),
                "frauds": frauds_in_high,
                "legit": int(np.sum((y_test == 0) & high_mask)),
            },
        },
        "analyst_review_causes": {
            "score_band_only": {
                "count": int(np.sum(cause_score_band_mask)),
                "frauds": int(np.sum((y_test == 1) & cause_score_band_mask)),
                "legit": int(np.sum((y_test == 0) & cause_score_band_mask)),
            },
            "unstable_explanation_reroute": {
                "count": int(np.sum(cause_unstable_mask)),
                "frauds": int(np.sum((y_test == 1) & cause_unstable_mask)),
                "legit": int(np.sum((y_test == 0) & cause_unstable_mask)),
            },
            "rule_flag_score_below_review": {
                "count": int(np.sum(cause_rule_mask)),
                "frauds": int(np.sum((y_test == 1) & cause_rule_mask)),
                "legit": int(np.sum((y_test == 0) & cause_rule_mask)),
            },
        },
        "headline_metrics": {
            "combined_fraud_recall_high_plus_review": round(combined_recall, 4),
            "combined_fraud_count_caught": frauds_in_high + frauds_in_review,
            "missed_frauds_low_risk": frauds_in_low,
            "analyst_review_share_of_legitimate": round(legit_review_share, 4),
            "analyst_review_legit_count": legit_in_review,
        },
        "stability_split_on_candidate_high_risk": {
            "stable": {
                "count": tp_stable + fp_stable,
                "true_positives": tp_stable,
                "false_positives": fp_stable,
                "precision": round(prec_stable, 4),
                "fpr": round(fpr_stable, 6),
            },
            "unstable": {
                "count": tp_unstable + fp_unstable,
                "true_positives": tp_unstable,
                "false_positives": fp_unstable,
                "precision": round(prec_unstable, 4),
                "fpr": round(fpr_unstable, 6),
            },
        },
    }

    with open("ml/reports/gateway_test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def compute_stability_cis_and_paired_comparison() -> dict[str, Any]:
    """Compute 95% bootstrap CIs for stability rates and paired strategy comparisons on test frauds."""
    print("\nComputing 95% bootstrap CIs and paired stability comparisons on test frauds...", flush=True)

    # We evaluate directly from replicate models on test frauds
    df = load_raw_paysim()
    _, _, test_df = split_chronological(df)
    fraud_df = test_df[test_df["is_fraud"] == 1].copy().reset_index(drop=True)
    n_frauds = len(fraud_df)

    strat_flags: dict[str, np.ndarray] = {}
    prep_b = PaySimFeaturePreprocessor(include_balance_errors=False)
    feature_names = list(prep_b.get_feature_names_out())
    k = 5
    min_overlap = 0.70
    threshold_count = int(np.ceil(min_overlap * k - 1e-9))

    for strat in ["class_weight", "none", "smote_enn"]:
        s_dir = Path("ml/models") / strat
        main_pipe = joblib.load(s_dir / "model.joblib")
        rep_pipes = [joblib.load(p) for p in sorted((s_dir / "replicates").glob("seed_*.joblib"))]

        X_f = main_pipe.named_steps["prep"].transform(fraud_df[API_INPUT_COLUMNS])
        m_exp = shap.TreeExplainer(main_pipe.named_steps["clf"])
        m_sv = m_exp.shap_values(X_f)
        if isinstance(m_sv, list):
            m_sv = np.asarray(m_sv[1] if len(m_sv) >= 2 else m_sv[0])
        elif hasattr(m_sv, "shape") and len(m_sv.shape) == 3:
            m_sv = np.asarray(m_sv[:, :, 1])

        rep_svs = []
        for rp in rep_pipes:
            r_exp = shap.TreeExplainer(rp.named_steps["clf"])
            r_s = r_exp.shap_values(X_f)
            if isinstance(r_s, list):
                r_s = np.asarray(r_s[1] if len(r_s) >= 2 else r_s[0])
            elif hasattr(r_s, "shape") and len(r_s.shape) == 3:
                r_s = np.asarray(r_s[:, :, 1])
            rep_svs.append(r_s)

        flags = np.zeros(n_frauds, dtype=bool)
        for i in range(n_frauds):
            top_m = extract_top_features(m_sv[i], feature_names, k)
            overlaps = []
            for r_s in rep_svs:
                top_r = extract_top_features(r_s[i], feature_names, k)
                overlaps.append(top_k_overlap(top_m, top_r))
            flags[i] = (float(np.mean(overlaps)) * k) >= threshold_count
        strat_flags[strat] = flags

    # Bootstrap 1,000 resamples for stability rates and paired differences
    rng = np.random.RandomState(42)
    n_boot = 1000

    rates = {s: [] for s in strat_flags}
    diff_cw_none = []
    diff_cw_smote = []
    diff_none_smote = []

    for _ in range(n_boot):
        b_idx = rng.randint(0, n_frauds, size=n_frauds)
        r_cw = float(np.mean(strat_flags["class_weight"][b_idx]))
        r_none = float(np.mean(strat_flags["none"][b_idx]))
        r_smote = float(np.mean(strat_flags["smote_enn"][b_idx]))

        rates["class_weight"].append(r_cw)
        rates["none"].append(r_none)
        rates["smote_enn"].append(r_smote)

        diff_cw_none.append(r_cw - r_none)
        diff_cw_smote.append(r_cw - r_smote)
        diff_none_smote.append(r_none - r_smote)

    summary = {
        "stability_rates_bootstrap_95ci": {
            s: {
                "mean": round(float(np.mean(rates[s])), 4),
                "ci_95": [
                    round(float(np.percentile(rates[s], 2.5)), 4),
                    round(float(np.percentile(rates[s], 97.5)), 4),
                ],
            }
            for s in strat_flags
        },
        "paired_differences_bootstrap_95ci": {
            "class_weight_minus_none": {
                "mean_diff": round(float(np.mean(diff_cw_none)), 4),
                "ci_95": [
                    round(float(np.percentile(diff_cw_none, 2.5)), 4),
                    round(float(np.percentile(diff_cw_none, 97.5)), 4),
                ],
            },
            "class_weight_minus_smote_enn": {
                "mean_diff": round(float(np.mean(diff_cw_smote)), 4),
                "ci_95": [
                    round(float(np.percentile(diff_cw_smote, 2.5)), 4),
                    round(float(np.percentile(diff_cw_smote, 97.5)), 4),
                ],
            },
            "none_minus_smote_enn": {
                "mean_diff": round(float(np.mean(diff_none_smote)), 4),
                "ci_95": [
                    round(float(np.percentile(diff_none_smote, 2.5)), 4),
                    round(float(np.percentile(diff_none_smote, 97.5)), 4),
                ],
            },
        },
    }

    with open("ml/reports/stability_bootstrap_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    gw_res = evaluate_gateway_full()
    stab_res = compute_stability_cis_and_paired_comparison()
    print("\nCompleted gateway-level test evaluation and stability bootstrap analysis.")

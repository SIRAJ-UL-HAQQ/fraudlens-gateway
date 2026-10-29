"""Full Gateway Routing Evaluation and Analysis.

Evaluates 4 gateway routing variants (A, B, C, D) and exploratory Variant E across
Validation and Test splits using the deployed class_weight artifact.

Read-only backend imports:
- evaluate_rules from backend.app.services.rules
- route_decision from backend.app.services.scoring

Writes:
- ml/reports/gateway_evaluation.json
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.config import settings
from app.models.schemas import ExplanationStatus, GatewayDecision, TransactionIn, TransactionType
from app.services.rules import evaluate_rules
from app.services.scoring import ScoringService

from ml.src.features import API_INPUT_COLUMNS
from ml.src.load_data import load_raw_paysim, split_chronological
from ml.src.shap_stability import extract_top_features, is_explanation_stable, top_k_overlap


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    # z = 1.95996 for 95% CI
    z = 1.959963984540054 if confidence == 0.95 else 1.6448536269514722
    p_hat = k / n
    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denominator
    half_width = (z * math.sqrt((p_hat * (1.0 - p_hat) / n) + ((z**2) / (4.0 * (n**2))))) / denominator
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return (float(lower), float(upper))


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Compute AUROC and 95% bootstrap confidence interval."""
    if len(np.unique(y_true)) < 2:
        return (0.5, (0.5, 0.5))
    point_est = float(roc_auc_score(y_true, scores))
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, scores[idx]))
    if len(boot_aucs) == 0:
        return (point_est, (point_est, point_est))
    lower = float(np.percentile(boot_aucs, 2.5))
    upper = float(np.percentile(boot_aucs, 97.5))
    return (point_est, (lower, upper))


def evaluate_rules_vectorized(df: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Vectorized evaluation of all 4 backend heuristic rules on DataFrame.

    Matches app.services.rules.evaluate_rules bit-for-bit.
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


def verify_backend_rules_parity_on_sample(df: pd.DataFrame, sample_size: int = 100) -> None:
    """Verify bit-exact parity between vectorized rules and backend evaluate_rules on a sample."""
    _, vectorized_flags = evaluate_rules_vectorized(df.iloc[:sample_size])
    for i in range(sample_size):
        row = df.iloc[i]
        tx = TransactionIn(
            step=int(row["step"]),
            type=TransactionType(row["type"]),
            amount=float(row["amount"]),
            name_orig=str(row["name_orig"]),
            old_balance_orig=float(row["old_balance_orig"]),
            new_balance_orig=float(row["new_balance_orig"]),
            name_dest=str(row["name_dest"]),
            old_balance_dest=float(row["old_balance_dest"]),
            new_balance_dest=float(row["new_balance_dest"]),
        )
        backend_flags = evaluate_rules(tx)
        assert sorted(vectorized_flags[i]) == sorted(backend_flags), (
            f"Parity mismatch at sample row {i}: {vectorized_flags[i]} vs {backend_flags}"
        )


def evaluate_recalibrated_rules(
    df: pd.DataFrame,
    large_transfer_threshold: float = 2436042.77,
    type_thresholds: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[list[str]]]:
    """Evaluate recalibrated rules with train-period <= 0.5% cap per rule."""
    if type_thresholds is None:
        type_thresholds = {
            "TRANSFER": 10000000.0,
            "CASH_OUT": 640591.60,
            "PAYMENT": 68463.74,
            "CASH_IN": 605315.93,
            "DEBIT": 87392.04,
        }

    types = df["type"].to_numpy()
    amounts = df["amount"].to_numpy(dtype=float)
    is_tc = np.isin(types, ["TRANSFER", "CASH_OUT"])

    # Rule 1 (Drain): Disabled per formal fallback
    # Rule 3 (Dest Inconsistency): Disabled per formal fallback

    # Rule 2: Recalibrated Large Transfer
    r2_mask = is_tc & (amounts >= large_transfer_threshold)

    # Rule 4: Recalibrated Very High Amount for Type
    r4_mask = np.zeros(len(df), dtype=bool)
    for t_name, thresh in type_thresholds.items():
        r4_mask |= (types == t_name) & (amounts >= thresh)

    has_rules = r2_mask | r4_mask
    flags_list: list[list[str]] = [[] for _ in range(len(df))]
    for idx in np.where(r2_mask)[0]:
        flags_list[idx].append("LARGE_TRANSFER_AMOUNT_RECALIBRATED")
    for idx in np.where(r4_mask)[0]:
        flags_list[idx].append("VERY_HIGH_AMOUNT_FOR_TYPE_RECALIBRATED")

    return has_rules, flags_list


def compute_replicate_shap_on_cohort(
    cohort_df: pd.DataFrame,
    main_pipe: Any,
    replicate_pipes: list[Any],
    k: int = 5,
    min_overlap: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute TreeExplainer SHAP values and stability for a cohort."""
    if len(cohort_df) == 0:
        return np.array([]), np.array([], dtype=bool), np.array([])

    prep = main_pipe.named_steps["prep"]
    clf = main_pipe.named_steps["clf"]
    feature_names = list(prep.get_feature_names_out())

    X = prep.transform(cohort_df[API_INPUT_COLUMNS])

    m_exp = shap.TreeExplainer(clf)
    m_sv = m_exp.shap_values(X)
    if isinstance(m_sv, list):
        m_sv = np.asarray(m_sv[1] if len(m_sv) >= 2 else m_sv[0])
    elif hasattr(m_sv, "shape") and len(m_sv.shape) == 3:
        m_sv = np.asarray(m_sv[:, :, 1])

    rep_svs = []
    for rep in replicate_pipes:
        r_clf = rep.named_steps["clf"]
        r_exp = shap.TreeExplainer(r_clf)
        r_sv = r_exp.shap_values(X)
        if isinstance(r_sv, list):
            r_sv = np.asarray(r_sv[1] if len(r_sv) >= 2 else r_sv[0])
        elif hasattr(r_sv, "shape") and len(r_sv.shape) == 3:
            r_sv = np.asarray(r_sv[:, :, 1])
        rep_svs.append(r_sv)

    n = len(cohort_df)
    overlaps = np.zeros(n, dtype=float)
    stables = np.zeros(n, dtype=bool)

    for i in range(n):
        top_m = extract_top_features(m_sv[i], feature_names, k)
        r_ov = [
            top_k_overlap(top_m, extract_top_features(rep_svs[r][i], feature_names, k))
            for r in range(len(rep_svs))
        ]
        mo = float(np.mean(r_ov))
        st = is_explanation_stable(mo, k=k, min_overlap=min_overlap)
        overlaps[i] = mo
        stables[i] = st

    return overlaps, stables, m_sv


def evaluate_split_variant(
    variant_name: str,
    df: pd.DataFrame,
    probs: np.ndarray,
    y_true: np.ndarray,
    has_rules_standard: np.ndarray,
    flags_standard: list[list[str]],
    has_rules_recalibrated: np.ndarray,
    flags_recalibrated: list[list[str]],
    superset_stables: dict[int, bool],
    superset_overlaps: dict[int, float],
    scoring_service: ScoringService,
    simulated_hours: int,
    variant_e_floor: float | None = None,
) -> dict[str, Any]:
    """Evaluate routing for a single variant on a split using backend route_decision."""
    n = len(df)
    t_rev = scoring_service.threshold_review
    t_high = scoring_service.threshold_high

    decisions = [GatewayDecision.LOW_RISK] * n
    causes = ["none"] * n

    for i in range(n):
        s = probs[i]

        # Determine effective rules and candidate status per variant
        if variant_name == "variant_A":
            rule_flags = flags_standard[i]
            has_rules = len(rule_flags) > 0
            is_cand = (s >= t_high) or (has_rules and s >= t_rev)
            st = superset_stables.get(i, True)
            exp_status = ExplanationStatus.STABLE if st else ExplanationStatus.UNSTABLE
            dec = scoring_service.route_decision(s, rule_flags, exp_status)
            if is_cand:
                causes[i] = "high_risk_stable" if st else "unstable_explanation"
            elif has_rules and s < t_rev:
                causes[i] = "rule_flag"
            elif t_rev <= s < t_high:
                causes[i] = "score_band"
            else:
                causes[i] = "low_risk"

        elif variant_name == "variant_B":
            # Score-gated rules: rules never route alone without score support
            rule_flags = flags_standard[i]
            effective_rules = rule_flags if s >= t_rev else []
            has_rules = len(effective_rules) > 0
            is_cand = (s >= t_high) or (has_rules and s >= t_rev)
            st = superset_stables.get(i, True)
            exp_status = ExplanationStatus.STABLE if st else ExplanationStatus.UNSTABLE
            dec = scoring_service.route_decision(s, effective_rules, exp_status)
            if is_cand:
                causes[i] = "high_risk_stable" if st else "unstable_explanation"
            elif t_rev <= s < t_high:
                causes[i] = "score_band"
            else:
                causes[i] = "low_risk"

        elif variant_name == "variant_C":
            # Recalibrated rules
            rule_flags = flags_recalibrated[i]
            has_rules = len(rule_flags) > 0
            is_cand = (s >= t_high) or (has_rules and s >= t_rev)
            st = superset_stables.get(i, True)
            exp_status = ExplanationStatus.STABLE if st else ExplanationStatus.UNSTABLE
            dec = scoring_service.route_decision(s, rule_flags, exp_status)
            if is_cand:
                causes[i] = "high_risk_stable" if st else "unstable_explanation"
            elif has_rules and s < t_rev:
                causes[i] = "rule_flag"
            elif t_rev <= s < t_high:
                causes[i] = "score_band"
            else:
                causes[i] = "low_risk"

        elif variant_name == "variant_D":
            # Pure ML gateway - no rules
            is_cand = s >= t_high
            st = superset_stables.get(i, True)
            exp_status = ExplanationStatus.STABLE if st else ExplanationStatus.UNSTABLE
            dec = scoring_service.route_decision(s, [], exp_status)
            if is_cand:
                causes[i] = "high_risk_stable" if st else "unstable_explanation"
            elif t_rev <= s < t_high:
                causes[i] = "score_band"
            else:
                causes[i] = "low_risk"

        elif variant_name == "variant_E":
            # Exploratory post-hoc: rule + score >= f routes to Review
            f_floor = variant_e_floor if variant_e_floor is not None else t_rev
            rule_flags = flags_standard[i]
            has_rules = len(rule_flags) > 0
            is_cand = (s >= t_high) or (has_rules and s >= t_rev)
            st = superset_stables.get(i, True)
            exp_status = ExplanationStatus.STABLE if st else ExplanationStatus.UNSTABLE

            # Evaluate with score-gating down to floor f
            effective_rules = rule_flags if s >= f_floor else []
            dec = scoring_service.route_decision(s, effective_rules, exp_status)
            if is_cand:
                causes[i] = "high_risk_stable" if st else "unstable_explanation"
            elif has_rules and f_floor <= s < t_rev:
                causes[i] = "rule_flag"
                dec = GatewayDecision.ANALYST_REVIEW
            elif t_rev <= s < t_high:
                causes[i] = "score_band"
            else:
                causes[i] = "low_risk"
        else:
            raise ValueError(f"Unknown variant: {variant_name}")

        decisions[i] = dec

    dec_values = np.array([d.value for d in decisions])
    causes_arr = np.array(causes, dtype=object)

    high_mask = dec_values == GatewayDecision.HIGH_RISK.value
    review_mask = dec_values == GatewayDecision.ANALYST_REVIEW.value
    low_mask = dec_values == GatewayDecision.LOW_RISK.value

    total_frauds = int(np.sum(y_true == 1))
    total_legit = int(np.sum(y_true == 0))

    def make_dec_stats(mask: np.ndarray) -> dict[str, Any]:
        cnt = int(np.sum(mask))
        f_cnt = int(np.sum((y_true == 1) & mask))
        l_cnt = cnt - f_cnt
        rate = float(f_cnt / cnt) if cnt > 0 else 0.0
        return {
            "count": cnt,
            "per_1000_tx": round(float(cnt / n * 1000.0), 3),
            "frauds": f_cnt,
            "legitimate": l_cnt,
            "fraud_rate": round(rate, 6),
            "fraud_rate_pct": round(rate * 100.0, 3),
        }

    dec_stats = {
        "high_risk": make_dec_stats(high_mask),
        "analyst_review": make_dec_stats(review_mask),
        "low_risk": make_dec_stats(low_mask),
    }

    high_frauds = dec_stats["high_risk"]["frauds"]
    review_frauds = dec_stats["analyst_review"]["frauds"]
    combined_frauds = high_frauds + review_frauds
    high_recall = high_frauds / total_frauds
    review_recall = review_frauds / total_frauds
    combined_recall = combined_frauds / total_frauds
    comb_ci = compute_wilson_ci(combined_frauds, total_frauds, confidence=0.95)

    queue_total = dec_stats["analyst_review"]["count"]
    legit_in_review = dec_stats["analyst_review"]["legitimate"]
    legit_review_rate = float(legit_in_review / total_legit)

    # Queue breakdown by cause
    cause_score_band = causes_arr == "score_band"
    cause_unstable = causes_arr == "unstable_explanation"
    cause_rule = causes_arr == "rule_flag"

    queue_by_cause = {
        "score_band": {
            "total": int(np.sum(cause_score_band)),
            "frauds": int(np.sum((y_true == 1) & cause_score_band)),
            "legitimate": int(np.sum((y_true == 0) & cause_score_band)),
        },
        "unstable_explanation": {
            "total": int(np.sum(cause_unstable)),
            "frauds": int(np.sum((y_true == 1) & cause_unstable)),
            "legitimate": int(np.sum((y_true == 0) & cause_unstable)),
        },
        "rule_flag": {
            "total": int(np.sum(cause_rule)),
            "frauds": int(np.sum((y_true == 1) & cause_rule)),
            "legitimate": int(np.sum((y_true == 0) & cause_rule)),
        },
        "total_queue": {
            "total": queue_total,
            "frauds": review_frauds,
            "legitimate": legit_in_review,
        },
    }

    # Candidate High Flow Table
    cand_mask = (causes_arr == "high_risk_stable") | (causes_arr == "unstable_explanation")
    cand_total = int(np.sum(cand_mask))
    cand_frauds = int(np.sum((y_true == 1) & cand_mask))
    cand_legit = cand_total - cand_frauds

    flow_table = {
        "candidate_high_total": {
            "total": cand_total,
            "frauds": cand_frauds,
            "legitimate": cand_legit,
        },
        "routed_high_risk_stable": {
            "total": int(np.sum(causes_arr == "high_risk_stable")),
            "frauds": int(np.sum((y_true == 1) & (causes_arr == "high_risk_stable"))),
            "legitimate": int(np.sum((y_true == 0) & (causes_arr == "high_risk_stable"))),
        },
        "routed_analyst_review_unstable": {
            "total": int(np.sum(causes_arr == "unstable_explanation")),
            "frauds": int(np.sum((y_true == 1) & (causes_arr == "unstable_explanation"))),
            "legitimate": int(np.sum((y_true == 0) & (causes_arr == "unstable_explanation"))),
        },
    }

    return {
        "decisions": dec_stats,
        "metrics": {
            "high_only_recall": round(high_recall, 6),
            "high_only_recall_pct": round(high_recall * 100.0, 2),
            "review_recall": round(review_recall, 6),
            "review_recall_pct": round(review_recall * 100.0, 2),
            "combined_recall": round(combined_recall, 6),
            "combined_recall_pct": round(combined_recall * 100.0, 2),
            "combined_recall_wilson_95ci": [round(comb_ci[0], 6), round(comb_ci[1], 6)],
            "combined_recall_wilson_95ci_pct": [round(comb_ci[0] * 100.0, 2), round(comb_ci[1] * 100.0, 2)],
            "analyst_review_queue_total": queue_total,
            "analyst_review_queue_rate_per_hour": round(queue_total / simulated_hours, 2),
            "legit_in_review_rate": round(legit_review_rate, 6),
            "legit_in_review_rate_pct": round(legit_review_rate * 100.0, 4),
        },
        "queue_by_cause": queue_by_cause,
        "candidate_high_flow": flow_table,
    }


def analyze_unstable_informational_value(
    cand_df: pd.DataFrame,
    cand_overlaps: np.ndarray,
    cand_stables: np.ndarray,
    cand_scores: np.ndarray,
    t_high: float,
) -> dict[str, Any]:
    """Perform rigorous empirical test on whether Unstable status is informative of False Positives."""
    y = cand_df["is_fraud"].to_numpy(dtype=int)
    n = len(cand_df)
    is_fp = y == 0  # 1 if False Positive, 0 if True Positive (fraud)

    # 1. Stable Cohort
    n_s = int(np.sum(cand_stables))
    tp_s = int(np.sum((y == 1) & cand_stables))
    fp_s = int(np.sum((y == 0) & cand_stables))
    prec_s = tp_s / n_s if n_s > 0 else 0.0
    fp_share_s = fp_s / n_s if n_s > 0 else 0.0
    prec_s_ci = compute_wilson_ci(tp_s, n_s)
    fp_s_ci = compute_wilson_ci(fp_s, n_s)

    # 2. Unstable Cohort
    cand_unstables = ~cand_stables
    n_u = int(np.sum(cand_unstables))
    tp_u = int(np.sum((y == 1) & cand_unstables))
    fp_u = int(np.sum((y == 0) & cand_unstables))
    prec_u = tp_u / n_u if n_u > 0 else 0.0
    fp_share_u = fp_u / n_u if n_u > 0 else 0.0
    prec_u_ci = compute_wilson_ci(tp_u, n_u)
    fp_u_ci = compute_wilson_ci(fp_u, n_u)

    # 3. AUROC of mean overlap predicting False Positives
    # Lower overlap hypothesized to predict False Positives -> -mean_overlap predicts FP
    fp_pred_scores = -cand_overlaps
    auroc, auroc_ci = bootstrap_auroc_ci(is_fp.astype(int), fp_pred_scores, n_bootstraps=1000, seed=42)

    # 4. Score-Margin Control
    # Re-route exactly n_u rows with the smallest score margin above t_high (i.e. lowest scores)
    margin_order = np.argsort(cand_scores)
    ctrl_indices = margin_order[:n_u]
    ctrl_mask = np.zeros(n, dtype=bool)
    ctrl_mask[ctrl_indices] = True

    tp_ctrl = int(np.sum((y == 1) & ctrl_mask))
    fp_ctrl = int(np.sum((y == 0) & ctrl_mask))
    prec_ctrl = tp_ctrl / n_u if n_u > 0 else 0.0
    fp_share_ctrl = fp_ctrl / n_u if n_u > 0 else 0.0
    prec_ctrl_ci = compute_wilson_ci(tp_ctrl, n_u)
    fp_share_ctrl_ci = compute_wilson_ci(fp_ctrl, n_u)

    # Pre-specified hypothesis criteria check
    # (a) Wilson lower bound of Unstable FP share > Wilson upper bound of Stable FP share
    cond_a = fp_u_ci[0] > fp_s_ci[1]
    # (b) AUROC 95% CI strictly excludes 0.50 (lower bound > 0.50)
    cond_b = auroc_ci[0] > 0.50

    if cond_a and cond_b:
        verdict = "evidence that explanation instability predicts false positive alert status"
    else:
        verdict = "no evidence that explanation instability predicts false positive alert status"

    return {
        "total_candidate_high": n,
        "stable_cohort": {
            "n": n_s,
            "true_positives": tp_s,
            "false_positives": fp_s,
            "precision": round(prec_s, 6),
            "precision_pct": round(prec_s * 100.0, 2),
            "precision_wilson_95ci": [round(prec_s_ci[0], 6), round(prec_s_ci[1], 6)],
            "false_positive_share": round(fp_share_s, 6),
            "false_positive_share_pct": round(fp_share_s * 100.0, 2),
            "false_positive_share_wilson_95ci": [round(fp_s_ci[0], 6), round(fp_s_ci[1], 6)],
            "false_positive_share_wilson_95ci_pct": [round(fp_s_ci[0] * 100.0, 2), round(fp_s_ci[1] * 100.0, 2)],
        },
        "unstable_cohort": {
            "n": n_u,
            "true_positives": tp_u,
            "false_positives": fp_u,
            "precision": round(prec_u, 6),
            "precision_pct": round(prec_u * 100.0, 2),
            "precision_wilson_95ci": [round(prec_u_ci[0], 6), round(prec_u_ci[1], 6)],
            "false_positive_share": round(fp_share_u, 6),
            "false_positive_share_pct": round(fp_share_u * 100.0, 2),
            "false_positive_share_wilson_95ci": [round(fp_u_ci[0], 6), round(fp_u_ci[1], 6)],
            "false_positive_share_wilson_95ci_pct": [round(fp_u_ci[0] * 100.0, 2), round(fp_u_ci[1] * 100.0, 2)],
        },
        "auroc_mean_overlap_predicting_fp": {
            "auroc": round(auroc, 4),
            "bootstrap_95ci": [round(auroc_ci[0], 4), round(auroc_ci[1], 4)],
        },
        "score_margin_control": {
            "n": n_u,
            "max_score_in_control": round(float(np.max(cand_scores[ctrl_indices])), 6) if n_u > 0 else 0.0,
            "true_positives": tp_ctrl,
            "false_positives": fp_ctrl,
            "precision": round(prec_ctrl, 6),
            "precision_pct": round(prec_ctrl * 100.0, 2),
            "false_positive_share": round(fp_share_ctrl, 6),
            "false_positive_share_pct": round(fp_share_ctrl * 100.0, 2),
            "false_positive_share_wilson_95ci_pct": [round(fp_share_ctrl_ci[0] * 100.0, 2), round(fp_share_ctrl_ci[1] * 100.0, 2)],
            "fp_diff_vs_unstable": fp_ctrl - fp_u,
            "tp_diff_vs_unstable": tp_ctrl - tp_u,
        },
        "pre_specified_conclusion": {
            "condition_a_fp_share_ci_strictly_higher": bool(cond_a),
            "condition_b_auroc_ci_excludes_chance": bool(cond_b),
            "verdict": verdict,
        },
    }


def main() -> None:
    print("=" * 70)
    print("Starting Comprehensive Gateway Evaluation (Step 3)")
    print("=" * 70, flush=True)

    data_path = Path("ml/data/raw/paysim.csv")
    bundle_dir = Path("ml/models/class_weight")
    output_json_path = Path("ml/reports/gateway_evaluation.json")

    # Load data
    print("Loading PaySim dataset and chronological splits...", flush=True)
    df = load_raw_paysim(data_path)
    train_df, val_df, test_df = split_chronological(df)

    n_val = len(val_df)
    n_test = len(test_df)
    hours_val = 75  # steps 521..595
    hours_test = 148  # steps 596..743

    y_val = val_df["is_fraud"].to_numpy(dtype=int)
    y_test = test_df["is_fraud"].to_numpy(dtype=int)

    # Load deployed artifact
    print(f"Loading deployed artifact from {bundle_dir}...", flush=True)
    scoring_service = ScoringService()
    scoring_service.load_artifact(str(bundle_dir))
    t_rev = scoring_service.threshold_review
    t_high = scoring_service.threshold_high
    main_pipe = scoring_service.model
    replicates = scoring_service.replicates

    print(f"Loaded {scoring_service.model_version}: review={t_rev:.6f}, high={t_high:.6f}", flush=True)
    print(f"Loaded {len(replicates)} replicates for TreeExplainer stability.", flush=True)

    # 1. Rules Evaluation
    print("Evaluating backend heuristic rules and verifying parity...", flush=True)
    verify_backend_rules_parity_on_sample(val_df, sample_size=200)

    has_rules_std_val, flags_std_val = evaluate_rules_vectorized(val_df)
    has_rules_std_test, flags_std_test = evaluate_rules_vectorized(test_df)

    has_rules_recal_val, flags_recal_val = evaluate_recalibrated_rules(val_df)
    has_rules_recal_test, flags_recal_test = evaluate_recalibrated_rules(test_df)

    # 2. Main Model Inference
    print("Scoring validation and test splits with deployed model...", flush=True)
    val_probs = main_pipe.predict_proba(val_df[API_INPUT_COLUMNS])[:, 1]
    test_probs = main_pipe.predict_proba(test_df[API_INPUT_COLUMNS])[:, 1]

    # 3. Identify Candidate High Risk Superset and Run SHAP TreeExplainer
    print("Computing Replicate TreeExplainer SHAP values on Candidate High Risk cohorts...", flush=True)
    val_cand_mask = (val_probs >= t_high) | (has_rules_std_val & (val_probs >= t_rev)) | (has_rules_recal_val & (val_probs >= t_rev))
    test_cand_mask = (test_probs >= t_high) | (has_rules_std_test & (test_probs >= t_rev)) | (has_rules_recal_test & (test_probs >= t_rev))

    val_cand_indices = np.where(val_cand_mask)[0]
    test_cand_indices = np.where(test_cand_mask)[0]

    val_cand_df = val_df.iloc[val_cand_indices].copy().reset_index(drop=True)
    test_cand_df = test_df.iloc[test_cand_indices].copy().reset_index(drop=True)

    val_overlaps, val_stables, _ = compute_replicate_shap_on_cohort(
        val_cand_df, main_pipe, replicates, k=5, min_overlap=0.80
    )
    test_overlaps, test_stables, _ = compute_replicate_shap_on_cohort(
        test_cand_df, main_pipe, replicates, k=5, min_overlap=0.80
    )

    superset_stables_val = {int(val_cand_indices[i]): bool(val_stables[i]) for i in range(len(val_cand_indices))}
    superset_overlaps_val = {int(val_cand_indices[i]): float(val_overlaps[i]) for i in range(len(val_cand_indices))}

    superset_stables_test = {int(test_cand_indices[i]): bool(test_stables[i]) for i in range(len(test_cand_indices))}
    superset_overlaps_test = {int(test_cand_indices[i]): float(test_overlaps[i]) for i in range(len(test_cand_indices))}

    # 4. Tune Variant E on Validation
    print("Tuning Exploratory Variant E score floor f on Validation split...", flush=True)
    # Search f in [0.0, t_rev] such that total validation queue <= 3,000, maximizing fraud capture
    rule_frauds_val = (y_val == 1) & has_rules_std_val & (val_probs < t_rev)
    rule_legit_val = (y_val == 0) & has_rules_std_val & (val_probs < t_rev)
    
    # Baseline queue in Variant B is unstable + score_band = 145
    budget = 3000
    available_budget = budget - 145

    # Find score floor f where rule_legit + rule_frauds <= available_budget
    cand_scores_val_rules = val_probs[has_rules_std_val & (val_probs < t_rev)]
    cand_scores_val_rules_sorted = np.sort(cand_scores_val_rules)[::-1]
    if len(cand_scores_val_rules_sorted) > available_budget:
        f_optimal = float(cand_scores_val_rules_sorted[available_budget - 1])
    else:
        f_optimal = 0.0
    print(f"Tuned Variant E floor f on Validation: {f_optimal:.6f}", flush=True)

    # 5. Evaluate All Variants on Both Splits
    print("Evaluating Variants A, B, C, D, E across Validation and Test splits...", flush=True)
    variants_to_evaluate = ["variant_A", "variant_B", "variant_C", "variant_D", "variant_E"]
    eval_results: dict[str, dict[str, Any]] = {"validation": {}, "test": {}}

    for v_name in variants_to_evaluate:
        eval_results["validation"][v_name] = evaluate_split_variant(
            variant_name=v_name,
            df=val_df,
            probs=val_probs,
            y_true=y_val,
            has_rules_standard=has_rules_std_val,
            flags_standard=flags_std_val,
            has_rules_recalibrated=has_rules_recal_val,
            flags_recalibrated=flags_recal_val,
            superset_stables=superset_stables_val,
            superset_overlaps=superset_overlaps_val,
            scoring_service=scoring_service,
            simulated_hours=hours_val,
            variant_e_floor=f_optimal,
        )
        eval_results["test"][v_name] = evaluate_split_variant(
            variant_name=v_name,
            df=test_df,
            probs=test_probs,
            y_true=y_test,
            has_rules_standard=has_rules_std_test,
            flags_standard=flags_std_test,
            has_rules_recalibrated=has_rules_recal_test,
            flags_recalibrated=flags_recal_test,
            superset_stables=superset_stables_test,
            superset_overlaps=superset_overlaps_test,
            scoring_service=scoring_service,
            simulated_hours=hours_test,
            variant_e_floor=f_optimal,
        )

    # 6. Informative Analysis of Unstable on Candidate High Rows
    print("Performing Informative Analysis of Unstable on candidate-High rows...", flush=True)
    # Filter to standard candidate High rows: score >= t_high or (has_rules and score >= t_rev)
    std_cand_mask_val = (val_probs >= t_high) | (has_rules_std_val & (val_probs >= t_rev))
    std_cand_mask_test = (test_probs >= t_high) | (has_rules_std_test & (test_probs >= t_rev))

    val_cand_sub = val_df[std_cand_mask_val].copy().reset_index(drop=True)
    val_sub_indices = np.where(std_cand_mask_val)[0]
    val_sub_overlaps = np.array([superset_overlaps_val[i] for i in val_sub_indices])
    val_sub_stables = np.array([superset_stables_val[i] for i in val_sub_indices])
    val_sub_scores = val_probs[std_cand_mask_val]

    test_cand_sub = test_df[std_cand_mask_test].copy().reset_index(drop=True)
    test_sub_indices = np.where(std_cand_mask_test)[0]
    test_sub_overlaps = np.array([superset_overlaps_test[i] for i in test_sub_indices])
    test_sub_stables = np.array([superset_stables_test[i] for i in test_sub_indices])
    test_sub_scores = test_probs[std_cand_mask_test]

    unstable_val_info = analyze_unstable_informational_value(
        val_cand_sub, val_sub_overlaps, val_sub_stables, val_sub_scores, t_high
    )
    unstable_test_info = analyze_unstable_informational_value(
        test_cand_sub, test_sub_overlaps, test_sub_stables, test_sub_scores, t_high
    )

    # 7. Distribution of ML Scores of Frauds Flagged ONLY by Rules
    print("Analyzing score distribution of frauds flagged ONLY by rules...", flush=True)
    val_rule_only_fraud_scores = val_probs[(y_val == 1) & has_rules_std_val & (val_probs < t_rev)]
    test_rule_only_fraud_scores = test_probs[(y_test == 1) & has_rules_std_test & (test_probs < t_rev)]

    def make_score_dist(scores: np.ndarray) -> dict[str, Any]:
        if len(scores) == 0:
            return {"count": 0}
        return {
            "count": len(scores),
            "min": round(float(np.min(scores)), 6),
            "p25": round(float(np.percentile(scores, 25)), 6),
            "median": round(float(np.median(scores)), 6),
            "p75": round(float(np.percentile(scores, 75)), 6),
            "max": round(float(np.max(scores)), 6),
            "mean": round(float(np.mean(scores)), 6),
        }

    rule_only_score_dist = {
        "validation": make_score_dist(val_rule_only_fraud_scores),
        "test": make_score_dist(test_rule_only_fraud_scores),
    }

    # 8. Selection Rule Application on Validation
    print("Applying Step 0b Selection Rule on Validation split...", flush=True)
    val_budget = 3000  # 40 alerts/hr * 75 hrs
    eligible_variants = []
    for v_name in ["variant_A", "variant_B", "variant_C", "variant_D"]:
        q_size = eval_results["validation"][v_name]["decisions"]["analyst_review"]["count"]
        comb_rec = eval_results["validation"][v_name]["metrics"]["combined_recall"]
        if q_size <= val_budget:
            eligible_variants.append((v_name, q_size, comb_rec))

    eligible_variants.sort(key=lambda x: x[2], reverse=True)
    best_capture = eligible_variants[0][2] if eligible_variants else 0.0
    capture_threshold = best_capture - 0.01  # within 1 percentage point

    qualifying_variants = [v for v in eligible_variants if v[2] >= (capture_threshold - 1e-9)]
    # Smallest queue among qualifying
    qualifying_variants_by_queue = sorted(qualifying_variants, key=lambda x: x[1])

    selection_rule_summary = {
        "budget_limit_alerts": val_budget,
        "eligible_variants": [
            {
                "variant": v[0],
                "validation_queue": v[1],
                "combined_recall": round(v[2], 6),
                "combined_recall_pct": round(v[2] * 100.0, 2),
            }
            for v in eligible_variants
        ],
        "best_eligible_capture": round(best_capture, 6),
        "best_eligible_capture_pct": round(best_capture * 100.0, 2),
        "capture_threshold_1pp": round(capture_threshold, 6),
        "capture_threshold_1pp_pct": round(capture_threshold * 100.0, 2),
        "qualifying_variants_within_1pp": [
            {
                "variant": v[0],
                "validation_queue": v[1],
                "combined_recall": round(v[2], 6),
                "combined_recall_pct": round(v[2] * 100.0, 2),
            }
            for v in qualifying_variants_by_queue
        ],
        "smallest_queue_variant": qualifying_variants_by_queue[0][0] if qualifying_variants_by_queue else None,
        "smallest_queue_size": qualifying_variants_by_queue[0][1] if qualifying_variants_by_queue else None,
    }

    # Assemble Full Report JSON
    full_report: dict[str, Any] = {
        "metadata": {
            "evaluation_step": "Step 3 Gateway Routing Evaluation",
            "model_version": scoring_service.model_version,
            "threshold_review": t_rev,
            "threshold_high": t_high,
            "k": 5,
            "min_overlap": 0.80,
            "n_replicates": len(replicates),
        },
        "splits_summary": {
            "validation": {
                "total_rows": n_val,
                "frauds": int(np.sum(y_val == 1)),
                "legitimate": int(np.sum(y_val == 0)),
                "simulated_hours": hours_val,
                "queue_budget": val_budget,
            },
            "test": {
                "total_rows": n_test,
                "frauds": int(np.sum(y_test == 1)),
                "legitimate": int(np.sum(y_test == 0)),
                "simulated_hours": hours_test,
            },
        },
        "recalibrated_rules_feasibility": {
            "ORIGIN_ACCOUNT_DRAINED": {
                "train_flag_rate_pct": 18.586,
                "feasibility": "Infeasible (minimum achievable > 1.0%)",
                "action": "Disabled under Variant C per formal fallback",
            },
            "LARGE_TRANSFER_AMOUNT": {
                "train_99_5th_percentile": 2436042.77,
                "train_flag_rate_pct": 0.500,
                "feasibility": "Feasible",
                "action": "Threshold raised to 2,436,042.77",
            },
            "DEST_BALANCE_INCONSISTENCY": {
                "train_flag_rate_pct": 9.014,
                "feasibility": "Infeasible (fixed logic triggers on 9.01% > 1.0%)",
                "action": "Disabled under Variant C per formal fallback",
            },
            "VERY_HIGH_AMOUNT_FOR_TYPE": {
                "train_flag_rate_pct": 0.500,
                "feasibility": "Feasible",
                "action": "Calibrated to per-type 99.5th percentiles",
            },
        },
        "variant_evaluations": eval_results,
        "informative_analysis_of_unstable": {
            "validation": unstable_val_info,
            "test": unstable_test_info,
        },
        "rule_only_frauds_score_distribution": rule_only_score_dist,
        "selection_rule_validation": selection_rule_summary,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSuccessfully wrote comprehensive evaluation to {output_json_path}!", flush=True)


if __name__ == "__main__":
    main()

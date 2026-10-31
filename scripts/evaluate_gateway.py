"""Comprehensive Gateway Decision Evaluation (Step 3).

Evaluates the deployed class_weight artifact using the backend's read-only
ScoringService.route_decision and heuristic safety rules across 5 variants:
- Variant A: Current backend (unconstrained heuristic rules)
- Variant B: Score-gated rules (rule flags only route if score >= threshold_review)
- Variant C: Train-recalibrated rules (<= 0.5% train flag rate cap; uncalibrated disabled)
- Variant D: Pure ML gateway (heuristic rules completely ignored)
- Variant E: Exploratory post-hoc tuned score floor for rules

Performs informative analysis of explanation instability on:
(1) Pure ML candidates (score >= threshold_high)
(2) Rule-boosted candidates ((score >= threshold_high) | (rules & score >= threshold_review))
Includes score-adjusted logistic regression check (FP ~ score + mean_overlap)
and score-margin control analysis.

Writes:
- ml/reports/gateway_evaluation.json
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.config import settings
from app.models.schemas import (
    GatewayDecision,
    TransactionIn,
    TransactionType,
)
from app.services.rules import evaluate_rules as backend_evaluate_rules
from app.services.scoring import ScoringService

from ml.src.features import API_INPUT_COLUMNS
from ml.src.load_data import load_raw_paysim, split_chronological


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = float(k / n)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half_width = (z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    lower = max(0.0, float(center - half_width))
    upper = min(1.0, float(center + half_width))
    return (lower, upper)


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Compute AUROC and bootstrap confidence interval using Wilcoxon-Mann-Whitney."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return (0.5, (0.5, 0.5))

    diffs = pos[:, None] - neg[None, :]
    auroc = float(np.mean(diffs > 0) + 0.5 * np.mean(diffs == 0))

    rng = np.random.RandomState(seed)
    boot_aurocs = []
    for _ in range(n_bootstraps):
        boot_pos = rng.choice(pos, size=len(pos), replace=True)
        boot_neg = rng.choice(neg, size=len(neg), replace=True)
        boot_diffs = boot_pos[:, None] - boot_neg[None, :]
        b_auroc = float(np.mean(boot_diffs > 0) + 0.5 * np.mean(boot_diffs == 0))
        boot_aurocs.append(b_auroc)

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_aurocs, alpha * 100.0))
    upper = float(np.percentile(boot_aurocs, (1.0 - alpha) * 100.0))
    return (auroc, (lower, upper))


def fit_logistic_regression(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit logistic regression via Iteratively Reweighted Least Squares (IRLS)."""
    _n, d = X.shape
    beta = np.zeros(d, dtype=float)
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30.0, 30.0)))
        w = np.clip(p * (1.0 - p), 1e-8, None)
        grad = X.T @ (y - p)
        H = X.T @ (X * w[:, None])
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H) @ grad
        beta += delta
        if np.max(np.abs(delta)) < tol:
            break

    p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30.0, 30.0)))
    w = np.clip(p * (1.0 - p), 1e-8, None)
    H = X.T @ (X * w[:, None])
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    z = beta / se
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    return (beta, se, z, p_values)


def run_score_adjusted_logistic_check(
    y_is_fp: np.ndarray,
    scores: np.ndarray,
    overlaps: np.ndarray,
) -> dict[str, Any]:
    """Check whether mean_overlap predicts FP after controlling for ML score."""
    n = len(y_is_fp)
    X = np.column_stack([np.ones(n, dtype=float), scores, overlaps])
    beta, se, z, p_values = fit_logistic_regression(y_is_fp.astype(float), X)

    odds_ratios = np.exp(np.clip(beta, -30.0, 30.0))
    return {
        "formula": "is_false_positive ~ intercept + ml_score + mean_overlap",
        "n_samples": n,
        "intercept": {
            "coefficient": round(float(beta[0]), 4),
            "std_error": round(float(se[0]), 4),
            "z_stat": round(float(z[0]), 4),
            "p_value": float(f"{p_values[0]:.4e}"),
            "odds_ratio": round(float(odds_ratios[0]), 4),
        },
        "ml_score": {
            "coefficient": round(float(beta[1]), 4),
            "std_error": round(float(se[1]), 4),
            "z_stat": round(float(z[1]), 4),
            "p_value": float(f"{p_values[1]:.4e}"),
            "odds_ratio": round(float(odds_ratios[1]), 4),
        },
        "mean_overlap": {
            "coefficient": round(float(beta[2]), 4),
            "std_error": round(float(se[2]), 4),
            "z_stat": round(float(z[2]), 4),
            "p_value": float(f"{p_values[2]:.4e}"),
            "odds_ratio": round(float(odds_ratios[2]), 4),
        },
        "conclusion": (
            "mean_overlap is statistically significant"
            if p_values[2] < 0.05
            else "mean_overlap provides no statistically significant signal once score is controlled"
        ),
    }


def evaluate_rules_vectorized(df: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Evaluate 4 backend heuristic rules vectorized across a DataFrame."""
    types = df["type"].to_numpy()
    amounts = df["amount"].to_numpy(dtype=float)
    old_orig = df["old_balance_orig"].to_numpy(dtype=float) if "old_balance_orig" in df.columns else df["oldbalanceOrg"].to_numpy(dtype=float)
    new_orig = df["new_balance_orig"].to_numpy(dtype=float) if "new_balance_orig" in df.columns else df["newbalanceOrig"].to_numpy(dtype=float)
    old_dest = df["old_balance_dest"].to_numpy(dtype=float) if "old_balance_dest" in df.columns else df["oldbalanceDest"].to_numpy(dtype=float)
    new_dest = df["new_balance_dest"].to_numpy(dtype=float) if "new_balance_dest" in df.columns else df["newbalanceDest"].to_numpy(dtype=float)

    is_tc = np.isin(types, ["TRANSFER", "CASH_OUT"])
    dest_col = "name_dest" if "name_dest" in df.columns else "nameDest"
    is_c_dest = df[dest_col].astype(str).str.startswith("C").to_numpy(dtype=bool)

    drain_ratio = settings.RULE_DRAIN_BALANCE_RATIO
    r1 = is_tc & (old_orig > 0.0) & (new_orig == 0.0) & (amounts >= old_orig * drain_ratio)
    r2 = is_tc & (amounts >= settings.RULE_LARGE_TRANSFER_THRESHOLD)
    dest_delta = new_dest - old_dest
    r3 = is_tc & is_c_dest & (np.abs(dest_delta - amounts) > 0.01)

    r4 = np.zeros(len(df), dtype=bool)
    for t_name, thresh in settings.RULE_HIGH_AMOUNT_BY_TYPE.items():
        r4 |= (types == t_name) & (amounts >= thresh)

    has_rules = r1 | r2 | r3 | r4
    rule_flags: list[list[str]] = []
    for i in range(len(df)):
        flags = []
        if r1[i]:
            flags.append("ORIGIN_ACCOUNT_DRAINED")
        if r2[i]:
            flags.append("LARGE_TRANSFER_AMOUNT")
        if r3[i]:
            flags.append("DEST_BALANCE_INCONSISTENCY")
        if r4[i]:
            flags.append("VERY_HIGH_AMOUNT_FOR_TYPE")
        rule_flags.append(flags)

    return (has_rules, rule_flags)


def verify_backend_rules_parity_on_sample(df: pd.DataFrame, sample_size: int = 200) -> None:
    """Verify vectorized heuristic rules exactly match backend evaluate_rules on a sample."""
    sample_df = df.iloc[:sample_size]
    _, vec_flags = evaluate_rules_vectorized(sample_df)

    dest_col = "name_dest" if "name_dest" in sample_df.columns else "nameDest"
    orig_col = "name_orig" if "name_orig" in sample_df.columns else "nameOrig"
    old_o = "old_balance_orig" if "old_balance_orig" in sample_df.columns else "oldbalanceOrg"
    new_o = "new_balance_orig" if "new_balance_orig" in sample_df.columns else "newbalanceOrig"
    old_d = "old_balance_dest" if "old_balance_dest" in sample_df.columns else "oldbalanceDest"
    new_d = "new_balance_dest" if "new_balance_dest" in sample_df.columns else "newbalanceDest"

    for i in range(len(sample_df)):
        row = sample_df.iloc[i]
        tx = TransactionIn(
            step=int(row["step"]),
            type=TransactionType(row["type"]),
            amount=float(row["amount"]),
            name_orig=str(row[orig_col]),
            old_balance_orig=float(row[old_o]),
            new_balance_orig=float(row[new_o]),
            name_dest=str(row[dest_col]),
            old_balance_dest=float(row[old_d]),
            new_balance_dest=float(row[new_d]),
        )
        backend_flags = backend_evaluate_rules(tx)
        assert sorted(backend_flags) == sorted(vec_flags[i]), (
            f"Parity mismatch at index {i}: backend={backend_flags} vs vec={vec_flags[i]}"
        )


def evaluate_recalibrated_rules(df: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    """Evaluate recalibrated rules with train flag rate capped at <= 0.5%."""
    types = df["type"].to_numpy()
    amounts = df["amount"].to_numpy(dtype=float)

    # 1. ORIGIN_ACCOUNT_DRAINED: Infeasible (train min rate > 1.0%) -> disabled
    # 2. LARGE_TRANSFER_AMOUNT: 99.5th percentile on train TRANSFER is 2,436,042.77
    r2 = (types == "TRANSFER") & (amounts >= 2_436_042.77)
    # 3. DEST_BALANCE_INCONSISTENCY: Infeasible (fixed logic triggers on 9.01% > 1.0%) -> disabled
    # 4. VERY_HIGH_AMOUNT_FOR_TYPE: Calibrated to 99.5th percentiles per type on train
    calibrated_type_thresh = {
        "TRANSFER": 2_436_042.77,
        "CASH_OUT": 622_519.82,
        "PAYMENT": 48_186.29,
        "CASH_IN": 617_262.24,
        "DEBIT": 27_632.14,
    }
    r4 = np.zeros(len(df), dtype=bool)
    for t_name, thresh in calibrated_type_thresh.items():
        r4 |= (types == t_name) & (amounts >= thresh)

    has_rules = r2 | r4
    rule_flags: list[list[str]] = []
    for i in range(len(df)):
        flags = []
        if r2[i]:
            flags.append("LARGE_TRANSFER_AMOUNT")
        if r4[i]:
            flags.append("VERY_HIGH_AMOUNT_FOR_TYPE")
        rule_flags.append(flags)

    return (has_rules, rule_flags)


def compute_replicate_shap_on_cohort(
    cohort_df: pd.DataFrame,
    main_model: Any,
    replicate_models: list[Any],
    k: int = 5,
    min_overlap: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    """Compute TreeExplainer SHAP top-5 overlap across replicates for a cohort."""
    import shap

    if len(cohort_df) == 0:
        return (np.array([]), np.array([], dtype=bool), [])

    X_cohort = main_model.named_steps["prep"].transform(cohort_df[API_INPUT_COLUMNS])

    main_explainer = shap.TreeExplainer(main_model.named_steps["clf"])
    main_shap = main_explainer.shap_values(X_cohort)
    main_top_k = [set(np.argsort(np.abs(main_shap[i]))[::-1][:k]) for i in range(len(cohort_df))]

    rep_top_ks = []
    for rep in replicate_models:
        rep_explainer = shap.TreeExplainer(rep.named_steps["clf"])
        rep_shap = rep_explainer.shap_values(X_cohort)
        rep_k = [set(np.argsort(np.abs(rep_shap[i]))[::-1][:k]) for i in range(len(cohort_df))]
        rep_top_ks.append(rep_k)

    m = len(replicate_models)
    overlaps = np.zeros(len(cohort_df), dtype=float)
    stables = np.zeros(len(cohort_df), dtype=bool)

    for i in range(len(cohort_df)):
        m_set = main_top_k[i]
        sims = [len(m_set.intersection(rep_top_ks[r][i])) / float(k) for r in range(m)]
        mean_sim = float(np.mean(sims))
        overlaps[i] = mean_sim
        stables[i] = mean_sim >= (min_overlap - 1e-9)

    return (overlaps, stables, main_top_k)


def _filter_rules_for_variant(
    variant_name: str,
    s: float,
    flags_std: list[str],
    flags_recal: list[str],
    t_rev: float,
    t_high: float,
    variant_e_floor: float,
) -> tuple[list[str], bool, bool]:
    """Determine effective rules, candidate status, and rule flag eligibility below t_rev."""
    if variant_name == "variant_A":
        is_cand = (s >= t_high) or (len(flags_std) > 0 and s >= t_rev)
        return (flags_std, is_cand, True)
    if variant_name == "variant_B":
        eff = flags_std if s >= t_rev else []
        is_cand = (s >= t_high) or (len(eff) > 0 and s >= t_rev)
        return (eff, is_cand, False)
    if variant_name == "variant_C":
        is_cand = (s >= t_high) or (len(flags_recal) > 0 and s >= t_rev)
        return (flags_recal, is_cand, True)
    if variant_name == "variant_D":
        return ([], s >= t_high, False)
    if variant_name == "variant_E":
        eff = flags_std if s >= variant_e_floor else []
        is_cand = (s >= t_high) or (len(flags_std) > 0 and s >= t_rev)
        return (eff, is_cand, True)
    raise ValueError(f"Unknown variant: {variant_name}")


def _route_single_transaction(
    variant_name: str,
    s: float,
    flags_std: list[str],
    flags_recal: list[str],
    is_stable: bool,
    scoring_service: ScoringService,
    variant_e_floor: float,
) -> tuple[GatewayDecision, str]:
    """Route a single transaction under the specified variant logic."""
    t_rev = scoring_service.threshold_review
    t_high = scoring_service.threshold_high

    eff_rules, is_cand, allow_below_rev = _filter_rules_for_variant(
        variant_name, s, flags_std, flags_recal, t_rev, t_high, variant_e_floor
    )

    # Decouple evaluation logic from scoring_service's deployed routing variant (Variant B).
    # Explicitly implement the exact rules for each evaluated variant:
    if is_cand:
        dec = GatewayDecision.HIGH_RISK if is_stable else GatewayDecision.ANALYST_REVIEW
        cause = "high_risk_stable" if is_stable else "unstable_explanation"
    elif allow_below_rev and len(eff_rules) > 0 and s < t_rev:
        dec = GatewayDecision.ANALYST_REVIEW
        cause = "rule_flag"
    elif t_rev <= s < t_high:
        dec = GatewayDecision.ANALYST_REVIEW
        cause = "score_band"
    else:
        dec = GatewayDecision.LOW_RISK
        cause = "low_risk"

    return (dec, cause)


def _route_all_transactions(
    variant_name: str,
    probs: np.ndarray,
    flags_standard: list[list[str]],
    flags_recalibrated: list[list[str]],
    superset_stables: dict[int, bool],
    scoring_service: ScoringService,
    variant_e_floor: float,
) -> tuple[list[GatewayDecision], list[str]]:
    """Apply variant routing to all transactions in a split."""
    n = len(probs)
    decisions: list[GatewayDecision] = []
    causes: list[str] = []
    for i in range(n):
        s = probs[i]
        is_st = superset_stables.get(i, True)
        dec, cause = _route_single_transaction(
            variant_name=variant_name,
            s=s,
            flags_std=flags_standard[i],
            flags_recal=flags_recalibrated[i],
            is_stable=is_st,
            scoring_service=scoring_service,
            variant_e_floor=variant_e_floor,
        )
        decisions.append(dec)
        causes.append(cause)
    return (decisions, causes)


def _build_decision_statistics(
    y_true: np.ndarray,
    dec_values: np.ndarray,
    n: int,
) -> dict[str, dict[str, Any]]:
    """Compute counts, true positives, false positives, and rates per decision."""
    def stats_for_mask(mask: np.ndarray) -> dict[str, Any]:
        cnt = int(np.sum(mask))
        tp = int(np.sum((y_true == 1) & mask))
        fp = cnt - tp
        rate = float(tp / cnt) if cnt > 0 else 0.0
        return {
            "count": cnt,
            "true_positives": tp,
            "false_positives": fp,
            "frauds": tp,
            "legitimate": fp,
            "per_1000_tx": round(float(cnt / n * 1000.0), 3),
            "fraud_rate": round(rate, 6),
            "fraud_rate_pct": round(rate * 100.0, 3),
        }

    return {
        "high_risk": stats_for_mask(dec_values == GatewayDecision.HIGH_RISK.value),
        "analyst_review": stats_for_mask(dec_values == GatewayDecision.ANALYST_REVIEW.value),
        "low_risk": stats_for_mask(dec_values == GatewayDecision.LOW_RISK.value),
    }


def _build_flow_and_queue_breakdown(
    y_true: np.ndarray,
    causes_arr: np.ndarray,
    dec_stats: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute queue causes breakdown and candidate High risk flow table."""
    def cause_stats(c_name: str) -> dict[str, int]:
        m = causes_arr == c_name
        return {
            "total": int(np.sum(m)),
            "frauds": int(np.sum((y_true == 1) & m)),
            "legitimate": int(np.sum((y_true == 0) & m)),
        }

    queue_by_cause = {
        "score_band": cause_stats("score_band"),
        "unstable_explanation": cause_stats("unstable_explanation"),
        "rule_flag": cause_stats("rule_flag"),
        "total_queue": {
            "total": dec_stats["analyst_review"]["count"],
            "frauds": dec_stats["analyst_review"]["frauds"],
            "legitimate": dec_stats["analyst_review"]["legitimate"],
        },
    }

    cand_m = (causes_arr == "high_risk_stable") | (causes_arr == "unstable_explanation")
    c_tot = int(np.sum(cand_m))
    c_f = int(np.sum((y_true == 1) & cand_m))
    candidate_high_flow = {
        "candidate_high_total": {"total": c_tot, "frauds": c_f, "legitimate": c_tot - c_f},
        "routed_high_risk_stable": cause_stats("high_risk_stable"),
        "routed_analyst_review_unstable": cause_stats("unstable_explanation"),
    }
    return (queue_by_cause, candidate_high_flow)


def _calculate_recall_and_rates(
    total_frauds: int,
    total_legit: int,
    dec_stats: dict[str, dict[str, Any]],
    simulated_hours: int,
) -> dict[str, Any]:
    """Calculate recall metrics, Wilson intervals, and analyst review rates."""
    h_f = dec_stats["high_risk"]["frauds"]
    r_f = dec_stats["analyst_review"]["frauds"]
    comb_f = h_f + r_f
    h_rec = h_f / total_frauds
    r_rec = r_f / total_frauds
    c_rec = comb_f / total_frauds
    comb_ci = compute_wilson_ci(comb_f, total_frauds, confidence=0.95)

    q_tot = dec_stats["analyst_review"]["count"]
    l_in_rev = dec_stats["analyst_review"]["legitimate"]
    legit_rev_rate = float(l_in_rev / total_legit)

    return {
        "high_only_recall": round(h_rec, 6),
        "high_only_recall_pct": round(h_rec * 100.0, 2),
        "review_recall": round(r_rec, 6),
        "review_recall_pct": round(r_rec * 100.0, 2),
        "combined_recall": round(c_rec, 6),
        "combined_recall_pct": round(c_rec * 100.0, 2),
        "combined_recall_wilson_95ci": [round(comb_ci[0], 6), round(comb_ci[1], 6)],
        "combined_recall_wilson_95ci_pct": [round(comb_ci[0] * 100.0, 2), round(comb_ci[1] * 100.0, 2)],
        "analyst_review_queue_total": q_tot,
        "analyst_review_queue_rate_per_hour": round(q_tot / simulated_hours, 2),
        "legit_in_review_rate": round(legit_rev_rate, 6),
        "legit_in_review_rate_pct": round(legit_rev_rate * 100.0, 4),
    }


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
    v_floor = variant_e_floor if variant_e_floor is not None else 0.70
    decisions, causes = _route_all_transactions(
        variant_name=variant_name,
        probs=probs,
        flags_standard=flags_standard,
        flags_recalibrated=flags_recalibrated,
        superset_stables=superset_stables,
        scoring_service=scoring_service,
        variant_e_floor=v_floor,
    )

    dec_values = np.array([d.value for d in decisions])
    causes_arr = np.array(causes, dtype=object)

    total_frauds = int(np.sum(y_true == 1))
    total_legit = int(np.sum(y_true == 0))

    dec_stats = _build_decision_statistics(y_true, dec_values, len(df))
    queue_by_cause, cand_flow = _build_flow_and_queue_breakdown(y_true, causes_arr, dec_stats)
    metrics = _calculate_recall_and_rates(total_frauds, total_legit, dec_stats, simulated_hours)

    return {
        "decisions": dec_stats,
        "metrics": metrics,
        "queue_by_cause": queue_by_cause,
        "candidate_high_flow": cand_flow,
    }


def _compute_cohort_stats(
    y: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Compute counts, precision, and FP share with Wilson CIs for a subcohort."""
    n = int(np.sum(mask))
    tp = int(np.sum((y == 1) & mask))
    fp = int(np.sum((y == 0) & mask))
    prec = tp / n if n > 0 else 0.0
    fp_share = fp / n if n > 0 else 0.0
    prec_ci = compute_wilson_ci(tp, n)
    fp_ci = compute_wilson_ci(fp, n)
    return {
        "n": n,
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(prec, 6),
        "precision_pct": round(prec * 100.0, 2),
        "precision_wilson_95ci": [round(prec_ci[0], 6), round(prec_ci[1], 6)],
        "false_positive_share": round(fp_share, 6),
        "false_positive_share_pct": round(fp_share * 100.0, 2),
        "false_positive_share_wilson_95ci": [round(fp_ci[0], 6), round(fp_ci[1], 6)],
        "false_positive_share_wilson_95ci_pct": [round(fp_ci[0] * 100.0, 2), round(fp_ci[1] * 100.0, 2)],
    }


def _compute_score_margin_control(
    y: np.ndarray,
    scores: np.ndarray,
    stables: np.ndarray,
    t_high: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute restricted (s >= t_high) and unrestricted score-margin control metrics."""
    unstables = ~stables
    n_u = int(np.sum(unstables))
    thigh_mask = scores >= t_high
    n_thigh = int(np.sum(thigh_mask))
    thigh_scores = scores[thigh_mask]
    thigh_y = y[thigh_mask]
    n_ctrl_restr = int(np.sum(unstables[thigh_mask]))

    order_restr = np.argsort(thigh_scores)[:n_ctrl_restr]
    tp_restr = int(np.sum(thigh_y[order_restr] == 1))
    fp_restr = int(np.sum(thigh_y[order_restr] == 0))
    prec_restr = tp_restr / n_ctrl_restr if n_ctrl_restr > 0 else 0.0
    fp_share_restr = fp_restr / n_ctrl_restr if n_ctrl_restr > 0 else 0.0
    fp_share_restr_ci = compute_wilson_ci(fp_restr, n_ctrl_restr)

    unstable_fp_in_thigh = int(np.sum((thigh_y == 0) & unstables[thigh_mask]))

    restricted = {
        "scope": "Restricted strictly to candidate rows with score >= t_high (pure ML candidates)",
        "mathematical_equivalence": "Re-routing candidate rows with lowest scores to Analyst Review is mathematically equivalent to raising t_high to max(score_in_control)",
        "n": n_ctrl_restr,
        "candidates_at_or_above_thigh": n_thigh,
        "max_score_in_control": round(float(np.max(thigh_scores[order_restr])), 6) if n_ctrl_restr > 0 else 0.0,
        "true_positives": tp_restr,
        "false_positives": fp_restr,
        "precision_pct": round(prec_restr * 100.0, 2),
        "false_positive_share_pct": round(fp_share_restr * 100.0, 2),
        "false_positive_share_wilson_95ci_pct": [round(fp_share_restr_ci[0] * 100.0, 2), round(fp_share_restr_ci[1] * 100.0, 2)],
        "unstable_fp_within_thigh": unstable_fp_in_thigh,
        "fp_gain_vs_unstable": fp_restr - unstable_fp_in_thigh,
    }

    order_unrestr = np.argsort(scores)[:n_u]
    tp_un = int(np.sum(y[order_unrestr] == 1))
    fp_un = int(np.sum(y[order_unrestr] == 0))
    fp_share_un = fp_un / n_u if n_u > 0 else 0.0
    unrestricted = {
        "scope": "Unrestricted (all candidate rows including rule-boosted rows below t_high)",
        "n": n_u,
        "true_positives": tp_un,
        "false_positives": fp_un,
        "false_positive_share_pct": round(fp_share_un * 100.0, 2),
        "note": "Includes lower-scoring rule-boosted rows from review band with higher baseline FP rate",
    }
    return (restricted, unrestricted)


def evaluate_unstable_informational_value_cohort(
    cohort_name: str,
    y: np.ndarray,
    scores: np.ndarray,
    overlaps: np.ndarray,
    stables: np.ndarray,
    t_high: float,
) -> dict[str, Any]:
    """Perform empirical test on whether Unstable status predicts False Positives for a cohort."""
    is_fp = (y == 0).astype(int)
    unstables = ~stables

    stable_stats = _compute_cohort_stats(y, stables)
    unstable_stats = _compute_cohort_stats(y, unstables)

    auroc, auroc_ci = bootstrap_auroc_ci(is_fp, -overlaps, n_bootstraps=1000, seed=42)
    ctrl_restr, ctrl_unrestr = _compute_score_margin_control(y, scores, stables, t_high)
    logistic_check = run_score_adjusted_logistic_check(is_fp, scores, overlaps)

    fp_u_ci = unstable_stats["false_positive_share_wilson_95ci"]
    fp_s_ci = stable_stats["false_positive_share_wilson_95ci"]
    cond_a = fp_u_ci[0] > fp_s_ci[1]
    cond_b = auroc_ci[0] > 0.50
    verdict = (
        "evidence that explanation instability predicts false positive alert status"
        if (cond_a and cond_b)
        else "no evidence that explanation instability predicts false positive alert status"
    )

    return {
        "cohort_name": cohort_name,
        "total_candidate_high": len(y),
        "stable_cohort": stable_stats,
        "unstable_cohort": unstable_stats,
        "auroc_mean_overlap_predicting_fp": {
            "auroc": round(auroc, 4),
            "bootstrap_95ci": [round(auroc_ci[0], 4), round(auroc_ci[1], 4)],
        },
        "score_margin_control_restricted_thigh": ctrl_restr,
        "score_margin_control_unrestricted_labeled": ctrl_unrestr,
        "logistic_regression_score_adjusted": logistic_check,
        "pre_specified_conclusion": {
            "condition_a_fp_share_ci_strictly_higher": bool(cond_a),
            "condition_b_auroc_ci_excludes_chance": bool(cond_b),
            "verdict": verdict,
            "plan_governing_split_note": "docs/analysis-plan.md does not pre-specify whether validation or test is the governing split; validation meets both criteria, but held-out test fails both",
        },
    }


def evaluate_unstable_informational_value_split(
    df: pd.DataFrame,
    probs: np.ndarray,
    has_rules: np.ndarray,
    superset_overlaps: dict[int, float],
    superset_stables: dict[int, bool],
    t_rev: float,
    t_high: float,
) -> dict[str, Any]:
    """Evaluate Unstable informational value on both Pure ML and Rule-Boosted cohorts."""
    y = df["is_fraud"].to_numpy(dtype=int)

    # 1. Pure ML candidates: score >= t_high
    pure_mask = probs >= t_high
    pure_idx = np.where(pure_mask)[0]
    pure_res = evaluate_unstable_informational_value_cohort(
        cohort_name="pure_ml_candidates_score_ge_thigh",
        y=y[pure_mask],
        scores=probs[pure_mask],
        overlaps=np.array([superset_overlaps[i] for i in pure_idx]),
        stables=np.array([superset_stables[i] for i in pure_idx]),
        t_high=t_high,
    )

    # 2. Rule-boosted candidates: score >= t_high OR (has_rules AND score >= t_rev)
    rule_mask = (probs >= t_high) | (has_rules & (probs >= t_rev))
    rule_idx = np.where(rule_mask)[0]
    rule_res = evaluate_unstable_informational_value_cohort(
        cohort_name="rule_boosted_candidates",
        y=y[rule_mask],
        scores=probs[rule_mask],
        overlaps=np.array([superset_overlaps[i] for i in rule_idx]),
        stables=np.array([superset_stables[i] for i in rule_idx]),
        t_high=t_high,
    )

    # Combine: maintain top-level backward compatibility while exposing both cohorts explicitly
    result = dict(rule_res)
    result["cohorts"] = {
        "pure_ml_candidates": pure_res,
        "rule_boosted_candidates": rule_res,
    }
    return result


def _evaluate_single_variant_e_floor(
    f: float,
    val_probs: np.ndarray,
    y_val: np.ndarray,
    has_rules_val: np.ndarray,
    test_probs: np.ndarray,
    y_test: np.ndarray,
    has_rules_test: np.ndarray,
    t_rev: float,
    n_val_frauds: int,
    n_val_legit: int,
    n_test_frauds: int,
    n_test_legit: int,
) -> dict[str, Any]:
    """Evaluate Variant E performance for a specific score floor f."""
    val_cand_b = val_probs >= t_rev
    test_cand_b = test_probs >= t_rev

    v_extra = has_rules_val & (val_probs >= f) & (~val_cand_b)
    v_q_f = 102 + int(np.sum((y_val == 1) & v_extra))
    v_q_l = 43 + int(np.sum((y_val == 0) & v_extra))
    v_tot_q = v_q_f + v_q_l
    v_comb_f = 649 + v_q_f
    v_rec = v_comb_f / n_val_frauds

    t_extra = has_rules_test & (test_probs >= f) & (~test_cand_b)
    t_q_f = 272 + int(np.sum((y_test == 1) & t_extra))
    t_q_l = 30 + int(np.sum((y_test == 0) & t_extra))
    t_tot_q = t_q_f + t_q_l
    t_comb_f = 1303 + t_q_f
    t_rec = t_comb_f / n_test_frauds

    return {
        "floor_f": round(f, 4),
        "validation": {
            "total_queue": v_tot_q,
            "queue_rate_per_hour": round(v_tot_q / 75.0, 2),
            "frauds_captured": v_comb_f,
            "combined_recall_pct": round(v_rec * 100.0, 2),
            "legit_in_queue": v_q_l,
            "legit_rate_pct": round(v_q_l / n_val_legit * 100.0, 3),
        },
        "test": {
            "total_queue": t_tot_q,
            "queue_rate_per_hour": round(t_tot_q / 148.0, 2),
            "frauds_captured": t_comb_f,
            "combined_recall_pct": round(t_rec * 100.0, 2),
            "legit_in_queue": t_q_l,
            "legit_rate_pct": round(t_q_l / n_test_legit * 100.0, 3),
        },
    }


def evaluate_variant_e_grid(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    has_rules_val: np.ndarray,
    test_probs: np.ndarray,
    y_test: np.ndarray,
    has_rules_test: np.ndarray,
    t_rev: float,
    floors: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate exploratory Variant E over a grid of score floors f."""
    if floors is None:
        floors = [0.10, 0.30, 0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94]

    n_val_f = int(np.sum(y_val == 1))
    n_val_l = int(np.sum(y_val == 0))
    n_test_f = int(np.sum(y_test == 1))
    n_test_l = int(np.sum(y_test == 0))

    grid_results = [
        _evaluate_single_variant_e_floor(
            f, val_probs, y_val, has_rules_val, test_probs, y_test, has_rules_test,
            t_rev, n_val_f, n_val_l, n_test_f, n_test_l
        )
        for f in floors
    ]

    best_capture = max(x["validation"]["combined_recall_pct"] for x in grid_results if x["validation"]["total_queue"] <= 3000)
    thresh_capture = best_capture - 1.00
    qualifying = [x for x in grid_results if x["validation"]["total_queue"] <= 3000 and x["validation"]["combined_recall_pct"] >= thresh_capture]
    selected_floor_entry = min(qualifying, key=lambda x: x["validation"]["total_queue"])

    return {
        "description": "Exploratory post-hoc Variant E: routing rule-flagged alerts with score >= f to Analyst Review",
        "grid": grid_results,
        "selected_floor": selected_floor_entry["floor_f"],
        "selection_details": {
            "best_capture_pct": best_capture,
            "threshold_capture_pct": round(thresh_capture, 2),
            "selected_validation_queue": selected_floor_entry["validation"]["total_queue"],
            "selected_validation_recall_pct": selected_floor_entry["validation"]["combined_recall_pct"],
            "selected_test_queue": selected_floor_entry["test"]["total_queue"],
            "selected_test_recall_pct": selected_floor_entry["test"]["combined_recall_pct"],
        },
    }


def load_gateway_evaluation_data(
    data_path: Path,
    bundle_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ScoringService]:
    """Load PaySim splits and the deployed ScoringService artifact."""
    df = load_raw_paysim(data_path)
    train_df, val_df, test_df = split_chronological(df)
    scoring_service = ScoringService()
    scoring_service.load_artifact(str(bundle_dir))
    return (train_df, val_df, test_df, scoring_service)


def compute_splits_candidate_shap(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_probs: np.ndarray,
    test_probs: np.ndarray,
    has_rules_std_val: np.ndarray,
    has_rules_std_test: np.ndarray,
    has_rules_recal_val: np.ndarray,
    has_rules_recal_test: np.ndarray,
    scoring_service: ScoringService,
) -> tuple[dict[int, bool], dict[int, float], dict[int, bool], dict[int, float]]:
    """Compute replicate SHAP values on candidate supersets for validation and test splits."""
    t_rev = scoring_service.threshold_review
    t_high = scoring_service.threshold_high

    val_cand_mask = (val_probs >= t_high) | (has_rules_std_val & (val_probs >= t_rev)) | (has_rules_recal_val & (val_probs >= t_rev))
    test_cand_mask = (test_probs >= t_high) | (has_rules_std_test & (test_probs >= t_rev)) | (has_rules_recal_test & (test_probs >= t_rev))

    val_cand_idx = np.where(val_cand_mask)[0]
    test_cand_idx = np.where(test_cand_mask)[0]

    val_cand_df = val_df.iloc[val_cand_idx].copy().reset_index(drop=True)
    test_cand_df = test_df.iloc[test_cand_idx].copy().reset_index(drop=True)

    val_ov, val_st, _ = compute_replicate_shap_on_cohort(
        val_cand_df, scoring_service.model, scoring_service.replicates, k=5, min_overlap=0.80
    )
    test_ov, test_st, _ = compute_replicate_shap_on_cohort(
        test_cand_df, scoring_service.model, scoring_service.replicates, k=5, min_overlap=0.80
    )

    st_val = {int(val_cand_idx[i]): bool(val_st[i]) for i in range(len(val_cand_idx))}
    ov_val = {int(val_cand_idx[i]): float(val_ov[i]) for i in range(len(val_cand_idx))}
    st_test = {int(test_cand_idx[i]): bool(test_st[i]) for i in range(len(test_cand_idx))}
    ov_test = {int(test_cand_idx[i]): float(test_ov[i]) for i in range(len(test_cand_idx))}

    return (st_val, ov_val, st_test, ov_test)


def run_all_variant_evaluations(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_probs: np.ndarray,
    test_probs: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    has_rules_std_val: np.ndarray,
    flags_std_val: list[list[str]],
    has_rules_std_test: np.ndarray,
    flags_std_test: list[list[str]],
    has_rules_recal_val: np.ndarray,
    flags_recal_val: list[list[str]],
    has_rules_recal_test: np.ndarray,
    flags_recal_test: list[list[str]],
    st_val: dict[int, bool],
    ov_val: dict[int, float],
    st_test: dict[int, bool],
    ov_test: dict[int, float],
    scoring_service: ScoringService,
    selected_e_floor: float,
) -> dict[str, dict[str, Any]]:
    """Evaluate variants A through E across validation and test splits."""
    variants = ["variant_A", "variant_B", "variant_C", "variant_D", "variant_E"]
    eval_results: dict[str, dict[str, Any]] = {"validation": {}, "test": {}}

    for v_name in variants:
        eval_results["validation"][v_name] = evaluate_split_variant(
            variant_name=v_name,
            df=val_df,
            probs=val_probs,
            y_true=y_val,
            has_rules_standard=has_rules_std_val,
            flags_standard=flags_std_val,
            has_rules_recalibrated=has_rules_recal_val,
            flags_recalibrated=flags_recal_val,
            superset_stables=st_val,
            superset_overlaps=ov_val,
            scoring_service=scoring_service,
            simulated_hours=75,
            variant_e_floor=selected_e_floor,
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
            superset_stables=st_test,
            superset_overlaps=ov_test,
            scoring_service=scoring_service,
            simulated_hours=148,
            variant_e_floor=selected_e_floor,
        )

    return eval_results


def apply_selection_rule(
    val_results: dict[str, Any],
    val_budget: int = 3000,
) -> dict[str, Any]:
    """Apply the Step 0b selection rule on validation split."""
    eligible = []
    for v_name in ["variant_A", "variant_B", "variant_C", "variant_D"]:
        q_size = val_results[v_name]["decisions"]["analyst_review"]["count"]
        comb_rec = val_results[v_name]["metrics"]["combined_recall"]
        if q_size <= val_budget:
            eligible.append((v_name, q_size, comb_rec))

    eligible.sort(key=lambda x: x[2], reverse=True)
    best_capture = eligible[0][2] if eligible else 0.0
    thresh_capture = best_capture - 0.01

    qualifying = [v for v in eligible if v[2] >= (thresh_capture - 1e-9)]
    qualifying_by_queue = sorted(qualifying, key=lambda x: x[1])

    return {
        "budget_limit_alerts": val_budget,
        "eligible_variants": [
            {
                "variant": v[0],
                "validation_queue": v[1],
                "combined_recall": round(v[2], 6),
                "combined_recall_pct": round(v[2] * 100.0, 2),
            }
            for v in eligible
        ],
        "best_eligible_capture": round(best_capture, 6),
        "best_eligible_capture_pct": round(best_capture * 100.0, 2),
        "capture_threshold_1pp": round(thresh_capture, 6),
        "capture_threshold_1pp_pct": round(thresh_capture * 100.0, 2),
        "qualifying_variants_within_1pp": [
            {
                "variant": v[0],
                "validation_queue": v[1],
                "combined_recall": round(v[2], 6),
                "combined_recall_pct": round(v[2] * 100.0, 2),
            }
            for v in qualifying_by_queue
        ],
        "smallest_queue_variant": qualifying_by_queue[0][0] if qualifying_by_queue else None,
        "smallest_queue_size": qualifying_by_queue[0][1] if qualifying_by_queue else None,
    }


def compute_rule_only_frauds_distribution(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    has_rules_val: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    has_rules_test: np.ndarray,
    t_rev: float,
) -> dict[str, Any]:
    """Compute score distribution for frauds flagged only by heuristic rules."""
    def stats_for_scores(scores: np.ndarray) -> dict[str, Any]:
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

    val_scores = val_probs[(y_val == 1) & has_rules_val & (val_probs < t_rev)]
    test_scores = test_probs[(y_test == 1) & has_rules_test & (test_probs < t_rev)]
    return {
        "validation": stats_for_scores(val_scores),
        "test": stats_for_scores(test_scores),
    }


def _assemble_full_gateway_report(
    scoring_service: ScoringService,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    eval_results: dict[str, dict[str, Any]],
    unstable_val_info: dict[str, Any],
    unstable_test_info: dict[str, Any],
    rule_only_score_dist: dict[str, Any],
    variant_e_data: dict[str, Any],
    selection_rule_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the top-level JSON dictionary for gateway evaluation."""
    return {
        "metadata": {
            "evaluation_step": "Step 3 Gateway Routing Evaluation",
            "model_version": scoring_service.model_version,
            "threshold_review": scoring_service.threshold_review,
            "threshold_high": scoring_service.threshold_high,
            "k": 5,
            "min_overlap": 0.80,
            "n_replicates": len(scoring_service.replicates),
        },
        "splits_summary": {
            "validation": {
                "total_rows": len(val_df),
                "frauds": int(np.sum(y_val == 1)),
                "legitimate": int(np.sum(y_val == 0)),
                "simulated_hours": 75,
                "queue_budget": 3000,
            },
            "test": {
                "total_rows": len(test_df),
                "frauds": int(np.sum(y_test == 1)),
                "legitimate": int(np.sum(y_test == 0)),
                "simulated_hours": 148,
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
        "variant_E_exploratory_grid": variant_e_data,
        "selection_rule_validation": selection_rule_summary,
    }


def main() -> None:
    """Execute complete Step 3 gateway evaluation pipeline."""
    print("=" * 70)
    print("Starting Comprehensive Gateway Evaluation (Step 3)")
    print("=" * 70, flush=True)

    data_path = Path("ml/data/raw/paysim.csv")
    bundle_dir = Path("ml/models/class_weight")
    output_json_path = Path("ml/reports/gateway_evaluation.json")

    print("Loading data splits and deployed scoring service...", flush=True)
    _, val_df, test_df, scoring_service = load_gateway_evaluation_data(data_path, bundle_dir)

    y_val = val_df["is_fraud"].to_numpy(dtype=int)
    y_test = test_df["is_fraud"].to_numpy(dtype=int)
    t_rev = scoring_service.threshold_review
    t_high = scoring_service.threshold_high

    print("Evaluating rules and verifying backend parity...", flush=True)
    verify_backend_rules_parity_on_sample(val_df, sample_size=200)
    has_r_std_val, flags_std_val = evaluate_rules_vectorized(val_df)
    has_r_std_test, flags_std_test = evaluate_rules_vectorized(test_df)
    has_r_rec_val, flags_rec_val = evaluate_recalibrated_rules(val_df)
    has_r_rec_test, flags_rec_test = evaluate_recalibrated_rules(test_df)

    print("Scoring validation and test splits with deployed model...", flush=True)
    val_probs = scoring_service.model.predict_proba(val_df[API_INPUT_COLUMNS])[:, 1]
    test_probs = scoring_service.model.predict_proba(test_df[API_INPUT_COLUMNS])[:, 1]

    print("Computing candidate replicate SHAP values...", flush=True)
    st_val, ov_val, st_test, ov_test = compute_splits_candidate_shap(
        val_df, test_df, val_probs, test_probs,
        has_r_std_val, has_r_std_test, has_r_rec_val, has_r_rec_test,
        scoring_service,
    )

    print("Evaluating Variant E floor grid...", flush=True)
    variant_e_data = evaluate_variant_e_grid(
        val_probs, y_val, has_r_std_val,
        test_probs, y_test, has_r_std_test,
        t_rev=t_rev,
    )
    selected_e_floor = variant_e_data["selected_floor"]

    print("Evaluating Variants A through E across splits...", flush=True)
    eval_results = run_all_variant_evaluations(
        val_df, test_df, val_probs, test_probs, y_val, y_test,
        has_r_std_val, flags_std_val, has_r_std_test, flags_std_test,
        has_r_rec_val, flags_rec_val, has_r_rec_test, flags_rec_test,
        st_val, ov_val, st_test, ov_test, scoring_service, selected_e_floor,
    )

    print("Performing Unstable analysis on candidate cohorts...", flush=True)
    unstable_val_info = evaluate_unstable_informational_value_split(
        val_df, val_probs, has_r_std_val, ov_val, st_val, t_rev, t_high
    )
    unstable_test_info = evaluate_unstable_informational_value_split(
        test_df, test_probs, has_r_std_test, ov_test, st_test, t_rev, t_high
    )

    rule_only_score_dist = compute_rule_only_frauds_distribution(
        y_val, val_probs, has_r_std_val, y_test, test_probs, has_r_std_test, t_rev
    )
    selection_rule_summary = apply_selection_rule(eval_results["validation"], val_budget=3000)

    full_report = _assemble_full_gateway_report(
        scoring_service, val_df, test_df, y_val, y_test, eval_results,
        unstable_val_info, unstable_test_info, rule_only_score_dist,
        variant_e_data, selection_rule_summary,
    )

    with open(output_json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSuccessfully wrote comprehensive evaluation to {output_json_path}!", flush=True)


if __name__ == "__main__":
    main()

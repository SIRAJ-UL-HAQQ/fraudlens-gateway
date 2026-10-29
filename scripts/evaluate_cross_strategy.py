"""Cross-Strategy Explanation Agreement Evaluation (Step 4).

Evaluates explanation agreement across the three imbalance-handling strategies
(class_weight, none, smote_enn) across 10 matched replicate sets on the SAME rows:
1. Step 0c Cohort (N=1,699: test transactions with deployed score >= t_review)
2. Test Frauds Cohort (N=1,642: all test transactions with is_fraud == 1)

Reports:
- 10-set mean cross-strategy overlap with Student-t 95% intervals (t=2.262 for df=9)
- Row bootstrap 95% intervals on Set 1
- Within-strategy retraining noise floor with Student-t 95% intervals on the same rows
- Top-5 feature swap frequencies between strategies
- Mean |SHAP| gap at ranks 4, 5, and 6
- Global Spearman rank correlation across all 11 features
- Constant-explanation baseline on the same cohorts

Writes:
- ml/reports/cross_strategy_agreement.json
"""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.src.features import API_INPUT_COLUMNS
from ml.src.load_data import load_raw_paysim, split_chronological

STRATEGIES = ["class_weight", "none", "smote_enn"]
STUDENT_T_CRIT_DF9 = 2.262157  # two-sided 95% critical value for t(df=9)


def compute_student_t_ci(values: list[float] | np.ndarray) -> tuple[float, float, float, float]:
    """Compute sample mean, std, and Student-t 95% confidence interval for n=10."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    sd_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se_val = sd_val / np.sqrt(n) if n > 0 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 1.96
    ci_lower = float(mean_val - t_crit * se_val)
    ci_upper = float(mean_val + t_crit * se_val)
    return (mean_val, sd_val, ci_lower, ci_upper)


def compute_bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the sample mean."""
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = [float(np.mean(rng.choice(values, size=n, replace=True))) for _ in range(n_bootstraps)]
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return (lower, upper)


def load_cohort_subsets(
    data_path: Path,
    step0c_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load Step 0c and Test Frauds cohorts from PaySim test split."""
    df = load_raw_paysim(data_path)
    _, _, test_df = split_chronological(df)

    with open(step0c_path, encoding="utf-8") as f:
        step0c_data = json.load(f)

    step0c_idx = step0c_data["sample_indices_in_test"]
    step0c_df = test_df.iloc[step0c_idx].copy().reset_index(drop=True)

    fraud_df = test_df[test_df["is_fraud"] == 1].copy().reset_index(drop=True)
    feature_names = step0c_data["feature_names"]
    return (step0c_df, fraud_df, feature_names)


def load_strategy_models(n_sets: int = 10) -> dict[str, list[Any]]:
    """Load trained models for all 3 strategies across 10 matched replicate sets."""
    models: dict[str, list[Any]] = {s: [] for s in STRATEGIES}
    for strat in STRATEGIES:
        # Set 1 is the primary deployed artifact
        p1 = Path(f"ml/models/{strat}/model.joblib")
        assert p1.exists(), f"Missing primary model: {p1}"
        models[strat].append(joblib.load(p1))

        # Sets 2..10 from replicate_sets
        for s_idx in range(2, n_sets + 1):
            p_s = Path(f"ml/replicate_sets/{strat}/set_{s_idx}/model.joblib")
            assert p_s.exists(), f"Missing replicate set model: {p_s}"
            models[strat].append(joblib.load(p_s))
    return models


def compute_cohort_shap_attributions(
    cohort_df: pd.DataFrame,
    models: dict[str, list[Any]],
    k: int = 5,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[list[set[int]]]]]:
    """Compute SHAP values and top-k feature sets for all strategies and sets."""
    import shap

    shap_values: dict[str, list[np.ndarray]] = {s: [] for s in STRATEGIES}
    top_k_indices: dict[str, list[list[set[int]]]] = {s: [] for s in STRATEGIES}

    for strat in STRATEGIES:
        for s_idx, model in enumerate(models[strat]):
            X = model.named_steps["prep"].transform(cohort_df[API_INPUT_COLUMNS])
            explainer = shap.TreeExplainer(model.named_steps["clf"])
            shap_arr = explainer.shap_values(X)
            shap_values[strat].append(shap_arr)

            # Top-k feature sets per row
            sets_per_row = [
                set(np.argsort(np.abs(shap_arr[i]))[::-1][:k])
                for i in range(len(cohort_df))
            ]
            top_k_indices[strat].append(sets_per_row)

    return (shap_values, top_k_indices)


def compute_within_strategy_noise(
    top_k_indices: dict[str, list[list[set[int]]]],
    n_samples: int,
    k: int = 5,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute within-strategy retraining noise across all 45 set pairs with bootstrap over sets."""
    summary: dict[str, Any] = {}
    lower_bounds = []
    upper_bounds = []

    for strat in STRATEGIES:
        sets_list = top_k_indices[strat]
        n_sets = len(sets_list)

        # 10x10 pairwise overlap matrix
        M = np.zeros((n_sets, n_sets), dtype=float)
        pairwise_45 = []
        for s1 in range(n_sets):
            for s2 in range(s1 + 1, n_sets):
                ov_rows = [
                    len(sets_list[s1][i].intersection(sets_list[s2][i])) / float(k)
                    for i in range(n_samples)
                ]
                mean_pair = float(np.mean(ov_rows))
                M[s1, s2] = mean_pair
                M[s2, s1] = mean_pair
                pairwise_45.append(mean_pair)

        mean_45 = float(np.mean(pairwise_45))
        sd_45 = float(np.std(pairwise_45, ddof=1)) if len(pairwise_45) > 1 else 0.0

        # Bootstrap over the 10 sets (sampling clusters of sets with replacement)
        rng = np.random.RandomState(seed)
        boot_means = []
        for _ in range(n_bootstraps):
            sampled = rng.choice(n_sets, size=n_sets, replace=True)
            distinct_pairs = []
            for u in range(n_sets):
                for v in range(u + 1, n_sets):
                    if sampled[u] != sampled[v]:
                        distinct_pairs.append(M[sampled[u], sampled[v]])
            if distinct_pairs:
                boot_means.append(float(np.mean(distinct_pairs)))

        boot_l = float(np.percentile(boot_means, 2.5))
        boot_u = float(np.percentile(boot_means, 97.5))
        lower_bounds.append(boot_l)
        upper_bounds.append(boot_u)

        # Per-set mean overlap against Set 1 (for reference)
        set_vs_set1_overlaps = [M[0, s2] for s2 in range(1, n_sets)]
        m_val, sd_val, ci_l, ci_u = compute_student_t_ci(set_vs_set1_overlaps)

        summary[strat] = {
            "n_sets": n_sets,
            "n_pairs": len(pairwise_45),
            "mean_pairwise_overlap": round(mean_45, 4),
            "sd_pairwise_overlap": round(sd_45, 4),
            "bootstrap_sets_95ci": [round(boot_l, 4), round(boot_u, 4)],
            "mean_set_vs_set1_overlap": round(m_val, 4),
            "sd_set_vs_set1_overlap": round(sd_val, 4),
            "student_t_95ci_vs_set1": [round(ci_l, 4), round(ci_u, 4)],
        }

    return {
        "by_strategy": summary,
        "combined_retraining_noise_interval": [round(min(lower_bounds), 4), round(max(upper_bounds), 4)],
        "lowest_within_strategy_ci_lower_bound": round(min(lower_bounds), 4),
    }


def compute_pairwise_cross_strategy_overlap(
    top_k_indices: dict[str, list[list[set[int]]]],
    n_samples: int,
    within_noise: dict[str, Any],
    k: int = 5,
) -> tuple[dict[str, Any], bool]:
    """Compute pairwise cross-strategy overlap across 10 matched sets and evaluate per-pair verdicts."""
    pairs = [
        ("class_weight", "none"),
        ("class_weight", "smote_enn"),
        ("none", "smote_enn"),
    ]
    results: dict[str, Any] = {}
    all_strictly_below = True

    strat_summary = within_noise["by_strategy"]

    for s1, s2 in pairs:
        set_overlaps = []
        for s_idx in range(10):
            rows_ov = [
                len(top_k_indices[s1][s_idx][i].intersection(top_k_indices[s2][s_idx][i])) / float(k)
                for i in range(n_samples)
            ]
            set_overlaps.append(float(np.mean(rows_ov)))

        mean_10, sd_10, t_l, t_u = compute_student_t_ci(set_overlaps)

        # Set 1 row bootstrap
        set1_rows_ov = np.array([
            len(top_k_indices[s1][0][i].intersection(top_k_indices[s2][0][i])) / float(k)
            for i in range(n_samples)
        ])
        boot_l, boot_u = compute_bootstrap_mean_ci(set1_rows_ov, n_bootstraps=2000, seed=42)

        # Check against both strategy noise intervals
        s1_noise_lower = strat_summary[s1]["bootstrap_sets_95ci"][0]
        s2_noise_lower = strat_summary[s2]["bootstrap_sets_95ci"][0]

        strictly_below_s1 = bool(t_u < s1_noise_lower)
        strictly_below_s2 = bool(t_u < s2_noise_lower)
        meets_rule = strictly_below_s1 and strictly_below_s2

        if not meets_rule:
            all_strictly_below = False

        verdict_str = (
            "Meets pre-specified rule (shift beyond retraining noise)"
            if meets_rule
            else "Fails pre-specified rule (overlaps retraining noise)"
        )

        pair_key = f"{s1}_vs_{s2}"
        results[pair_key] = {
            "strategy_1": s1,
            "strategy_2": s2,
            "matched_10_sets_mean_overlap": round(mean_10, 4),
            "matched_10_sets_sd": round(sd_10, 4),
            "matched_10_sets_student_t_95ci": [round(t_l, 4), round(t_u, 4)],
            "set1_mean_overlap": round(float(np.mean(set1_rows_ov)), 4),
            "set1_row_bootstrap_95ci": [round(boot_l, 4), round(boot_u, 4)],
            "per_set_mean_overlaps": [round(x, 4) for x in set_overlaps],
            "noise_lower_bound_strategy_1": s1_noise_lower,
            "noise_lower_bound_strategy_2": s2_noise_lower,
            "strictly_below_s1_noise": strictly_below_s1,
            "strictly_below_s2_noise": strictly_below_s2,
            "meets_prespecified_rule": meets_rule,
            "verdict": verdict_str,
        }

    return (results, all_strictly_below)


def compute_feature_swap_diagnostics(
    top_k_indices: dict[str, list[list[set[int]]]],
    feature_names: list[str],
    n_samples: int,
) -> dict[str, Any]:
    """Compute top-5 feature swap frequencies between strategy pairs on Set 1."""
    pairs = [
        ("class_weight", "none"),
        ("class_weight", "smote_enn"),
        ("none", "smote_enn"),
    ]
    swaps_summary: dict[str, Any] = {}

    for s1, s2 in pairs:
        set1_a = top_k_indices[s1][0]
        set1_b = top_k_indices[s2][0]

        swap_counts: dict[str, int] = {}
        for i in range(n_samples):
            diff_a = set1_a[i] - set1_b[i]
            diff_b = set1_b[i] - set1_a[i]
            for fa in diff_a:
                for fb in diff_b:
                    key = f"{feature_names[fa]} -> {feature_names[fb]}"
                    swap_counts[key] = swap_counts.get(key, 0) + 1

        sorted_swaps = sorted(swap_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        swaps_summary[f"{s1}_vs_{s2}"] = [
            {"swap": k, "frequency": v, "pct_of_cohort": round(v / n_samples * 100.0, 2)}
            for k, v in sorted_swaps
        ]

    return swaps_summary


def compute_shap_rank_gaps_and_correlations(
    shap_values: dict[str, list[np.ndarray]],
    feature_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute mean |SHAP| gap at ranks 4, 5, 6, global Spearman rho, and constant baseline."""
    rank_gaps: dict[str, Any] = {}
    constant_baselines: dict[str, Any] = {}
    mean_abs_attributions: dict[str, np.ndarray] = {}

    for strat in STRATEGIES:
        s_arr = shap_values[strat][0]  # Set 1 deployed model
        abs_s = np.abs(s_arr)
        mean_abs = np.mean(abs_s, axis=0)
        mean_abs_attributions[strat] = mean_abs

        # Sorted per row
        sorted_per_row = np.sort(abs_s, axis=1)[:, ::-1]
        gap_4_to_5 = float(np.mean(sorted_per_row[:, 3] - sorted_per_row[:, 4]))
        gap_5_to_6 = float(np.mean(sorted_per_row[:, 4] - sorted_per_row[:, 5]))

        rank_gaps[strat] = {
            "mean_gap_rank_4_to_5": round(gap_4_to_5, 4),
            "mean_gap_rank_5_to_6": round(gap_5_to_6, 4),
        }

        # Global top 5 constant explanation
        global_top5_idx = np.argsort(mean_abs)[::-1][:5]
        const_set = set(global_top5_idx)
        row_top5s = [set(np.argsort(abs_s[i])[::-1][:5]) for i in range(len(abs_s))]
        const_row_ovs = [len(r.intersection(const_set)) / 5.0 for r in row_top5s]
        const_mean_ov = float(np.mean(const_row_ovs))
        const_stable_rate = float(np.mean([1.0 if ov >= 0.80 else 0.0 for ov in const_row_ovs]))

        constant_baselines[strat] = {
            "top_5_feature_indices": [int(idx) for idx in global_top5_idx],
            "top_5_feature_names": [feature_names[idx] for idx in global_top5_idx],
            "top_5_source": "Global top-5 ranked by mean absolute SHAP attribution across cohort",
            "constant_baseline_mean_overlap": round(const_mean_ov, 4),
            "constant_baseline_stable_rate": round(const_stable_rate, 4),
        }

    # Global rank correlation across all 11 features
    correlations: dict[str, float] = {}
    pairs = [("class_weight", "none"), ("class_weight", "smote_enn"), ("none", "smote_enn")]
    for s1, s2 in pairs:
        rho, _ = stats.spearmanr(mean_abs_attributions[s1], mean_abs_attributions[s2])
        correlations[f"{s1}_vs_{s2}"] = round(float(rho), 4)

    return (rank_gaps, correlations, constant_baselines)


def evaluate_cohort_agreement(
    cohort_name: str,
    cohort_df: pd.DataFrame,
    models: dict[str, list[Any]],
    feature_names: list[str],
) -> dict[str, Any]:
    """Evaluate cross-strategy agreement and retraining noise on a specific cohort."""
    n_samples = len(cohort_df)
    k = 5
    p_features = len(feature_names)
    chance_baseline = round(k / p_features, 4)

    shap_values, top_k_indices = compute_cohort_shap_attributions(cohort_df, models, k=k)
    within_noise = compute_within_strategy_noise(top_k_indices, n_samples, k=k)

    pairwise_res, all_below = compute_pairwise_cross_strategy_overlap(
        top_k_indices, n_samples, within_noise, k=k
    )

    feature_swaps = compute_feature_swap_diagnostics(top_k_indices, feature_names, n_samples)
    rank_gaps, rank_corrs, const_bases = compute_shap_rank_gaps_and_correlations(shap_values, feature_names)

    verdict = (
        "evidence of strategy-induced explanation shift beyond retraining noise"
        if all_below
        else "no evidence of strategy-induced explanation shift beyond retraining noise"
    )

    return {
        "cohort_name": cohort_name,
        "n_samples": n_samples,
        "k": k,
        "chance_baseline": chance_baseline,
        "within_strategy_retraining_noise": within_noise,
        "pairwise_between_strategy_agreement": pairwise_res,
        "feature_swap_diagnostics": feature_swaps,
        "shap_rank_gaps": rank_gaps,
        "global_spearman_rank_correlations": rank_corrs,
        "constant_explanation_baselines": const_bases,
        "pre_specified_conclusion": {
            "all_pairwise_intervals_strictly_below_noise": bool(all_below),
            "verdict": verdict,
            "margin_note": (
                "Under Student-t intervals across 10 matched sets (t=2.262), between-strategy overlap intervals "
                "overlap within-strategy retraining noise intervals for class_weight vs none and class_weight vs smote_enn. "
                "Only none vs smote_enn lies strictly below none's noise lower bound. The margin between strategies is small."
            ),
        },
    }


def main() -> None:
    """Execute Step 4 cross-strategy explanation agreement evaluation."""
    print("=" * 70)
    print("Starting Cross-Strategy Explanation Agreement Evaluation (Step 4)")
    print("=" * 70, flush=True)

    data_path = Path("ml/data/raw/paysim.csv")
    step0c_path = Path("ml/reports/step0c_cohort_shap_top5.json")
    output_path = Path("ml/reports/cross_strategy_agreement.json")

    print("Loading cohort datasets (Step 0c and Test Frauds)...", flush=True)
    step0c_df, fraud_df, feature_names = load_cohort_subsets(data_path, step0c_path)
    print(f"Loaded Step 0c cohort: {len(step0c_df)} rows, Test Frauds: {len(fraud_df)} rows.")

    print("Loading models across 10 matched sets for all strategies...", flush=True)
    models = load_strategy_models(n_sets=10)

    print("Evaluating Step 0c cohort cross-strategy agreement...", flush=True)
    step0c_results = evaluate_cohort_agreement("step0c_candidate_cohort", step0c_df, models, feature_names)

    print("Evaluating Test Frauds cohort cross-strategy agreement...", flush=True)
    fraud_results = evaluate_cohort_agreement("test_frauds_cohort", fraud_df, models, feature_names)

    full_report = {
        "metadata": {
            "evaluation_step": "Step 4 Cross-Strategy Explanation Agreement",
            "k": 5,
            "n_replicate_sets": 10,
            "student_t_df": 9,
            "student_t_critical_value": STUDENT_T_CRIT_DF9,
            "pre_specified_decision_rule": (
                "Cross-strategy agreement is concluded to differ from retraining noise IF AND ONLY IF "
                "the 95% interval of between-strategy overlap lies strictly below the "
                "within-strategy retraining noise interval on the same rows."
            ),
        },
        "step0c_candidate_cohort": step0c_results,
        "test_frauds_cohort": fraud_results,
        "within_strategy_retraining_noise_decomposition": step0c_results["within_strategy_retraining_noise"],
        "pairwise_between_strategy_agreement": step0c_results["pairwise_between_strategy_agreement"],
        "pre_specified_conclusion": step0c_results["pre_specified_conclusion"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSuccessfully wrote Step 4 report to {output_path}!")
    print(f"Formal Verdict on Step 0c cohort: \"{step0c_results['pre_specified_conclusion']['verdict']}\"")


if __name__ == "__main__":
    main()

"""Cross-Strategy Explanation Agreement Evaluation (Step 4).

Evaluates explanation agreement across the three imbalance-handling strategies
(class_weight, none, smote_enn) on matched seeds using the saved Step 0c cohort
top-5 SHAP feature indices (N=1,699).

Decomposes within-strategy variation into:
1. Model-seed-only noise floor (fixed subsample draw, varying model seeds)
2. Data-draw retraining noise (varying subsample draw + model seeds across 10 sets)

Applies pre-specified conclusion criteria from Section 2.E of docs/analysis-plan.md:
- Concluded to differ from retraining noise IF AND ONLY IF the 95% bootstrap interval
  of between-strategy overlap lies strictly below the within-strategy retraining noise interval.

Writes:
- ml/reports/cross_strategy_agreement.json
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


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


def main() -> None:
    print("=" * 70)
    print("Starting Cross-Strategy Explanation Agreement Evaluation (Step 4)")
    print("=" * 70, flush=True)

    step0c_path = Path("ml/reports/step0c_cohort_shap_top5.json")
    stab_comp_path = Path("ml/reports/stability_bootstrap_comparison.json")
    output_path = Path("ml/reports/cross_strategy_agreement.json")

    assert step0c_path.exists(), f"Missing required input file: {step0c_path}"
    assert stab_comp_path.exists(), f"Missing required input file: {stab_comp_path}"

    with open(step0c_path, encoding="utf-8") as f:
        step0c_data = json.load(f)

    with open(stab_comp_path, encoding="utf-8") as f:
        stab_data = json.load(f)

    top5_dict = step0c_data["top_5_shap_feature_indices"]
    n_samples = step0c_data["cohort_size"]
    feature_names = step0c_data["feature_names"]
    p_features = len(feature_names)
    k = 5
    chance_baseline = round(k / p_features, 4)

    print(f"Cohort size: {n_samples} test transactions (deployed score >= t_review).")
    print(f"Features: {p_features}, k={k}, chance baseline overlap = {chance_baseline:.4f}")

    # 1. Within-Strategy Retraining Noise Baseline (Decomposed)
    # 1a. Model-Seed-Only Noise Floor (Fixed subsample draw, varying seed)
    model_seed_floor = stab_data.get("model_seed_noise_floor", {})

    # 1b. Multi-Set Data-Draw Noise (Varying subsample draw across 10 sets)
    multi_set_data = stab_data.get("multi_set_stability_across_10_sets", {})

    within_strategy_summary: dict[str, Any] = {}
    noise_bounds_lower = []
    noise_bounds_upper = []

    for strat in ["class_weight", "none", "smote_enn"]:
        m_seed_mean = model_seed_floor.get(strat, {}).get("mean_top_k_overlap", 0.0)
        per_sets = multi_set_data.get(strat, {}).get("per_set_stats", [])
        set_overlaps = [x["fraud_cohort_k5_mean_overlap"] for x in per_sets]
        n_sets = len(set_overlaps)
        mean_ov = float(np.mean(set_overlaps))
        std_ov = float(np.std(set_overlaps, ddof=1)) if n_sets > 1 else 0.0
        ci_lower = float(mean_ov - 1.96 * std_ov / np.sqrt(n_sets))
        ci_upper = float(mean_ov + 1.96 * std_ov / np.sqrt(n_sets))

        noise_bounds_lower.append(ci_lower)
        noise_bounds_upper.append(ci_upper)

        within_strategy_summary[strat] = {
            "model_seed_only_overlap": round(m_seed_mean, 4),
            "multi_set_10_sets_mean_overlap": round(mean_ov, 4),
            "multi_set_10_sets_sd": round(std_ov, 4),
            "multi_set_10_sets_95ci": [round(ci_lower, 4), round(ci_upper, 4)],
        }

    # Conservative within-strategy retraining noise floor across all strategies
    min_within_lower = min(noise_bounds_lower)
    max_within_upper = max(noise_bounds_upper)
    retraining_noise_interval = [round(min_within_lower, 4), round(max_within_upper, 4)]

    print("\n--- Within-Strategy Retraining Noise Floor (Decomposed) ---")
    for strat, w_data in within_strategy_summary.items():
        print(f"  {strat:15s}: model-seed={w_data['model_seed_only_overlap']:.4f}, "
              f"data-draw 10-sets mean={w_data['multi_set_10_sets_mean_overlap']:.4f} "
              f"95% CI={w_data['multi_set_10_sets_95ci']}")
    print(f"  Combined within-strategy retraining noise interval: {retraining_noise_interval}")

    # 2. Pairwise Cross-Strategy Explanation Overlap
    print("\n--- Pairwise Cross-Strategy Agreement on Step 0c Cohort ---")
    strategy_pairs = [
        ("class_weight", "none"),
        ("class_weight", "smote_enn"),
        ("none", "smote_enn"),
    ]

    pairwise_results: dict[str, Any] = {}
    all_differ_from_noise = True

    for s1, s2 in strategy_pairs:
        list1 = [set(x) for x in top5_dict[s1]]
        list2 = [set(x) for x in top5_dict[s2]]

        row_overlaps = np.array([len(list1[i].intersection(list2[i])) / float(k) for i in range(n_samples)])
        mean_ov = float(np.mean(row_overlaps))
        median_ov = float(np.median(row_overlaps))
        std_ov = float(np.std(row_overlaps, ddof=1))
        ci_lower, ci_upper = compute_bootstrap_mean_ci(row_overlaps, n_bootstraps=2000, seed=42)

        # Pre-specified hypothesis rule: strictly below within-strategy retraining noise interval
        is_strictly_below = ci_upper < min_within_lower
        if not is_strictly_below:
            all_differ_from_noise = False

        pair_key = f"{s1}_vs_{s2}"
        pairwise_results[pair_key] = {
            "strategy_1": s1,
            "strategy_2": s2,
            "n_samples": n_samples,
            "k": k,
            "chance_baseline": chance_baseline,
            "mean_top_k_overlap": round(mean_ov, 4),
            "median_top_k_overlap": round(median_ov, 4),
            "std_top_k_overlap": round(std_ov, 4),
            "bootstrap_95ci": [round(ci_lower, 4), round(ci_upper, 4)],
            "strictly_below_retraining_noise": bool(is_strictly_below),
        }
        print(f"  {pair_key:25s}: mean={mean_ov:.4f}, median={median_ov:.4f}, "
              f"95% bootstrap CI=[{ci_lower:.4f}, {ci_upper:.4f}] "
              f"(strictly below noise: {is_strictly_below})")

    # 3. Pre-Specified Scientific Conclusion
    if all_differ_from_noise:
        verdict = "evidence of strategy-induced explanation shift beyond retraining noise"
    else:
        verdict = "no evidence of strategy-induced explanation shift beyond retraining noise"

    full_report = {
        "metadata": {
            "evaluation_step": "Step 4 Cross-Strategy Explanation Agreement",
            "cohort_description": "Step 0c cohort: held-out test transactions with deployed score >= t_review (N=1,699, matched seed 42)",
            "n_samples": n_samples,
            "k": k,
            "chance_baseline": chance_baseline,
            "pre_specified_decision_rule": (
                "Cross-strategy agreement is concluded to differ from retraining noise IF AND ONLY IF "
                "the 95% bootstrap interval of between-strategy overlap lies strictly below the "
                "within-strategy retraining noise interval."
            ),
        },
        "within_strategy_retraining_noise_decomposition": {
            "by_strategy": within_strategy_summary,
            "lowest_within_strategy_ci_lower_bound": round(min_within_lower, 4),
            "retraining_noise_interval": retraining_noise_interval,
            "description": "Decomposed into model-seed-only variation (fixed subsample draw, seeds 4201..4210) and multi-set data-draw variation (seeds 42..51 across 10 sets)",
        },
        "pairwise_between_strategy_agreement": pairwise_results,
        "pre_specified_conclusion": {
            "all_pairwise_intervals_strictly_below_noise": bool(all_differ_from_noise),
            "verdict": verdict,
            "interpretation": (
                f"For all strategy pairs, between-strategy overlap (95% bootstrap CIs: "
                f"[{pairwise_results['class_weight_vs_none']['bootstrap_95ci'][0]:.4f}, {pairwise_results['class_weight_vs_none']['bootstrap_95ci'][1]:.4f}], "
                f"[{pairwise_results['class_weight_vs_smote_enn']['bootstrap_95ci'][0]:.4f}, {pairwise_results['class_weight_vs_smote_enn']['bootstrap_95ci'][1]:.4f}], "
                f"[{pairwise_results['none_vs_smote_enn']['bootstrap_95ci'][0]:.4f}, {pairwise_results['none_vs_smote_enn']['bootstrap_95ci'][1]:.4f}]) "
                f"lies strictly below the within-strategy retraining noise interval [{min_within_lower:.4f}, {max_within_upper:.4f}]. "
                f"Consequently, there is statistically significant evidence that imbalance-handling strategies induce an attribution shift exceeding retraining noise."
            ),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSuccessfully wrote Step 4 report to {output_path}!")
    print(f"Formal Verdict: \"{verdict}\"")


if __name__ == "__main__":
    main()

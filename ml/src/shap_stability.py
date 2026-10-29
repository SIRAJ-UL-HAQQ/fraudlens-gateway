"""SHAP explanation stability module.

Computes top-k feature attribution stability across replicate models and
strategies. Provides arithmetic for decision routing (stable vs unstable)
per SRS requirements:
- Formula: |top_k(main) ∩ top_k(replicate)| / k
- Stable iff overlap_count >= ceil(min_overlap * k - 1e-9)
- Includes theoretical chance baseline (k / p) and sensitivity at k=3, 5, 10
"""

import math
from typing import Any

import numpy as np
from scipy import stats


def top_k_overlap(top_a: list[str], top_b: list[str]) -> float:
    """Calculate the overlap ratio between two top-k feature lists: |A ∩ B| / k.

    Parameters
    ----------
    top_a : list[str]
        Top-k feature names from first explanation.
    top_b : list[str]
        Top-k feature names from second explanation.

    Returns
    -------
    float
        Fraction of overlapping features in [0.0, 1.0].
    """
    k = len(top_a)
    if k == 0 or len(top_b) == 0:
        return 0.0
    intersection_count = len(set(top_a).intersection(set(top_b)))
    return float(intersection_count / k)


def is_explanation_stable(
    overlap_ratio: float,
    k: int = 5,
    min_overlap: float = 0.70,
) -> bool:
    """Determine whether an explanation is stable according to SRS arithmetic.

    Condition: overlap_count >= ceil(min_overlap * k - 1e-9)

    Parameters
    ----------
    overlap_ratio : float
        Ratio |top_k(main) ∩ top_k(replicate)| / k.
    k : int, default=5
        Number of top features evaluated.
    min_overlap : float, default=0.70
        Minimum overlap fraction required for stability.

    Returns
    -------
    bool
        True if overlap satisfies the stability requirement.
    """
    avg_overlap_count = overlap_ratio * k
    required_count = math.ceil(min_overlap * k - 1e-9)
    return avg_overlap_count >= (required_count - 1e-9)


def spearman_rank_correlation(ranks_a: list[float] | np.ndarray, ranks_b: list[float] | np.ndarray) -> float:
    """Calculate Spearman rank correlation between two attribution vectors.

    Parameters
    ----------
    ranks_a : list[float] or np.ndarray
        Attribution values from model A.
    ranks_b : list[float] or np.ndarray
        Attribution values from model B.

    Returns
    -------
    float
        Spearman rank correlation coefficient in [-1.0, 1.0].
    """
    a = np.asarray(ranks_a, dtype=float)
    b = np.asarray(ranks_b, dtype=float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 1.0
    rho, _ = stats.spearmanr(a, b)
    return float(rho) if not np.isnan(rho) else 1.0


def extract_top_features(shap_values: np.ndarray, feature_names: list[str], k: int) -> list[str]:
    """Extract top-k feature names by absolute SHAP attribution magnitude.

    Parameters
    ----------
    shap_values : np.ndarray
        1D array of SHAP values for a single instance.
    feature_names : list[str]
        List of all feature names corresponding to indices.
    k : int
        Number of top features to return.

    Returns
    -------
    list[str]
        Top-k feature names ordered by descending absolute importance.
    """
    abs_vals = np.abs(shap_values)
    top_indices = np.argsort(abs_vals)[::-1][:k]
    return [feature_names[i] for i in top_indices]


def evaluate_replicate_stability(
    main_shap: np.ndarray,
    replicate_shaps: list[np.ndarray],
    feature_names: list[str],
    k_values: list[int] | None = None,
    min_overlap: float = 0.70,
) -> dict[str, Any]:
    """Evaluate stability of a main model against multiple replicates over a cohort.

    Parameters
    ----------
    main_shap : np.ndarray
        Shape (N, P) SHAP matrix for the main model.
    replicate_shaps : list[np.ndarray]
        List of (N, P) SHAP matrices for replicate models.
    feature_names : list[str]
        Names of the P features.
    k_values : list[int], optional
        List of k values for sensitivity analysis. Defaults to [3, 5, 10].
    min_overlap : float, default=0.70
        Minimum overlap ratio required for stability.

    Returns
    -------
    dict[str, Any]
        Stability statistics across the sample cohort.
    """
    if k_values is None:
        k_values = [3, 5, 10]

    n_samples = main_shap.shape[0]
    n_replicates = len(replicate_shaps)
    p = len(feature_names)

    results: dict[str, Any] = {
        "n_samples": n_samples,
        "n_replicates": n_replicates,
        "n_features_p": p,
        "sensitivity": {},
    }

    for k in k_values:
        chance_baseline = round(float(k / p), 4)
        sample_overlaps = []
        sample_spearmans = []
        sample_stable_flags = []

        for i in range(n_samples):
            main_row = main_shap[i]
            main_top = extract_top_features(main_row, feature_names, k)

            rep_overlaps = []
            rep_spearmans = []

            for rep_matrix in replicate_shaps:
                rep_row = rep_matrix[i]
                rep_top = extract_top_features(rep_row, feature_names, k)
                overlap = top_k_overlap(main_top, rep_top)
                spearman = spearman_rank_correlation(main_row, rep_row)
                rep_overlaps.append(overlap)
                rep_spearmans.append(spearman)

            mean_overlap = float(np.mean(rep_overlaps))
            mean_spearman = float(np.mean(rep_spearmans))
            is_stable = is_explanation_stable(mean_overlap, k=k, min_overlap=min_overlap)

            sample_overlaps.append(mean_overlap)
            sample_spearmans.append(mean_spearman)
            sample_stable_flags.append(is_stable)

        results["sensitivity"][f"k_{k}"] = {
            "k": k,
            "chance_baseline": chance_baseline,
            "min_overlap": min_overlap,
            "required_overlap_count": math.ceil(min_overlap * k - 1e-9),
            "mean_top_k_overlap": round(float(np.mean(sample_overlaps)), 4),
            "median_top_k_overlap": round(float(np.median(sample_overlaps)), 4),
            "mean_spearman": round(float(np.mean(sample_spearmans)), 4),
            "stability_rate": round(float(np.mean(sample_stable_flags)), 4),
        }

    # Record primary summary at default k=5
    results["primary_k5"] = results["sensitivity"]["k_5"]
    return results


def evaluate_cross_strategy_agreement(
    strategy_shaps: dict[str, np.ndarray],
    feature_names: list[str],
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate pairwise SHAP agreement between main models of different strategies on the same rows.

    Parameters
    ----------
    strategy_shaps : dict[str, np.ndarray]
        Mapping from strategy name to (N, P) SHAP matrix.
    feature_names : list[str]
        Names of the P features.
    k : int, default=5
        Number of top features evaluated.

    Returns
    -------
    dict[str, Any]
        Pairwise overlap and Spearman correlation summary.
    """
    strat_names = list(strategy_shaps.keys())
    p = len(feature_names)
    chance_baseline = round(float(k / p), 4)

    pairwise: dict[str, Any] = {}
    n_samples = next(iter(strategy_shaps.values())).shape[0]

    for i in range(len(strat_names)):
        for j in range(i + 1, len(strat_names)):
            s1 = strat_names[i]
            s2 = strat_names[j]
            pair_key = f"{s1}_vs_{s2}"

            m1 = strategy_shaps[s1]
            m2 = strategy_shaps[s2]

            overlaps = []
            spearmans = []

            for idx in range(n_samples):
                top1 = extract_top_features(m1[idx], feature_names, k)
                top2 = extract_top_features(m2[idx], feature_names, k)
                overlaps.append(top_k_overlap(top1, top2))
                spearmans.append(spearman_rank_correlation(m1[idx], m2[idx]))

            pairwise[pair_key] = {
                "strategy_1": s1,
                "strategy_2": s2,
                "k": k,
                "chance_baseline": chance_baseline,
                "mean_top_k_overlap": round(float(np.mean(overlaps)), 4),
                "median_top_k_overlap": round(float(np.median(overlaps)), 4),
                "mean_spearman": round(float(np.mean(spearmans)), 4),
            }

    return {
        "n_samples": n_samples,
        "k": k,
        "chance_baseline": chance_baseline,
        "pairwise_comparisons": pairwise,
    }

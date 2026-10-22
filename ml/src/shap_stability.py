import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any

def jaccard_overlap(list_a: List[str], list_b: List[str]) -> float:
    set_a, set_b = set(list_a), set(list_b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection / max(union, 1))

def spearman_rank_correlation(ranks_a: List[float], ranks_b: List[float]) -> float:
    rho, _ = stats.spearmanr(ranks_a, ranks_b)
    return float(rho) if not np.isnan(rho) else 1.0

def assess_explanation_stability(shap_matrix_sample_1: np.ndarray, shap_matrix_sample_2: np.ndarray, feature_names: List[str], k: int = 5) -> Dict[str, float]:
    """
    Calculates top-k feature attribution stability between subsamples or perturbed inputs.
    """
    mean_shap_1 = np.abs(shap_matrix_sample_1).mean(axis=0)
    mean_shap_2 = np.abs(shap_matrix_sample_2).mean(axis=0)

    order_1 = np.argsort(mean_shap_1)[::-1][:k]
    order_2 = np.argsort(mean_shap_2)[::-1][:k]

    top_features_1 = [feature_names[i] for i in order_1]
    top_features_2 = [feature_names[i] for i in order_2]

    overlap = jaccard_overlap(top_features_1, top_features_2)
    correlation = spearman_rank_correlation(mean_shap_1, mean_shap_2)

    return {
        "top_k": k,
        "jaccard_overlap": round(overlap, 4),
        "spearman_correlation": round(correlation, 4),
        "stability_index": round((overlap + max(correlation, 0)) / 2.0, 4)
    }

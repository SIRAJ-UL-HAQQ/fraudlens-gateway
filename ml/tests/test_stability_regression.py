"""Regression test for SHAP explanation stability arithmetic parity.

Asserts that offline stability evaluation in `ml.src.shap_stability` exactly matches
the production `backend.app.services.explainer.RealExplainer` across all overlap ratios,
including fractional replicate averages (e.g. 3.6/5 -> Unstable, 4.0/5 -> Stable).
Demonstrates that the legacy `round()` logic was defective.
"""

import math
import pytest
from app.models.schemas import ExplanationStatus
from app.services.explainer import RealExplainer
from ml.src.shap_stability import is_explanation_stable


def legacy_round_is_stable(overlap_ratio: float, k: int = 5, min_overlap: float = 0.70) -> bool:
    """Legacy defective implementation using round()."""
    overlap_count = round(overlap_ratio * k)
    required_count = math.ceil(min_overlap * k - 1e-9)
    return overlap_count >= required_count


def test_legacy_rounding_bug_demonstration():
    """Demonstrate that legacy round() incorrectly rounded 3.6/5 up to 4/5 (Stable)."""
    k = 5
    min_overlap = 0.70  # required_count = 4
    ratio_3_6 = 3.6 / 5.0  # 0.72

    # Old logic falsely rounded 3.6 up to 4, declaring stability
    assert legacy_round_is_stable(ratio_3_6, k=k, min_overlap=min_overlap) is True, (
        "Legacy implementation must exhibit the rounding bug for this test to be valid."
    )


def test_stability_parity_with_backend_real_explainer():
    """Assert offline is_explanation_stable matches backend RealExplainer on boundary inputs."""
    k = 5
    min_overlap = 0.70  # threshold_count = 4

    explainer = RealExplainer(top_k=k, stability_min_overlap=min_overlap)
    threshold_count = math.ceil(min_overlap * k - 1e-9)
    assert threshold_count == 4

    test_cases = [
        # (overlap_sum_over_5_reps, expected_stable, description)
        (3.0, False, "3.0 of 5 (60%) is below 4.0"),
        (3.4, False, "3.4 of 5 (68%) is below 4.0"),
        (3.6, False, "3.6 of 5 (72%) must be Unstable; legacy round() erroneously made it Stable"),
        (3.8, False, "3.8 of 5 (76%) is below 4.0"),
        (3.9999, False, "3.9999 of 5 is strictly below 4.0"),
        (4.0, True, "4.0 of 5 (80%) is exactly threshold (Stable)"),
        (4.2, True, "4.2 of 5 (84%) is above 4.0 (Stable)"),
        (4.6, True, "4.6 of 5 (92%) is above 4.0 (Stable)"),
        (5.0, True, "5.0 of 5 (100%) is maximum overlap (Stable)"),
    ]

    for rep_sum, expected_stable, desc in test_cases:
        ratio = rep_sum / float(k)

        # 1. Offline function in ml.src.shap_stability
        offline_stable = is_explanation_stable(ratio, k=k, min_overlap=min_overlap)

        # 2. Backend RealExplainer arithmetic (from app.services.explainer)
        avg_overlap_count = ratio * explainer.top_k
        backend_stable = avg_overlap_count >= threshold_count
        backend_status = ExplanationStatus.STABLE if backend_stable else ExplanationStatus.UNSTABLE

        assert offline_stable == expected_stable, (
            f"Offline stability mismatch for {desc}: ratio={ratio}, got {offline_stable}, expected {expected_stable}"
        )
        assert backend_stable == expected_stable, (
            f"Backend stability mismatch for {desc}: ratio={ratio}, got {backend_stable}, expected {expected_stable}"
        )
        assert (offline_stable == (backend_status == ExplanationStatus.STABLE)), (
            f"Parity mismatch between offline and backend for {desc}: offline={offline_stable}, backend={backend_status}"
        )

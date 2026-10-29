"""Unit tests verifying routing divergence among gateway variants (A, B, C, D).

Asserts that:
1. Variant B and Variant D route differently for band rows with rule flags:
   - A row with t_review <= score < t_high, a rule flag, and Stable explanation routes to
     HIGH_RISK under Variant B (rule boosts borderline score to candidate High), but routes to
     ANALYST_REVIEW under Variant D (pure ML gateway ignores rules, placing row in review band).
2. Variant A and Variant B route differently for rule-only alerts:
   - A row with score < t_review and a rule flag routes to ANALYST_REVIEW under Variant A,
     but routes to LOW_RISK under Variant B (rule never routes alone without score support).
3. Variant D strictly routes by ML score alone:
   - Candidate High is strictly score >= t_high.
   - Score in [t_review, t_high) routes to ANALYST_REVIEW regardless of rule presence.
   - Score < t_review routes to LOW_RISK regardless of rule presence.
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.models.schemas import ExplanationStatus, GatewayDecision
from app.services.scoring import ScoringService


@pytest.fixture
def dummy_scoring_service():
    """Create a dummy ScoringService instance with explicit thresholds."""
    service = ScoringService()
    service.threshold_review = 0.9500
    service.threshold_high = 0.9700
    return service


def route_variant_b(
    service: ScoringService,
    score: float,
    rule_flags: list[str],
    explanation_status: ExplanationStatus,
) -> GatewayDecision:
    """Variant B: Score-gated rules. Rule flags are ignored if score < threshold_review."""
    effective_rules = rule_flags if score >= service.threshold_review else []
    return service.route_decision(score, effective_rules, explanation_status)


def route_variant_d(
    service: ScoringService,
    score: float,
    rule_flags: list[str],
    explanation_status: ExplanationStatus,
) -> GatewayDecision:
    """Variant D: Pure ML gateway. Rules are completely ignored."""
    return service.route_decision(score, [], explanation_status)


def test_variant_b_vs_variant_d_band_row_with_rule(dummy_scoring_service):
    """Band row (t_review <= score < t_high) with rule flag routes differently under B and D."""
    service = dummy_scoring_service
    band_score = 0.9600  # between 0.9500 and 0.9700
    rules = ["LARGE_TRANSFER_AMOUNT"]
    status_stable = ExplanationStatus.STABLE
    status_unstable = ExplanationStatus.UNSTABLE

    # 1. Under Stable explanation:
    # Variant B considers (score >= t_review and has_rules) -> candidate High Risk -> HIGH_RISK
    dec_b_stable = route_variant_b(service, band_score, rules, status_stable)
    assert dec_b_stable == GatewayDecision.HIGH_RISK

    # Variant D ignores rules -> score < t_high is NOT candidate High -> score band -> ANALYST_REVIEW
    dec_d_stable = route_variant_d(service, band_score, rules, status_stable)
    assert dec_d_stable == GatewayDecision.ANALYST_REVIEW

    # Direct divergence proof
    assert dec_b_stable != dec_d_stable

    # 2. Under Unstable explanation:
    # Both route to ANALYST_REVIEW, but under B it was candidate High routed due to instability,
    # whereas under D it routed due to the score band.
    dec_b_unstable = route_variant_b(service, band_score, rules, status_unstable)
    dec_d_unstable = route_variant_d(service, band_score, rules, status_unstable)
    assert dec_b_unstable == GatewayDecision.ANALYST_REVIEW
    assert dec_d_unstable == GatewayDecision.ANALYST_REVIEW


def test_variant_a_vs_variant_b_rule_only(dummy_scoring_service):
    """Rule-only row (score < t_review with rule) routes differently under A and B."""
    service = dummy_scoring_service
    low_score = 0.5000  # < 0.9500
    rules = ["ORIGIN_ACCOUNT_DRAINED"]
    status = ExplanationStatus.STABLE

    # Variant A routes rule-only directly to ANALYST_REVIEW
    dec_a = service.route_decision(low_score, rules, status)
    assert dec_a == GatewayDecision.ANALYST_REVIEW

    # Variant B gates rules by ML score: score < t_review routes to LOW_RISK
    dec_b = route_variant_b(service, low_score, rules, status)
    assert dec_b == GatewayDecision.LOW_RISK

    # Direct divergence proof
    assert dec_a != dec_b


def test_variant_d_pure_ml_invariance_to_rules(dummy_scoring_service):
    """Variant D decision is strictly invariant to presence of any heuristic rules."""
    service = dummy_scoring_service
    rules = ["ORIGIN_ACCOUNT_DRAINED", "LARGE_TRANSFER_AMOUNT", "DEST_BALANCE_INCONSISTENCY"]

    test_cases = [
        # (score, status, expected_decision)
        (0.40, ExplanationStatus.STABLE, GatewayDecision.LOW_RISK),
        (0.40, ExplanationStatus.UNSTABLE, GatewayDecision.LOW_RISK),
        (0.96, ExplanationStatus.STABLE, GatewayDecision.ANALYST_REVIEW),
        (0.96, ExplanationStatus.UNSTABLE, GatewayDecision.ANALYST_REVIEW),
        (0.98, ExplanationStatus.STABLE, GatewayDecision.HIGH_RISK),
        (0.98, ExplanationStatus.UNSTABLE, GatewayDecision.ANALYST_REVIEW),
    ]

    for score, status, expected in test_cases:
        dec_no_rules = route_variant_d(service, score, [], status)
        dec_with_rules = route_variant_d(service, score, rules, status)
        assert dec_no_rules == expected
        assert dec_with_rules == expected

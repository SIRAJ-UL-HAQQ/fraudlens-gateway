from typing import List, Tuple
from app.models.schemas import PredictionRequest, RuleViolation

class RuleEngine:
    """
    Deterministic heuristic safety layer that runs before or alongside ML inference.
    """
    HIGH_AMOUNT_THRESHOLD = 5000.0
    CRITICAL_AMOUNT_THRESHOLD = 20000.0
    RISKY_LOCATIONS = {"XX", "NK", "IR"}

    def evaluate(self, req: PredictionRequest) -> Tuple[List[RuleViolation], str]:
        violations: List[RuleViolation] = []
        rule_decision = "PASS"  # PASS, REVIEW, BLOCK

        # Rule 1: High Transaction Amount
        if req.amount >= self.CRITICAL_AMOUNT_THRESHOLD:
            violations.append(RuleViolation(
                rule_name="CRITICAL_AMOUNT_EXCEEDED",
                description=f"Transaction amount ${req.amount:,.2f} exceeds critical safeguard threshold.",
                severity="CRITICAL"
            ))
            rule_decision = "BLOCK"
        elif req.amount >= self.HIGH_AMOUNT_THRESHOLD:
            violations.append(RuleViolation(
                rule_name="HIGH_VALUE_TRANSACTION",
                description=f"Transaction amount ${req.amount:,.2f} requires supervisory review.",
                severity="MEDIUM"
            ))
            if rule_decision != "BLOCK":
                rule_decision = "REVIEW"

        # Rule 2: High Risk Geolocation
        if req.location.upper() in self.RISKY_LOCATIONS:
            violations.append(RuleViolation(
                rule_name="HIGH_RISK_GEO_LOCATION",
                description=f"Originated from restricted/high-risk jurisdiction: {req.location}",
                severity="CRITICAL"
            ))
            rule_decision = "BLOCK"

        # Rule 3: Velocity Check Simulation
        if req.custom_features and req.custom_features.get("velocity_1h", 0) > 10:
            violations.append(RuleViolation(
                rule_name="BURST_VELOCITY_ANOMALY",
                description="More than 10 transactions detected within a 1-hour window.",
                severity="MEDIUM"
            ))
            if rule_decision != "BLOCK":
                rule_decision = "REVIEW"

        return violations, rule_decision

rule_engine = RuleEngine()

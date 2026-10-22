import time
import math
import uuid
from typing import Dict, Any
from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.rules import rule_engine
from app.services.shap_explain import shap_service
from app.core.config import settings

class ScoringService:
    """
    Coordinates ML inference, rule-engine evaluation, and SHAP explanation generation.
    """
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        # Placeholder for joblib.load(settings.MODEL_PATH)
        # Defaults to calibrated heuristic scoring if artifact file isn't present yet
        pass

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        start_time = time.perf_counter()
        tx_id = req.transaction_id or f"tx_{uuid.uuid4().hex[:12]}"

        # 1. Evaluate Heuristic Rules
        violations, rule_decision = rule_engine.evaluate(req)

        # 2. Extract Features
        features: Dict[str, float] = {
            "Amount": req.amount,
            "V1_velocity": float(req.custom_features.get("velocity_1h", 1.0) if req.custom_features else 1.0),
            "V2_geo_distance": float(req.custom_features.get("geo_distance_km", 10.0) if req.custom_features else 10.0),
            "V3_device_trust": float(req.custom_features.get("device_trust_score", 0.85) if req.custom_features else 0.85),
            "V4_hour_of_day": float(req.timestamp.hour if req.timestamp else 12.0)
        }

        # 3. ML Inference (calibrated sigmoid approximation fallback)
        linear_combination = (
            (math.log10(max(req.amount, 1.0)) - 2.5) * 0.9 +
            (features["V1_velocity"] - 1.0) * 0.4 +
            (features["V2_geo_distance"] / 100.0) * 0.2 -
            (features["V3_device_trust"] - 0.5) * 1.5
        )
        base_prob = 1.0 / (1.0 + math.exp(-linear_combination))
        base_prob = min(max(base_prob, 0.01), 0.99)

        # 4. Integrate Rules with ML
        if rule_decision == "BLOCK":
            decision = "DECLINE"
            fraud_prob = max(base_prob, 0.95)
        elif rule_decision == "REVIEW":
            decision = "REVIEW"
            fraud_prob = max(base_prob, 0.65)
        else:
            if base_prob >= settings.FRAUD_THRESHOLD_HIGH:
                decision = "DECLINE"
            elif base_prob >= settings.FRAUD_THRESHOLD_REVIEW:
                decision = "REVIEW"
            else:
                decision = "APPROVE"
            fraud_prob = base_prob

        # 5. SHAP Explanation
        shap_explanations, reliability = shap_service.compute_local_explanation(features, fraud_prob)
        
        latency = (time.perf_counter() - start_time) * 1000.0

        return PredictionResponse(
            transaction_id=tx_id,
            fraud_probability=round(fraud_prob, 4),
            risk_score=int(round(fraud_prob * 100)),
            decision=decision,
            rules_triggered=violations,
            shap_explanation=shap_explanations,
            shap_reliability_status=reliability,
            latency_ms=round(latency, 2),
            timestamp=req.timestamp
        )

scoring_service = ScoringService()

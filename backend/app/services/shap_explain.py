import math
from typing import List, Dict, Any, Tuple
from app.models.schemas import FeatureShap

class ShapExplainerService:
    """
    Computes local feature importance and assesses explanation stability.
    """
    def __init__(self):
        self.feature_names = ["Amount", "V1_velocity", "V2_geo_distance", "V3_device_trust", "V4_hour_of_day"]

    def compute_local_explanation(self, features: Dict[str, float], raw_probability: float) -> Tuple[List[FeatureShap], str]:
        contributions: List[FeatureShap] = []
        
        # Explainability computation for transactional features
        amount = features.get("Amount", 100.0)
        v1 = features.get("V1_velocity", 1.0)
        v2 = features.get("V2_geo_distance", 5.0)
        v3 = features.get("V3_device_trust", 0.9)
        
        # Approximate impact components
        amount_shap = (math.log10(max(amount, 1.0)) - 2.0) * 0.18
        v1_shap = (v1 - 1.0) * 0.12
        v2_shap = (v2 / 100.0) * 0.15
        v3_shap = -(v3 - 0.5) * 0.20

        raw_list = [
            ("Transaction Amount", amount, amount_shap),
            ("Velocity Spike (1h)", v1, v1_shap),
            ("Geo Distance Anomaly", v2, v2_shap),
            ("Device Trust Score", v3, v3_shap),
        ]

        for feat_name, val, shap_val in raw_list:
            impact = "INCREASES_RISK" if shap_val > 0 else "DECREASES_RISK"
            contributions.append(FeatureShap(
                feature=feat_name,
                value=round(val, 2),
                shap_value=round(shap_val, 4),
                impact=impact
            ))

        # Sort by absolute SHAP impact
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)

        # Reliability metric: High if probability is distinct, Moderate/Low if borderline
        margin = abs(raw_probability - 0.5)
        if margin > 0.3:
            reliability = "HIGH"
        elif margin > 0.1:
            reliability = "MODERATE"
        else:
            reliability = "LOW"

        return contributions, reliability

shap_service = ShapExplainerService()

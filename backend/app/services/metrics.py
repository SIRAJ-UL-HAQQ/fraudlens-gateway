import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from app.core.config import settings
from app.models.schemas import MetricsSummary, MLMetrics, OperationalMetrics

logger = logging.getLogger(__name__)


class MetricsService:
    """Service responsible for loading research ML benchmark metrics and aggregating with operational telemetry."""

    def __init__(self, metrics_path: Path | str | None = None) -> None:
        self.metrics_path = Path(metrics_path or settings.METRICS_PATH)

    def load_ml_metrics(
        self,
    ) -> tuple[MLMetrics | None, Literal["available", "unavailable"], str | None]:
        """Load and validate ML benchmark metrics from report JSON.

        Returns:
            tuple: (ml_metrics, status, reason)
        """
        if not self.metrics_path.exists():
            return (
                None,
                "unavailable",
                f"ML benchmark report not found at {self.metrics_path}.",
            )

        try:
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            ml_metrics = MLMetrics.model_validate(raw_data)
            return ml_metrics, "available", None
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to parse ML metrics file: %s", e)
            return None, "unavailable", f"Malformed ML benchmark report: {e}"
        except ValidationError as ve:
            logger.warning("ML metrics failed schema validation: %s", ve)
            return None, "unavailable", f"Invalid ML benchmark report schema: {ve}"

    def get_metrics_summary(self, operational: OperationalMetrics) -> MetricsSummary:
        """Combine live DB operational metrics with validated ML benchmark reports."""
        ml_metrics, status, reason = self.load_ml_metrics()

        return MetricsSummary(
            total_transactions=operational.total_transactions,
            decision_distribution=operational.decision_distribution,
            explanation_stability_ratio=operational.explanation_stability_ratio,
            average_risk_score=operational.average_risk_score,
            latency_p50_ms=operational.latency_p50_ms,
            latency_p95_ms=operational.latency_p95_ms,
            ml_metrics=ml_metrics,
            ml_metrics_status=status,
            ml_metrics_reason=reason,
        )

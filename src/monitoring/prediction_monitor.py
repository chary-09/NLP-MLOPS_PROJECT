"""Prediction monitoring service calculating live distribution, confidence, and anomaly stats."""

from typing import Any, Dict, Optional
from src.database.repository import PredictionRepository


class PredictionMonitor:
    """Service to monitor predictions persisted in the production database."""

    def __init__(self, low_confidence_threshold: float = 0.65) -> None:
        self.low_confidence_threshold = low_confidence_threshold

    def get_metrics(
        self,
        repo: PredictionRepository,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate prediction database metrics."""
        stats = repo.get_detailed_prediction_metrics(
            low_confidence_threshold=self.low_confidence_threshold
        )
        if model_version:
            stats["model_version"] = model_version
        return stats


# Backward-compatible helper function
def monitor_prediction(result: dict) -> dict:
    """Extract basic prediction metadata (backward compatibility)."""
    return {"sentiment": result["sentiment"], "confidence": result["confidence"]}

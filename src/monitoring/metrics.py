"""Unified MLOps monitoring service consolidating model, prediction, system, and drift telemetry."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.database.repository import PredictionRepository
from src.model.predictor import SentimentPredictor
from src.monitoring.alerts import monitoring_alert_manager
from src.monitoring.drift_detector import nlp_data_drift_detector
from src.monitoring.latency_monitor import system_metrics_tracker
from src.monitoring.performance_monitor import model_performance_monitor
from src.monitoring.prediction_monitor import PredictionMonitor


class MonitoringService:
    """Consolidated production monitoring service implementing all four MLOps categories."""

    def __init__(self) -> None:
        self.prediction_monitor = PredictionMonitor()
        self.performance_monitor = model_performance_monitor
        self.system_tracker = system_metrics_tracker
        self.drift_detector = nlp_data_drift_detector
        self.alert_manager = monitoring_alert_manager

    def get_model_performance(self, model_version: str = "0.1.0") -> Dict[str, Any]:
        """Category 1: Model Performance Monitoring (baseline vs production ground truth)."""
        return self.performance_monitor.get_summary(model_version=model_version)

    def get_prediction_metrics(
        self, repo: PredictionRepository, model_version: str = "0.1.0"
    ) -> Dict[str, Any]:
        """Category 2: Prediction Monitoring from actual production database."""
        return self.prediction_monitor.get_metrics(repo=repo, model_version=model_version)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Category 3: System Monitoring (FastAPI HTTP telemetry and latency)."""
        return self.system_tracker.get_metrics()

    def get_drift_metrics(
        self,
        repo: PredictionRepository,
        window_size: int = 100,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Category 4: NLP Data Drift Monitoring comparing recent production inputs to reference."""
        production_texts = repo.get_recent_input_texts(limit=window_size)
        return self.drift_detector.calculate_drift(
            production_texts=production_texts,
            threshold=threshold,
        )

    def get_full_summary(
        self,
        repo: PredictionRepository,
        predictor: Optional[SentimentPredictor] = None,
    ) -> Dict[str, Any]:
        """Aggregate all monitoring categories and evaluate threshold alert statuses."""
        model_version = predictor.model_version if predictor else "0.1.0"

        # 1. Collect all metrics categories
        model_perf = self.get_model_performance(model_version=model_version)
        pred_metrics = self.get_prediction_metrics(repo=repo, model_version=model_version)
        sys_metrics = self.get_system_metrics()
        drift_metrics = self.get_drift_metrics(repo=repo)

        # 2. Evaluate threshold status
        alert_eval = self.alert_manager.evaluate(
            prediction_metrics=pred_metrics,
            system_metrics=sys_metrics,
            drift_metrics=drift_metrics,
        )

        now_iso = datetime.now(timezone.utc).isoformat() + "Z"

        return {
            # Legacy & top-level compatibility fields
            "total_predictions": pred_metrics["total_predictions"],
            "sentiment_distribution": pred_metrics["sentiment_distribution"],
            "average_confidence": pred_metrics["average_confidence"],
            "timestamp": now_iso,
            # Production MLOps Monitoring extensions
            "model_version": model_version,
            "status": alert_eval["system_status"],
            "alerts": alert_eval["alerts"],
            "thresholds": alert_eval["thresholds"],
            "model_performance": model_perf,
            "prediction_monitoring": pred_metrics,
            "system_monitoring": sys_metrics,
            "data_drift": drift_metrics,
        }


# Global singleton instance
monitoring_service = MonitoringService()


def prediction_metrics(predictions: List[Dict]) -> Dict[str, Any]:
    """Calculate basic prediction metrics (backward compatibility)."""
    return {"total_predictions": len(predictions)}

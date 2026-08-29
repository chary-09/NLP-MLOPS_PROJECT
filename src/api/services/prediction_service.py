from collections import deque
from datetime import datetime, timezone
import math
from threading import Lock
from typing import Dict, List, Optional
import uuid

from src.api.schemas.metrics import MetricsResponse
from src.api.schemas.prediction import (
    PredictionHistoryResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.services.model_service import ModelService, model_service
from src.core.config import settings
from src.core.logging import get_logger
from src.nlp.preprocessor import clean_text

logger = get_logger("prediction_service")


class PredictionService:
    """Coordinates validation, text preprocessing, vectorization, inference, and in-memory tracking."""

    def __init__(self, model_svc: Optional[ModelService] = None):
        self.model_svc = model_svc or model_service
        self._history: deque = deque(maxlen=settings.MAX_HISTORY_RECORDS)
        self._lock = Lock()

        # Metrics state
        self._total_requests = 0
        self._total_predictions = 0
        self._confidence_sum = 0.0
        self._sentiment_counts: Dict[str, int] = {"positive": 0, "negative": 0}

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run full prediction pipeline on incoming text."""
        with self._lock:
            self._total_requests += 1

        if not self.model_svc.is_loaded:
            logger.warning("Attempted prediction before model artifacts were loaded. Triggering load.")
            self.model_svc.load_artifacts()

        # 1. Validation & Preprocessing (Reusing Phase 1 clean_text logic)
        raw_text = request.text
        cleaned = clean_text(raw_text)

        # 2. Vectorization (Reusing Phase 1 TF-IDF Vectorizer)
        features = self.model_svc.vectorizer.transform([cleaned])

        # 3. Model Inference (Reusing Phase 1 Trained Classifier)
        raw_prediction = self.model_svc.model.predict(features)[0]
        sentiment_label = str(raw_prediction).lower()

        # 4. Confidence Calculation
        if hasattr(self.model_svc.model, "predict_proba"):
            probabilities = self.model_svc.model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        elif hasattr(self.model_svc.model, "decision_function"):
            score = float(self.model_svc.model.decision_function(features)[0])
            confidence = float(1.0 / (1.0 + math.exp(-abs(score))))
        else:
            confidence = 1.0

        confidence = round(confidence, 4)

        # 5. Build Response
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        model_version = self.model_svc.get_version()

        response = PredictionResponse(
            prediction_id=prediction_id,
            text=raw_text,
            sentiment=sentiment_label,
            confidence=confidence,
            model_version=model_version,
            timestamp=timestamp,
        )

        # 6. Store in application history & update metrics
        with self._lock:
            self._total_predictions += 1
            self._confidence_sum += confidence
            self._sentiment_counts[sentiment_label] = (
                self._sentiment_counts.get(sentiment_label, 0) + 1
            )
            self._history.appendleft(response)

        logger.info(
            f"Prediction id={prediction_id} sentiment={sentiment_label} "
            f"confidence={confidence:.4f} version={model_version}"
        )
        return response

    def get_history(self, limit: int = 10, offset: int = 0) -> PredictionHistoryResponse:
        """Retrieve recent prediction history from in-memory application layer."""
        with self._lock:
            total = len(self._history)
            records = list(self._history)[offset : offset + limit]

        return PredictionHistoryResponse(
            total=total,
            limit=limit,
            offset=offset,
            predictions=records,
        )

    def get_metrics(self) -> MetricsResponse:
        """Retrieve live inference and operational metrics."""
        with self._lock:
            total_reqs = self._total_requests
            total_preds = self._total_predictions
            avg_conf = (
                round(self._confidence_sum / total_preds, 4)
                if total_preds > 0
                else 0.0
            )
            distribution = dict(self._sentiment_counts)

        return MetricsResponse(
            total_requests=total_reqs,
            total_predictions=total_preds,
            sentiment_distribution=distribution,
            average_confidence=avg_conf,
            uptime_seconds=self.model_svc.get_uptime_seconds(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# Global singleton instance
prediction_service = PredictionService()

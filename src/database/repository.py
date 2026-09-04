import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from sqlalchemy import func
from sqlalchemy.orm import Session
from .models import PredictionRecord

logger = logging.getLogger(__name__)


class PredictionRepository:
    """Repository layer handling all database operations for prediction records."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        prediction_id: str,
        input_text: str,
        sentiment: str,
        confidence: float,
        model_version: str,
        timestamp: Optional[datetime] = None,
    ) -> PredictionRecord:
        """Store a new prediction record in the database."""
        try:
            record = PredictionRecord(
                prediction_id=prediction_id,
                input_text=input_text,
                sentiment=str(sentiment).lower(),
                confidence=float(confidence),
                model_version=str(model_version),
                timestamp=timestamp or datetime.now(timezone.utc),
            )
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
            logger.info(f"Saved prediction {prediction_id} to database.")
            return record
        except Exception as exc:
            self.db.rollback()
            logger.error(f"Failed to save prediction {prediction_id} to database: {exc}")
            raise

    def get_by_id(self, prediction_id: str) -> Optional[PredictionRecord]:
        """Retrieve a specific prediction record by its UUID."""
        return (
            self.db.query(PredictionRecord)
            .filter(PredictionRecord.prediction_id == prediction_id)
            .first()
        )

    def list_recent(
        self, limit: int = 10, offset: int = 0
    ) -> Tuple[List[PredictionRecord], int]:
        """Retrieve paginated prediction records ordered by newest first, along with total count."""
        total = self.db.query(func.count(PredictionRecord.prediction_id)).scalar() or 0
        records = (
            self.db.query(PredictionRecord)
            .order_by(PredictionRecord.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return records, total

    def count(self) -> int:
        """Get total count of stored predictions."""
        return self.db.query(func.count(PredictionRecord.prediction_id)).scalar() or 0

    def get_metrics_summary(self) -> Dict:
        """Calculate aggregation metrics directly from database records."""
        total = self.count()
        if total == 0:
            return {
                "total_predictions": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0},
                "average_confidence": 0.0,
            }

        avg_conf = (
            self.db.query(func.avg(PredictionRecord.confidence)).scalar() or 0.0
        )
        distribution_query = (
            self.db.query(
                PredictionRecord.sentiment,
                func.count(PredictionRecord.prediction_id),
            )
            .group_by(PredictionRecord.sentiment)
            .all()
        )
        distribution = {label.lower(): count for label, count in distribution_query}

        return {
            "total_predictions": total,
            "sentiment_distribution": distribution,
            "average_confidence": round(float(avg_conf), 4),
        }

    def get_detailed_prediction_metrics(self, low_confidence_threshold: float = 0.65) -> Dict:
        """Calculate comprehensive prediction metrics including confidence stats and class percentages."""
        total = self.count()
        if total == 0:
            return {
                "total_predictions": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0},
                "positive_percentage": 0.0,
                "negative_percentage": 0.0,
                "neutral_percentage": None,
                "average_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "low_confidence_threshold": low_confidence_threshold,
                "low_confidence_count": 0,
                "low_confidence_percentage": 0.0,
            }

        avg_conf = self.db.query(func.avg(PredictionRecord.confidence)).scalar() or 0.0
        min_conf = self.db.query(func.min(PredictionRecord.confidence)).scalar() or 0.0
        max_conf = self.db.query(func.max(PredictionRecord.confidence)).scalar() or 0.0

        low_conf_count = (
            self.db.query(func.count(PredictionRecord.prediction_id))
            .filter(PredictionRecord.confidence < low_confidence_threshold)
            .scalar()
            or 0
        )

        distribution_query = (
            self.db.query(
                PredictionRecord.sentiment,
                func.count(PredictionRecord.prediction_id),
            )
            .group_by(PredictionRecord.sentiment)
            .all()
        )
        distribution = {"positive": 0, "negative": 0}
        for label, count in distribution_query:
            distribution[label.lower()] = count

        pos_count = distribution.get("positive", 0)
        neg_count = distribution.get("negative", 0)
        pos_pct = round((pos_count / total * 100.0), 2) if total > 0 else 0.0
        neg_pct = round((neg_count / total * 100.0), 2) if total > 0 else 0.0
        low_conf_pct = round((low_conf_count / total * 100.0), 2) if total > 0 else 0.0

        return {
            "total_predictions": total,
            "sentiment_distribution": distribution,
            "positive_percentage": pos_pct,
            "negative_percentage": neg_pct,
            "neutral_percentage": None,  # Binary sentiment system: strictly no invented neutral predictions
            "average_confidence": round(float(avg_conf), 4),
            "min_confidence": round(float(min_conf), 4),
            "max_confidence": round(float(max_conf), 4),
            "low_confidence_threshold": low_confidence_threshold,
            "low_confidence_count": low_conf_count,
            "low_confidence_percentage": low_conf_pct,
        }

    def get_recent_input_texts(self, limit: int = 100) -> List[str]:
        """Fetch the most recent input texts for drift and distribution monitoring."""
        records = (
            self.db.query(PredictionRecord.input_text)
            .order_by(PredictionRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [r[0] for r in records if r and r[0]]

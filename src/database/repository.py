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

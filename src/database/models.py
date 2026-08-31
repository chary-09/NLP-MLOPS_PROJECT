import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Text, DateTime
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""
    pass


class PredictionRecord(Base):
    """Database model for storing sentiment prediction records."""

    __tablename__ = "predictions"

    prediction_id = Column(
        String(36),
        primary_key=True,
        index=True,
        default=lambda: str(uuid.uuid4()),
        nullable=False,
    )
    input_text = Column(Text, nullable=False)
    sentiment = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    def to_dict(self) -> dict:
        """Convert record to dictionary matching PredictionResponse schema."""
        ts = self.timestamp.isoformat() if self.timestamp else datetime.now(timezone.utc).isoformat()
        if not ts.endswith("Z") and "+" not in ts:
            ts += "Z"
        return {
            "prediction_id": self.prediction_id,
            "text": self.input_text,
            "sentiment": self.sentiment,
            "confidence": round(float(self.confidence), 4),
            "model_version": self.model_version,
            "timestamp": ts,
        }

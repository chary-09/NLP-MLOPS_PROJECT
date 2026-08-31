from fastapi import Depends
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.repository import PredictionRepository
from src.model.predictor import SentimentPredictor

# Shared singleton predictor instance
_predictor_instance = None


def get_predictor() -> SentimentPredictor:
    """Return the shared pre-loaded SentimentPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SentimentPredictor().load()
    return _predictor_instance


def get_prediction_repository(
    db: Session = Depends(get_db),
) -> PredictionRepository:
    """Dependency provider for PredictionRepository."""
    return PredictionRepository(db)

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


def get_model_service():
    """Dependency provider for ModelService (or SentimentPredictor)."""
    try:
        from src.api.services.model_service import model_service
        return model_service
    except Exception:
        return get_predictor()


def get_prediction_service():
    """Dependency provider for PredictionService."""
    try:
        from src.api.services.prediction_service import prediction_service
        return prediction_service
    except Exception:
        return None


# ---------------------------------------------------------------------------
# XAI — Explanation service dependency (Day 3)
# ---------------------------------------------------------------------------

_explanation_service_instance = None


def get_explanation_service():
    """Return the shared ExplanationService singleton (lazy initialised)."""
    global _explanation_service_instance
    if _explanation_service_instance is None:
        from src.xai.explanation_service import ExplanationService
        _explanation_service_instance = ExplanationService(predictor=get_predictor())
    return _explanation_service_instance

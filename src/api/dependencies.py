from src.api.services.model_service import ModelService, model_service
from src.api.services.prediction_service import PredictionService, prediction_service


def get_model_service() -> ModelService:
    """Dependency injection provider for ModelService."""
    return model_service


def get_prediction_service() -> PredictionService:
    """Dependency injection provider for PredictionService."""
    return prediction_service


# Backwards compatibility helper for existing references
def get_predictor():
    """Backward compatibility provider for legacy SentimentPredictor callers."""
    from src.model.predictor import SentimentPredictor
    return SentimentPredictor()

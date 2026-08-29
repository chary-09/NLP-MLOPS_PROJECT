from fastapi import APIRouter, Depends, HTTPException, Query, status
from src.api.dependencies import get_prediction_service
from src.api.schemas.prediction import (
    PredictionHistoryResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.services.prediction_service import PredictionService
from src.core.logging import get_logger

logger = get_logger("prediction_route")
router = APIRouter(tags=["Sentiment Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Text Sentiment",
    description=(
        "Classifies input text sentiment as positive or negative. "
        "Executes the full inference pipeline: text validation -> Phase 1 preprocessing -> "
        "TF-IDF vectorizer -> trained NLP model -> confidence calculation."
    ),
    responses={
        200: {
            "description": "Successful sentiment prediction",
            "content": {
                "application/json": {
                    "example": {
                        "prediction_id": "8a7c2b4d-1e3f-45a6-9b8c-7d6e5f4a3b2c",
                        "text": "The product was amazing!",
                        "sentiment": "positive",
                        "confidence": 0.94,
                        "model_version": "0.1.0",
                        "timestamp": "2026-08-29T14:06:03.123456Z",
                    }
                }
            },
        },
        422: {"description": "Validation error (e.g. empty or blank text, text too long)"},
        503: {"description": "Model artifacts not available in memory"},
    },
)
def predict_sentiment(
    request: PredictionRequest,
    pred_svc: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Classify sentiment for raw input text."""
    try:
        return pred_svc.predict(request)
    except FileNotFoundError as exc:
        logger.error(f"Model artifacts missing during prediction: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded or available on the server.",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected error during inference: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(exc)}",
        ) from exc


@router.get(
    "/predictions",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="List Prediction History",
    description="Returns recent prediction records stored in the application layer (in-memory buffer).",
)
def get_prediction_history(
    limit: int = Query(10, ge=1, le=100, description="Number of prediction records to return"),
    offset: int = Query(0, ge=0, description="Offset index for pagination"),
    pred_svc: PredictionService = Depends(get_prediction_service),
) -> PredictionHistoryResponse:
    """Retrieve in-memory prediction history."""
    return pred_svc.get_history(limit=limit, offset=offset)

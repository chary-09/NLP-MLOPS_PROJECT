import logging
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.database.repository import PredictionRepository
from src.api.dependencies import get_prediction_repository, get_predictor
from src.api.schemas.prediction import (
    PredictionHistoryResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.model.predictor import SentimentPredictor

logger = logging.getLogger("prediction_route")
router = APIRouter(tags=["Sentiment Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Sentiment & Persist to Database",
    description=(
        "Classifies input text sentiment using the Phase 1 NLP pipeline and "
        "persists the prediction record into the database."
    ),
    responses={
        200: {"description": "Successful sentiment prediction and persistence"},
        422: {"description": "Validation error (e.g. empty or blank text)"},
        503: {"description": "Model artifacts not available on server"},
    },
)
def predict_sentiment(
    request: PredictionRequest,
    predictor: SentimentPredictor = Depends(get_predictor),
    repo: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionResponse:
    """Run inference, persist result in database, and return formatted response."""
    try:
        # 1. Inference via Phase 1 NLP pipeline
        result = predictor.predict(request.text)
        sentiment = str(result["sentiment"]).lower()
        confidence = round(float(result["confidence"]), 4)
        model_version = result.get("model_version", predictor.model_version)

        # 2. Assign unique ID and timestamp
        prediction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # 3. Store record in database
        record = repo.create(
            prediction_id=prediction_id,
            input_text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            model_version=model_version,
            timestamp=now,
        )

        return PredictionResponse(
            prediction_id=record.prediction_id,
            text=record.input_text,
            sentiment=record.sentiment,
            confidence=record.confidence,
            model_version=record.model_version,
            timestamp=record.to_dict()["timestamp"],
        )
    except FileNotFoundError as exc:
        logger.error(f"Model artifacts missing during prediction: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded or available on the server.",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected error during inference or storage: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(exc)}",
        ) from exc


@router.get(
    "/predictions",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="List Prediction History",
    description="Reads prediction records from the database with pagination (newest first). Does NOT re-run model inference.",
)
def get_prediction_history(
    limit: int = Query(10, ge=1, le=100, description="Number of prediction records to return"),
    offset: int = Query(0, ge=0, description="Offset index for pagination"),
    repo: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionHistoryResponse:
    """Retrieve stored prediction history directly from the database."""
    try:
        records, total = repo.list_recent(limit=limit, offset=offset)
        predictions = [
            PredictionResponse(
                prediction_id=r.prediction_id,
                text=r.input_text,
                sentiment=r.sentiment,
                confidence=r.confidence,
                model_version=r.model_version,
                timestamp=r.to_dict()["timestamp"],
            )
            for r in records
        ]
        return PredictionHistoryResponse(
            total=total,
            limit=limit,
            offset=offset,
            predictions=predictions,
        )
    except Exception as exc:
        logger.error(f"Failed to query database predictions: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(exc)}",
        ) from exc

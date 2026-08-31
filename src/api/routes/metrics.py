from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status

from src.database.repository import PredictionRepository
from src.api.dependencies import get_prediction_repository
from src.api.schemas.metrics import MetricsResponse

router = APIRouter(tags=["Metrics & Monitoring"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Database & Inference Metrics",
    description="Returns live metrics aggregated directly from stored database records.",
)
def get_metrics(
    repo: PredictionRepository = Depends(get_prediction_repository),
) -> MetricsResponse:
    """Retrieve runtime metrics from database records."""
    summary = repo.get_metrics_summary()
    return MetricsResponse(
        total_predictions=summary["total_predictions"],
        sentiment_distribution=summary["sentiment_distribution"],
        average_confidence=summary["average_confidence"],
        timestamp=datetime.now(timezone.utc).isoformat() + "Z",
    )

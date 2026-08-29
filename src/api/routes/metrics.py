from fastapi import APIRouter, Depends, status
from src.api.dependencies import get_prediction_service
from src.api.schemas.metrics import MetricsResponse
from src.api.services.prediction_service import PredictionService

router = APIRouter(tags=["Metrics & Monitoring"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get API & Inference Metrics",
    description="Returns live metrics including request counts, prediction counts, sentiment distribution, and average confidence.",
)
def get_metrics(
    pred_svc: PredictionService = Depends(get_prediction_service),
) -> MetricsResponse:
    """Retrieve runtime operational metrics."""
    return pred_svc.get_metrics()

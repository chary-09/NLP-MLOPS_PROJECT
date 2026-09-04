"""API endpoints for comprehensive MLOps monitoring and telemetry."""

from fastapi import APIRouter, Depends, Query, status

from src.api.dependencies import get_prediction_repository, get_predictor
from src.api.schemas.metrics import (
    DataDriftResponse,
    GroundTruthEvaluationRequest,
    GroundTruthEvaluationResponse,
    MetricsResponse,
    ModelPerformanceResponse,
    PredictionMonitoringResponse,
    SystemMetricsResponse,
)
from src.database.repository import PredictionRepository
from src.model.predictor import SentimentPredictor
from src.monitoring.metrics import monitoring_service
from src.monitoring.performance_monitor import model_performance_monitor

router = APIRouter(tags=["Metrics & Monitoring"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Unified MLOps Monitoring Dashboard & Telemetry",
    description="Returns full monitoring summary across Model Performance, Predictions, System Latency, and NLP Data Drift.",
)
def get_metrics(
    repo: PredictionRepository = Depends(get_prediction_repository),
    predictor: SentimentPredictor = Depends(get_predictor),
) -> MetricsResponse:
    """Retrieve full live telemetry across all four MLOps monitoring categories."""
    summary = monitoring_service.get_full_summary(repo=repo, predictor=predictor)
    return MetricsResponse(**summary)


@router.get(
    "/metrics/model",
    response_model=ModelPerformanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Category 1: Model Performance Monitoring",
    description="Returns baseline evaluation metrics and production ground-truth performance if observed.",
)
def get_model_metrics(
    predictor: SentimentPredictor = Depends(get_predictor),
) -> ModelPerformanceResponse:
    """Return model baseline metrics and authentic production evaluation status."""
    perf = monitoring_service.get_model_performance(model_version=predictor.model_version)
    return ModelPerformanceResponse(**perf)


@router.get(
    "/metrics/predictions",
    response_model=PredictionMonitoringResponse,
    status_code=status.HTTP_200_OK,
    summary="Category 2: Prediction Telemetry & Confidence Distribution",
    description="Aggregates prediction statistics, confidence extremes, and low-confidence percentage directly from the SQLite database.",
)
def get_prediction_metrics(
    repo: PredictionRepository = Depends(get_prediction_repository),
    predictor: SentimentPredictor = Depends(get_predictor),
) -> PredictionMonitoringResponse:
    """Retrieve database prediction volume, class breakdown, and confidence stats."""
    stats = monitoring_service.get_prediction_metrics(repo=repo, model_version=predictor.model_version)
    return PredictionMonitoringResponse(**stats)


@router.get(
    "/metrics/system",
    response_model=SystemMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Category 3: System Request & Latency Monitoring",
    description="Returns FastAPI HTTP request counts, average/min/max latencies, error counts, and endpoint traffic breakdown.",
)
def get_system_metrics() -> SystemMetricsResponse:
    """Retrieve runtime HTTP server statistics and latency percentiles."""
    sys_metrics = monitoring_service.get_system_metrics()
    return SystemMetricsResponse(**sys_metrics)


@router.get(
    "/metrics/drift",
    response_model=DataDriftResponse,
    status_code=status.HTTP_200_OK,
    summary="Category 4: NLP Data Drift Monitoring",
    description="Performs two-sample Kolmogorov-Smirnov distribution test and TF-IDF vocabulary out-of-vocabulary (OOV) rate analysis against training baseline.",
)
def get_drift_metrics(
    window: int = Query(100, ge=5, le=1000, description="Number of recent production inputs to evaluate"),
    threshold: float = Query(0.20, ge=0.01, le=1.0, description="Statistical drift threshold"),
    repo: PredictionRepository = Depends(get_prediction_repository),
) -> DataDriftResponse:
    """Calculate statistical text length and vocabulary drift on recent production inputs."""
    drift = monitoring_service.get_drift_metrics(repo=repo, window_size=window, threshold=threshold)
    return DataDriftResponse(**drift)


@router.post(
    "/metrics/evaluate-production",
    response_model=GroundTruthEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Production Performance with Genuine Ground-Truth",
    description="Allows uploading verified production ground-truth labels to compute authentic production accuracy, precision, recall, and F1-score.",
)
def evaluate_production_ground_truth(
    request: GroundTruthEvaluationRequest,
) -> GroundTruthEvaluationResponse:
    """Calculate actual production performance from verified labels without fabricating metrics."""
    predictions = [item.prediction for item in request.evaluations]
    ground_truths = [item.ground_truth for item in request.evaluations]
    eval_result = model_performance_monitor.evaluate_ground_truth(
        predictions=predictions, ground_truth=ground_truths
    )
    return GroundTruthEvaluationResponse(**eval_result)

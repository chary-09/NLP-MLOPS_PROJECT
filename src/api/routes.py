import json
import logging
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import MODEL_DIR
from src.database.connection import get_db
from src.database.repository import PredictionRepository
from .dependencies import get_prediction_repository, get_predictor, get_explanation_service
from .schemas import (
    DataDriftResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureContribution,
    GroundTruthEvaluationRequest,
    GroundTruthEvaluationResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    ModelPerformanceResponse,
    PredictionHistoryResponse,
    PredictionMonitoringResponse,
    PredictionRequest,
    PredictionResponse,
    SystemMetricsResponse,
)
from src.xai.explanation_service import ExplanationService
from src.model.predictor import SentimentPredictor
from src.monitoring.metrics import monitoring_service
from src.monitoring.performance_monitor import model_performance_monitor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Sentiment & Store in Database",
    description="Preprocesses text, extracts TF-IDF features, runs classifier, saves to SQLite database, and returns prediction.",
)
def predict(
    request: PredictionRequest,
    predictor: SentimentPredictor = Depends(get_predictor),
    repo: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionResponse:
    """Classify sentiment, store the result in the database, and return response."""
    try:
        # 1. Run inference pipeline (Phase 1 preprocessor -> vectorizer -> model)
        result = predictor.predict(request.text)
        sentiment = str(result["sentiment"]).lower()
        confidence = round(float(result["confidence"]), 4)
        model_version = result.get("model_version", predictor.model_version)

        # 2. Generate unique ID and timestamp
        prediction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # 3. Persist record to database
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
    except FileNotFoundError as error:
        logger.error(f"Model artifacts missing: {error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not trained yet.",
        ) from error
    except Exception as error:
        logger.error(f"Prediction failed: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference or storage failure: {str(error)}",
        ) from error


@router.get(
    "/predictions",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Stored Prediction History",
    description="Reads prediction history directly from the database with pagination (newest first). Does NOT re-run model inference.",
)
def get_predictions(
    limit: int = Query(10, ge=1, le=100, description="Max number of records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    repo: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionHistoryResponse:
    """Fetch stored prediction records from the database."""
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
    except Exception as error:
        logger.error(f"Failed to retrieve predictions: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(error)}",
        ) from error


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verifies API status, model/vectorizer loading, and database connectivity.",
)
def healthcheck(
    predictor: SentimentPredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
) -> HealthResponse:
    """Check API, model, and database operational status."""
    model_loaded = predictor.model is not None and predictor.vectorizer is not None
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as exc:
        logger.warning(f"Database healthcheck ping failed: {exc}")

    is_healthy = model_loaded and db_connected
    overall_status = "healthy" if is_healthy else ("degraded" if model_loaded else "unhealthy")

    return HealthResponse(
        status=overall_status,
        api=True,
        model_loaded=model_loaded,
        vectorizer_loaded=predictor.vectorizer is not None,
        database_connected=db_connected,
        model_version=predictor.model_version,
        timestamp=datetime.now(timezone.utc).isoformat() + "Z",
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model & Vectorizer Metadata",
)
def model_info(predictor: SentimentPredictor = Depends(get_predictor)) -> ModelInfoResponse:
    """Return model specifications, vectorizer details, and metrics."""
    metrics_path = MODEL_DIR / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}

    vectorizer = predictor.vectorizer
    vectorizer_info = {
        "type": vectorizer.__class__.__name__ if vectorizer else "NotLoaded",
        "max_features": getattr(vectorizer, "max_features", 5000),
        "ngram_range": list(getattr(vectorizer, "ngram_range", (1, 2))),
        "vocabulary_size": len(getattr(vectorizer, "vocabulary_", {})),
    }

    classes = ["negative", "positive"]
    if predictor.model and hasattr(predictor.model, "classes_"):
        classes = [str(c).lower() for c in predictor.model.classes_]

    best_model_name = metrics_data.get("best_model", "logistic_regression")
    training_metrics = metrics_data.get("models", {}).get(best_model_name)

    return ModelInfoResponse(
        model_name=best_model_name,
        model_version=predictor.model_version,
        vectorizer=vectorizer_info,
        classes=classes,
        training_metrics=training_metrics,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
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


@router.post(
    "/explain",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Explain Sentiment Prediction (SHAP / LIME)",
    description=(
        "Runs the same TF-IDF → Logistic Regression inference as POST /predict, "
        "then produces word-level feature contributions using SHAP (LinearExplainer) "
        "or LIME (LimeTextExplainer). Set `method` to `'shap'`, `'lime'`, or `'both'`."
    ),
    responses={
        200: {"description": "Successful explanation with feature contributions"},
        422: {"description": "Validation error (empty text, invalid method)"},
        503: {"description": "Model or explainer not available"},
    },
)
def explain_sentiment(
    request: ExplainRequest,
    svc: ExplanationService = Depends(get_explanation_service),
) -> ExplainResponse:
    """Generate an XAI explanation for the given text using SHAP or LIME."""
    try:
        result = svc.explain(
            text=request.text,
            method=request.method,
            top_n=request.top_n,
        )
    except FileNotFoundError as exc:
        logger.error("Model artifacts missing during explanation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Run training scripts first.",
        ) from exc
    except Exception as exc:
        logger.error("Explanation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {str(exc)}",
        ) from exc

    if result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )

    contributions = [
        FeatureContribution(
            feature=f["feature"],
            importance=f["importance"],
            method=f.get("method"),
        )
        for f in result["explanation"]
    ]

    return ExplainResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        method=result["method"],
        explanation=contributions,
        positive_words=result["positive_words"],
        negative_words=result["negative_words"],
    )

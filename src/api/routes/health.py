from datetime import datetime, timezone
import logging
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.api.dependencies import get_predictor
from src.api.schemas.health import HealthResponse
from src.model.predictor import SentimentPredictor

logger = logging.getLogger("health_route")
router = APIRouter(tags=["Health & Status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service & Database Health Check",
    description="Verifies that the FastAPI server is running, ML model artifacts are loaded in memory, and the database is reachable.",
    status_code=status.HTTP_200_OK,
)
def check_health(
    predictor: SentimentPredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
):
    """Check health of API server, ML model, and database connection."""
    model_loaded = predictor.model is not None and predictor.vectorizer is not None
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as exc:
        logger.warning(f"Database healthcheck ping failed: {exc}")

    is_healthy = model_loaded and db_connected
    overall_status = "healthy" if is_healthy else ("degraded" if model_loaded else "unhealthy")
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    payload = HealthResponse(
        status=overall_status,
        api=True,
        model_loaded=model_loaded,
        vectorizer_loaded=predictor.vectorizer is not None,
        database_connected=db_connected,
        model_version=predictor.model_version if model_loaded else "unavailable",
        timestamp=datetime.now(timezone.utc).isoformat() + "Z",
    )

    return JSONResponse(status_code=status_code, content=payload.model_dump())

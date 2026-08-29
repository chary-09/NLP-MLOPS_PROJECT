from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from src.api.dependencies import get_model_service
from src.api.schemas.health import HealthResponse
from src.api.services.model_service import ModelService

router = APIRouter(tags=["Health & Status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Verifies that the FastAPI server is running and ML model artifacts are loaded in memory.",
    status_code=status.HTTP_200_OK,
)
def check_health(model_svc: ModelService = Depends(get_model_service)):
    """Check health of API server and model readiness."""
    is_loaded = model_svc.is_loaded
    overall_status = "healthy" if is_loaded else "degraded"
    status_code = status.HTTP_200_OK if is_loaded else status.HTTP_503_SERVICE_UNAVAILABLE

    payload = HealthResponse(
        status=overall_status,
        api=True,
        model_loaded=is_loaded,
        vectorizer_loaded=is_loaded,
        model_version=model_svc.get_version() if is_loaded else "unavailable",
        uptime_seconds=model_svc.get_uptime_seconds(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return JSONResponse(status_code=status_code, content=payload.model_dump())

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.middleware import configure_middleware
from src.api.routes import api_router
from src.api.services.model_service import model_service
from src.core.config import settings
from src.core.logging import get_logger, setup_logging

logger = get_logger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to load model artifacts once at startup."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("Initializing Sentiment Analysis Production API...")
    logger.info(f"API Version: {settings.API_VERSION}")

    # Load ML artifacts once during startup
    try:
        model_service.load_artifacts()
        logger.info(f"Model [{model_service.get_model_name()}] v{model_service.get_version()} ready.")
    except Exception as exc:
        logger.error(f"Failed to load model artifacts on startup: {exc}", exc_info=True)

    logger.info("=" * 60)
    yield
    logger.info("Shutting down Sentiment Analysis Production API...")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure Cross-Cutting Middleware
configure_middleware(app)

# Register Root Endpoints: /predict, /health, /metrics, /predictions, /model-info
app.include_router(api_router)

# Also mount under /api/v1 prefix for standard API versioning
app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom error formatter for request validation errors."""
    error_messages = []
    for err in exc.errors():
        field_loc = " -> ".join([str(loc) for loc in err.get("loc", [])])
        error_messages.append({
            "field": field_loc,
            "message": err.get("msg"),
            "type": err.get("type"),
        })

    logger.warning(f"Validation error on {request.url.path}: {error_messages}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": error_messages,
            "status_code": 422,
        },
    )


@app.get("/", include_in_schema=False)
def root_redirect():
    """Root endpoint providing quick navigation links."""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "metrics": "/metrics",
    }

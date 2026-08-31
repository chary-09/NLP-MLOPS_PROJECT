import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.database.migrations import create_tables
from .dependencies import get_predictor
from .middleware import configure_middleware
from .routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("sentiment_api")


def _sanitize_error(err: dict) -> dict:
    """Ensure error dictionary contains only JSON-serializable primitives."""
    clean = {}
    for k, v in err.items():
        if k == "ctx" and isinstance(v, dict):
            clean[k] = {ctx_k: str(ctx_v) for ctx_k, ctx_v in v.items()}
        elif isinstance(v, Exception):
            clean[k] = str(v)
        elif isinstance(v, (list, tuple)):
            clean[k] = [str(item) if isinstance(item, Exception) else item for item in v]
        else:
            clean[k] = v
    return clean


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to initialize database tables and pre-load ML model once at startup."""
    logger.info("Initializing Sentiment Analysis Production API...")
    try:
        create_tables()
        logger.info("Database initialized successfully.")
    except Exception as exc:
        logger.error(f"Database initialization warning: {exc}")

    try:
        predictor = get_predictor()
        logger.info(f"NLP Model loaded: version={predictor.model_version}")
    except Exception as exc:
        logger.warning(f"Model load on startup deferred: {exc}")

    yield
    logger.info("Shutting down Sentiment Analysis Production API...")


app = FastAPI(
    title="Sentiment Analysis API",
    version="0.1.0",
    description="Production-grade Sentiment Analysis API with SQLite persistence and TF-IDF NLP model.",
    lifespan=lifespan,
)

configure_middleware(app)

# Include routes at root level for /predict, /predictions, /health, /metrics, /model-info
app.include_router(router, tags=["sentiment"])

# Also include with prefix /api/v1 for standard API versioning
app.include_router(router, prefix="/api/v1", tags=["sentiment-v1"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom formatter for input validation errors that safely serializes error details."""
    clean_errors = [_sanitize_error(err) for err in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": clean_errors,
            "status_code": 422,
        },
    )


@app.get("/", include_in_schema=False)
def root_redirect():
    return {
        "service": "Sentiment Analysis API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "predictions": "/predictions",
    }

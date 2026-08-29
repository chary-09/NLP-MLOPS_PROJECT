"""API Route modular aggregation."""

from fastapi import APIRouter

from .health import router as health_router
from .metrics import router as metrics_router
from .model import router as model_router
from .prediction import router as prediction_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(prediction_router)
api_router.include_router(model_router)
api_router.include_router(metrics_router)

__all__ = [
    "api_router",
    "health_router",
    "metrics_router",
    "model_router",
    "prediction_router",
]

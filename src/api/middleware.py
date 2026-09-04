import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import get_logger

from src.monitoring.latency_monitor import system_metrics_tracker

logger = get_logger("api_middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for measuring request processing time, tracking system metrics, and logging HTTP requests."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            process_time = time.perf_counter() - start_time
            system_metrics_tracker.record_request(request.url.path, 500, process_time)
            raise

        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"

        # Record in system metrics tracker
        system_metrics_tracker.record_request(
            path=request.url.path,
            status_code=response.status_code,
            latency_seconds=process_time,
        )

        logger.info(
            f"{request.method} {request.url.path} "
            f"- Status: {response.status_code} - Duration: {process_time:.4f}s"
        )
        return response


def configure_middleware(app: FastAPI) -> None:
    """Register all application middleware."""
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Add timing and request logging
    app.add_middleware(RequestLoggingMiddleware)

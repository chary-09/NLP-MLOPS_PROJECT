"""Read-only access to recent records from the existing application logger."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from src.monitoring.log_buffer import log_buffer_handler

router = APIRouter(tags=["System Logs"])


@router.get("/logs", summary="Recent application logs")
def get_logs(
    level: Optional[str] = Query(None, description="INFO, WARNING, or ERROR"),
    component: Optional[str] = Query(None),
    date: Optional[str] = Query(None, description="UTC date in YYYY-MM-DD format"),
) -> Dict[str, Any]:
    """Return actual log records captured by the configured Python logger."""
    normalized_level = level.upper() if level else None
    if normalized_level and normalized_level not in {"INFO", "WARNING", "ERROR"}:
        return {"logs": [], "total": 0, "error": "Unsupported log level"}
    records = log_buffer_handler.get_records(
        level=normalized_level,
        component=component,
        date=date,
    )
    return {"logs": records, "total": len(records)}
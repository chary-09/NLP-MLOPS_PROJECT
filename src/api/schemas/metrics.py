from typing import Dict, Optional
from pydantic import BaseModel, Field


class MetricsResponse(BaseModel):
    """Inference and API runtime metrics schema."""

    total_requests: Optional[int] = Field(None, description="Total prediction requests received since startup", json_schema_extra={"example": 50})
    total_predictions: int = Field(..., description="Total successful predictions produced", json_schema_extra={"example": 50})
    sentiment_distribution: Dict[str, int] = Field(
        ...,
        description="Sentiment classification counts",
        json_schema_extra={"example": {"positive": 32, "negative": 18}},
    )
    average_confidence: float = Field(..., description="Mean prediction confidence across all queries", json_schema_extra={"example": 0.9231})
    uptime_seconds: Optional[float] = Field(None, description="Uptime in seconds", json_schema_extra={"example": 320.8})
    timestamp: str = Field(..., description="Current UTC timestamp (ISO-8601)", json_schema_extra={"example": "2026-08-29T14:06:03.123456Z"})


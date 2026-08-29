from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System and component health status schema."""

    status: str = Field(..., description="Overall service status (healthy, degraded, unhealthy)", json_schema_extra={"example": "healthy"})
    api: bool = Field(..., description="API server running indicator", json_schema_extra={"example": True})
    model_loaded: bool = Field(..., description="Status indicating if the sentiment classifier model is loaded", json_schema_extra={"example": True})
    vectorizer_loaded: bool = Field(..., description="Status indicating if the TF-IDF vectorizer is loaded", json_schema_extra={"example": True})
    model_version: str = Field(..., description="Loaded model version string", json_schema_extra={"example": "0.1.0"})
    uptime_seconds: float = Field(..., description="Application uptime in seconds", json_schema_extra={"example": 142.5})
    timestamp: str = Field(..., description="Current ISO-8601 UTC timestamp", json_schema_extra={"example": "2026-08-29T14:06:03.123456Z"})

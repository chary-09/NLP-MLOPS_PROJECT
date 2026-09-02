from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request payload for sentiment classification."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw input text to classify for sentiment",
        json_schema_extra={"example": "The product was amazing!"},
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Input text must not be empty or blank whitespace.")
        return value


class PredictionResponse(BaseModel):
    """Response payload for sentiment prediction."""

    prediction_id: str = Field(
        ...,
        description="Unique UUID identifier for this prediction",
        json_schema_extra={"example": "8a7c2b4d-1e3f-45a6-9b8c-7d6e5f4a3b2c"},
    )
    text: str = Field(
        ...,
        description="Original input text submitted for analysis",
        json_schema_extra={"example": "The product was amazing!"},
    )
    sentiment: str = Field(
        ...,
        description="Predicted sentiment class (positive / negative)",
        json_schema_extra={"example": "positive"},
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
        json_schema_extra={"example": 0.9412},
    )
    model_version: str = Field(
        ...,
        description="Loaded model version identifier",
        json_schema_extra={"example": "0.1.0"},
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of inference and persistence",
        json_schema_extra={"example": "2026-08-31T10:45:00.123456Z"},
    )


class PredictionHistoryResponse(BaseModel):
    """Response schema for paginated prediction history read from database."""

    total: int = Field(..., description="Total count of stored predictions in database")
    limit: int = Field(..., description="Maximum number of items returned")
    offset: int = Field(..., description="Offset index for pagination")
    predictions: List[PredictionResponse] = Field(
        ..., description="List of stored prediction records"
    )


class HealthResponse(BaseModel):
    """Service and database health check response schema."""

    status: str = Field(..., description="Overall health status (healthy, degraded, unhealthy)")
    api: bool = Field(..., description="FastAPI server running status")
    model_loaded: bool = Field(..., description="Model loaded status")
    vectorizer_loaded: bool = Field(..., description="TF-IDF vectorizer loaded status")
    database_connected: bool = Field(..., description="Database connection status")
    model_version: str = Field(..., description="Loaded model version")
    timestamp: str = Field(..., description="Current ISO-8601 UTC timestamp")


class ModelInfoResponse(BaseModel):
    """Model and vectorizer configuration details schema."""

    model_name: str = Field(..., description="Name of selected classifier")
    model_version: str = Field(..., description="Model version")
    vectorizer: Dict[str, Any] = Field(..., description="TF-IDF vectorizer details")
    classes: List[str] = Field(..., description="Supported sentiment classes")
    training_metrics: Optional[Dict[str, Any]] = Field(None, description="Training evaluation metrics")


class MetricsResponse(BaseModel):
    """Metrics aggregated from persistent database records."""

    total_predictions: int = Field(..., description="Total predictions stored in database")
    sentiment_distribution: Dict[str, int] = Field(
        ..., description="Count of predictions per sentiment class"
    )
    average_confidence: float = Field(..., description="Average prediction confidence")
    timestamp: str = Field(..., description="Current ISO-8601 UTC timestamp")


# ---------------------------------------------------------------------------
# XAI — Explainable AI schemas (Day 3)
# ---------------------------------------------------------------------------


class ExplainRequest(BaseModel):
    """Request payload for POST /explain."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw input text to explain",
        json_schema_extra={"example": "The product was amazing but delivery was terrible."},
    )
    method: str = Field(
        default="lime",
        description="Explanation method: 'shap', 'lime', or 'both'",
        json_schema_extra={"example": "lime"},
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top features to return",
        json_schema_extra={"example": 10},
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Input text must not be empty or blank whitespace.")
        return value

    @field_validator("method")
    @classmethod
    def method_must_be_valid(cls, value: str) -> str:
        allowed = {"shap", "lime", "both"}
        if value.lower() not in allowed:
            raise ValueError(f"method must be one of {allowed}, got '{value}'")
        return value.lower()


class FeatureContribution(BaseModel):
    """A single token/feature and its contribution score."""

    feature: str = Field(..., description="Token or TF-IDF feature name")
    importance: float = Field(
        ...,
        description="Contribution score: positive = supports positive class, negative = supports negative class",
    )
    method: Optional[str] = Field(
        None,
        description="Explainer that produced this contribution (present when method='both')",
    )


class ExplainResponse(BaseModel):
    """Response payload for POST /explain."""

    prediction: str = Field(..., description="Predicted sentiment (positive / negative)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    model_version: str = Field(..., description="Model version used for prediction and explanation")
    method: str = Field(..., description="Explanation method used (shap / lime / both)")
    explanation: List[FeatureContribution] = Field(
        ...,
        description="Ranked list of feature contributions (positive importance = toward positive class)",
    )
    positive_words: List[str] = Field(
        ...,
        description="Words with positive contribution (push toward positive sentiment)",
    )
    negative_words: List[str] = Field(
        ...,
        description="Words with negative contribution (push toward negative sentiment)",
    )

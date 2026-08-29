from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request payload for single text sentiment classification."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw input text to analyze for sentiment",
        json_schema_extra={"example": "The product was amazing!"},
    )

    @field_validator("text")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Input text must not be empty or blank whitespace.")
        return value


class PredictionResponse(BaseModel):
    """Response payload for sentiment prediction."""

    prediction_id: str = Field(
        ...,
        description="Unique UUID identifier for this prediction",
        json_schema_extra={"example": "9f14798e-4a6f-402a-9be2-4a0fbde2138e"},
    )
    text: str = Field(
        ...,
        description="Original text submitted for analysis",
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
        description="Prediction confidence score between 0.0 and 1.0",
        json_schema_extra={"example": 0.94},
    )
    model_version: str = Field(
        ...,
        description="Active model version identifier",
        json_schema_extra={"example": "0.1.0"},
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of inference",
        json_schema_extra={"example": "2026-08-29T14:06:03.123456Z"},
    )


class PredictionHistoryResponse(BaseModel):
    """Response payload containing stored prediction records."""

    total: int = Field(..., description="Total number of stored predictions")
    limit: int = Field(..., description="Max number of items returned")
    offset: int = Field(..., description="Starting offset index")
    predictions: List[PredictionResponse] = Field(
        ..., description="List of recent prediction responses"
    )

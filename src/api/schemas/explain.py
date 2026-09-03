"""Pydantic schemas for Explainable AI (SHAP & LIME)."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


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

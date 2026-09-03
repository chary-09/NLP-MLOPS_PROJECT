from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class VectorizerInfo(BaseModel):
    """Details of the TF-IDF vectorizer configuration and vocabulary."""

    vectorizer_type: Optional[str] = Field(None, description="Class name of the vectorizer")
    type: Optional[str] = Field(None, description="Class name alias")
    max_features: Optional[int] = Field(None, description="Max vocabulary features")
    ngram_range: Optional[List[int]] = Field(None, description="Configured n-gram range [min, max]")
    vocabulary_size: Optional[int] = Field(None, description="Learned vocabulary size")
    lowercase: bool = Field(True, description="Lowercasing flag")


class ModelInfoResponse(BaseModel):
    """Model, vectorizer, and evaluation metadata response schema."""

    model_name: str = Field(..., description="Selected best model name", json_schema_extra={"example": "logistic_regression"})
    model_type: Optional[str] = Field(None, description="Classifier class name", json_schema_extra={"example": "LogisticRegression"})
    model_version: str = Field(..., description="Model version", json_schema_extra={"example": "0.1.0"})
    vectorizer: Union[VectorizerInfo, Dict[str, Any]] = Field(..., description="TF-IDF vectorizer details")
    classes: List[str] = Field(..., description="Sentiment classes supported by the model", json_schema_extra={"example": ["negative", "positive"]})
    training_metrics: Optional[Dict[str, Any]] = Field(None, description="Phase 1 evaluation metrics from training")
    artifact_paths: Optional[Dict[str, str]] = Field(None, description="File paths to model artifacts")

"""Pydantic schemas for MLOps monitoring responses and telemetry requests."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelPerformanceResponse(BaseModel):
    """Category 1: Model Performance Monitoring schema."""

    model_version: str = Field(..., description="Loaded model version")
    baseline_metrics: Dict[str, Any] = Field(..., description="Baseline metrics from Phase 1 training evaluation")
    production_metrics: Dict[str, Any] = Field(..., description="Production metrics calculated from verified ground-truth labels")
    can_calculate_production_metrics: bool = Field(..., description="True only if ground-truth labels are available")


class PredictionMonitoringResponse(BaseModel):
    """Category 2: Prediction Monitoring schema."""

    total_predictions: int = Field(..., description="Total prediction count in database")
    sentiment_distribution: Dict[str, int] = Field(..., description="Count of predictions per sentiment class")
    positive_percentage: float = Field(..., description="Percentage of positive predictions")
    negative_percentage: float = Field(..., description="Percentage of negative predictions")
    neutral_percentage: Optional[float] = Field(None, description="Neutral percentage if supported (null for binary model)")
    average_confidence: float = Field(..., description="Mean prediction confidence score")
    min_confidence: float = Field(..., description="Minimum prediction confidence score")
    max_confidence: float = Field(..., description="Maximum prediction confidence score")
    low_confidence_threshold: float = Field(..., description="Threshold used to identify low-confidence predictions")
    low_confidence_count: int = Field(..., description="Count of predictions below low-confidence threshold")
    low_confidence_percentage: float = Field(..., description="Percentage of predictions below low-confidence threshold")
    model_version: Optional[str] = Field(None, description="Active model version")


class SystemMetricsResponse(BaseModel):
    """Category 3: System Monitoring schema."""

    total_requests: int = Field(..., description="Total HTTP requests processed")
    prediction_requests: int = Field(..., description="Requests received for prediction endpoints")
    successful_requests: int = Field(..., description="HTTP status < 400")
    failed_requests: int = Field(..., description="HTTP status >= 400")
    api_error_count: int = Field(..., description="Total API errors encountered")
    error_rate_percentage: float = Field(..., description="Error rate percentage")
    average_latency_seconds: float = Field(..., description="Mean request latency in seconds")
    min_latency_seconds: float = Field(..., description="Minimum request latency in seconds")
    max_latency_seconds: float = Field(..., description="Maximum request latency in seconds")
    average_latency_ms: float = Field(..., description="Mean request latency in milliseconds")
    min_latency_ms: float = Field(..., description="Minimum request latency in milliseconds")
    max_latency_ms: float = Field(..., description="Maximum request latency in milliseconds")
    endpoint_request_counts: Dict[str, int] = Field(..., description="Request counts broken down by endpoint")


class DataDriftResponse(BaseModel):
    """Category 4: NLP Data Drift Monitoring schema."""

    drift_detected: bool = Field(..., description="True if statistical drift exceeds configured threshold")
    metric: str = Field(..., description="Statistical drift metric used (e.g. Kolmogorov-Smirnov & Vocabulary OOV)")
    score: float = Field(..., description="Computed drift divergence score")
    threshold: float = Field(..., description="Configured drift decision threshold")
    status: str = Field(..., description="Drift status (NORMAL, DRIFT_DETECTED, INSUFFICIENT_DATA)")
    reference_dataset: str = Field(..., description="Reference baseline dataset name")
    current_production_window: str = Field(..., description="Description of the evaluated production sample window")
    interpretation: str = Field(..., description="Human-readable drift diagnosis")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed statistical divergence parameters")


class MetricsResponse(BaseModel):
    """Unified MLOps monitoring summary schema."""

    # Baseline & backward-compatible fields
    total_predictions: int = Field(..., description="Total predictions stored in database")
    sentiment_distribution: Dict[str, int] = Field(..., description="Count of predictions per sentiment class")
    average_confidence: float = Field(..., description="Average prediction confidence")
    timestamp: str = Field(..., description="Current UTC timestamp (ISO-8601)")

    # Extended monitoring telemetry
    model_version: Optional[str] = Field("0.1.0", description="Active model version")
    status: Optional[str] = Field("NORMAL", description="Overall health and alert status (NORMAL, WARNING, CRITICAL, DRIFT_DETECTED)")
    alerts: Optional[List[str]] = Field(default_factory=list, description="Active threshold alerts")
    thresholds: Optional[Dict[str, Any]] = Field(None, description="Configured threshold parameters")
    model_performance: Optional[Dict[str, Any]] = Field(None, description="Category 1: Model performance metrics")
    prediction_monitoring: Optional[Dict[str, Any]] = Field(None, description="Category 2: Prediction telemetry")
    system_monitoring: Optional[Dict[str, Any]] = Field(None, description="Category 3: System request & latency metrics")
    data_drift: Optional[Dict[str, Any]] = Field(None, description="Category 4: NLP data drift report")


class GroundTruthItem(BaseModel):
    """A prediction label paired with verified ground-truth."""

    prediction: str = Field(..., description="Predicted sentiment (positive / negative)")
    ground_truth: str = Field(..., description="Actual ground-truth sentiment (positive / negative)")


class GroundTruthEvaluationRequest(BaseModel):
    """Payload to evaluate production performance when ground-truth labels are acquired."""

    evaluations: List[GroundTruthItem] = Field(
        ...,
        min_length=1,
        description="List of paired predictions and ground truth labels",
    )


class GroundTruthEvaluationResponse(BaseModel):
    """Evaluation result for production performance with real labels."""

    status: str = Field("AVAILABLE", description="Production evaluation status")
    accuracy: float = Field(..., description="Actual production accuracy")
    precision: float = Field(..., description="Actual production precision")
    recall: float = Field(..., description="Actual production recall")
    f1_score: float = Field(..., description="Actual production F1-score")
    sample_count: int = Field(..., description="Number of evaluated samples")
    evaluated_at: str = Field(..., description="Evaluation timestamp")

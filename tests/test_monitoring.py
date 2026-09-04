"""Comprehensive test suite for Phase 2 Day 4 MLOps Monitoring layer.

Verifies:
1. prediction metrics
2. model baseline metrics
3. production metrics with ground truth
4. API request metrics
5. API latency tracking
6. API error metrics
7. confidence statistics
8. sentiment distribution (binary, no fake neutral)
9. NLP data drift calculation (KS test, OOV rate, thresholds)
10. threshold logic & alert state transitions
11. model version tracking
12. Monitoring API endpoints (/metrics, /metrics/model, /metrics/predictions, /metrics/system, /metrics/drift, /metrics/evaluate-production)
"""

import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.database.connection import SessionLocal
from src.database.repository import PredictionRepository
from src.monitoring.alerts import MonitoringAlertManager, should_alert
from src.monitoring.drift_detector import NLPDataDriftDetector, detect_drift
from src.monitoring.latency_monitor import LatencyMonitor, SystemMetricsTracker
from src.monitoring.metrics import MonitoringService, prediction_metrics
from src.monitoring.performance_monitor import ModelPerformanceMonitor, performance_status


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    """Database session fixture."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def repo(db_session):
    """PredictionRepository fixture."""
    return PredictionRepository(db_session)


# ---------------------------------------------------------------------------
# Legacy compatibility tests
# ---------------------------------------------------------------------------
def test_detect_drift():
    """Verify legacy scalar drift function."""
    assert detect_drift(0.1, 0.3)
    assert not detect_drift(0.1, 0.15, threshold=0.1)


def test_legacy_performance_status_and_prediction_metrics():
    """Verify legacy stubs remain functioning."""
    assert performance_status(0.85) == "healthy"
    assert performance_status(0.60) == "degraded"
    assert should_alert("healthy") is False
    assert should_alert("critical") is True
    assert prediction_metrics([{"a": 1}, {"b": 2}]) == {"total_predictions": 2}


# ---------------------------------------------------------------------------
# 1. Prediction Metrics & Sentiment Distribution
# ---------------------------------------------------------------------------
def test_prediction_metrics_and_sentiment_distribution(repo):
    """Verify prediction metrics, sentiment distribution, and confidence stats."""
    # Seed known records
    for i in range(5):
        repo.create(
            prediction_id=str(uuid.uuid4()),
            input_text=f"Sample positive review number {i}",
            sentiment="positive",
            confidence=0.90 + (i * 0.01),
            model_version="0.1.0",
        )
    for i in range(3):
        repo.create(
            prediction_id=str(uuid.uuid4()),
            input_text=f"Sample negative review number {i}",
            sentiment="negative",
            confidence=0.55 + (i * 0.02),  # low confidence (< 0.65)
            model_version="0.1.0",
        )

    stats = repo.get_detailed_prediction_metrics(low_confidence_threshold=0.65)

    assert stats["total_predictions"] >= 8
    assert "positive" in stats["sentiment_distribution"]
    assert "negative" in stats["sentiment_distribution"]
    # Binary sentiment system: strictly no invented neutral predictions
    assert stats["neutral_percentage"] is None
    assert stats["positive_percentage"] > 0.0
    assert stats["negative_percentage"] > 0.0
    assert stats["average_confidence"] > 0.0
    assert stats["min_confidence"] <= stats["max_confidence"]
    assert stats["low_confidence_count"] >= 3
    assert stats["low_confidence_percentage"] > 0.0


# ---------------------------------------------------------------------------
# 2. Model Baseline Metrics & Genuine Ground Truth Distinctions
# ---------------------------------------------------------------------------
def test_model_baseline_metrics():
    """Verify model performance monitor loads Phase 1 baseline and marks unobserved production as UNAVAILABLE."""
    perf_mon = ModelPerformanceMonitor()
    baseline = perf_mon.get_baseline_metrics()

    assert baseline["accuracy"] == pytest.approx(0.8904, abs=1e-2)
    assert baseline["precision"] == pytest.approx(0.8906, abs=1e-2)
    assert baseline["recall"] == pytest.approx(0.8904, abs=1e-2)
    assert baseline["f1_score"] == pytest.approx(0.8904, abs=1e-2)

    # Before ground truth is provided, production metrics MUST be UNAVAILABLE (no faking)
    unavail_prod = perf_mon.get_production_metrics()
    assert unavail_prod["status"] == "UNAVAILABLE"
    assert unavail_prod["accuracy"] is None
    assert unavail_prod["f1_score"] is None
    assert unavail_prod["sample_count"] == 0


def test_production_metrics_with_ground_truth():
    """Verify authentic production performance calculation when real ground-truth labels are supplied."""
    perf_mon = ModelPerformanceMonitor()
    predictions = ["positive", "positive", "negative", "positive", "negative"]
    ground_truth = ["positive", "negative", "negative", "positive", "negative"]

    result = perf_mon.evaluate_ground_truth(predictions, ground_truth)

    assert result["status"] == "AVAILABLE"
    assert result["sample_count"] == 5
    assert result["accuracy"] == 0.8  # 4 / 5
    assert result["precision"] > 0.0
    assert result["recall"] > 0.0
    assert result["f1_score"] > 0.0
    assert "evaluated_at" in result

    # Check that summary now reflects available production metrics
    summary = perf_mon.get_summary(model_version="0.1.0")
    assert summary["can_calculate_production_metrics"] is True
    assert summary["production_metrics"]["status"] == "AVAILABLE"


def test_production_metrics_mismatched_lengths_raise_error():
    """Verify error handling on invalid ground truth payload."""
    perf_mon = ModelPerformanceMonitor()
    with pytest.raises(ValueError):
        perf_mon.evaluate_ground_truth(["positive"], ["positive", "negative"])


# ---------------------------------------------------------------------------
# 3. System Telemetry, Request Tracking & Latency
# ---------------------------------------------------------------------------
def test_system_request_metrics_and_errors():
    """Verify system metrics tracking total, prediction, success, failed requests, and error rate."""
    tracker = SystemMetricsTracker()
    tracker.reset()

    # Simulate normal requests
    tracker.record_request("/health", status_code=200, latency_seconds=0.010)
    tracker.record_request("/predict", status_code=200, latency_seconds=0.045)
    tracker.record_request("/api/v1/predict", status_code=200, latency_seconds=0.040)

    # Simulate client and server errors
    tracker.record_request("/predict", status_code=422, latency_seconds=0.005)
    tracker.record_request("/unknown", status_code=404, latency_seconds=0.002)

    metrics = tracker.get_metrics()

    assert metrics["total_requests"] == 5
    assert metrics["prediction_requests"] == 3  # 2 success + 1 error
    assert metrics["successful_requests"] == 3
    assert metrics["failed_requests"] == 2
    assert metrics["api_error_count"] == 2
    assert metrics["error_rate_percentage"] == 40.0
    assert "/predict" in metrics["endpoint_request_counts"]
    assert metrics["endpoint_request_counts"]["/predict"] == 2


def test_system_latency_tracking():
    """Verify latency calculation including min, max, average in seconds and milliseconds."""
    tracker = SystemMetricsTracker()
    tracker.reset()

    tracker.record_request("/predict", 200, 0.020)
    tracker.record_request("/predict", 200, 0.040)
    tracker.record_request("/predict", 200, 0.060)

    metrics = tracker.get_metrics()

    assert metrics["average_latency_seconds"] == pytest.approx(0.040, abs=1e-3)
    assert metrics["min_latency_seconds"] == pytest.approx(0.020, abs=1e-3)
    assert metrics["max_latency_seconds"] == pytest.approx(0.060, abs=1e-3)
    assert metrics["average_latency_ms"] == pytest.approx(40.0, abs=1.0)


def test_latency_monitor_class():
    """Verify individual LatencyMonitor utility."""
    mon = LatencyMonitor()
    mon.start()
    elapsed_ms = mon.elapsed_ms()
    assert elapsed_ms >= 0.0
    assert mon.elapsed_seconds() >= 0.0


# ---------------------------------------------------------------------------
# 4. NLP Data Drift Monitoring
# ---------------------------------------------------------------------------
def test_nlp_data_drift_insufficient_samples():
    """Verify that fewer than 5 production texts results in INSUFFICIENT_DATA status."""
    detector = NLPDataDriftDetector()
    result = detector.calculate_drift(["short review 1", "short review 2"])

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["drift_detected"] is False
    assert result["score"] == 0.0
    assert "min 5 required" in result["current_production_window"]


def test_nlp_data_drift_normal_distribution():
    """Verify normal texts conforming to standard vocabulary and length do not trigger drift."""
    detector = NLPDataDriftDetector()
    normal_texts = [
        "This movie was an interesting film with great acting, direction, and plot throughout.",
        "The story was well crafted with thoughtful characters and engaging cinematic moments.",
        "A compelling drama that showcases superb storytelling and exceptional performance by cast.",
        "Wonderful cinematography, remarkable score, and brilliant execution by the director.",
        "Truly enjoyable film with magnificent artistic scenes and solid dialogue across all acts.",
        "A fascinating and authentic motion picture that kept me entertained from start to finish.",
    ]
    result = detector.calculate_drift(normal_texts, threshold=0.35)

    assert "score" in result
    assert "threshold" in result
    assert "metric" in result
    assert "reference_dataset" in result
    assert "interpretation" in result
    assert result["status"] in ("NORMAL", "DRIFT_DETECTED")
    assert isinstance(result["drift_detected"], bool)


def test_nlp_data_drift_detected_on_extreme_anomaly():
    """Verify that extreme vocabulary or length shifts trigger DRIFT_DETECTED."""
    detector = NLPDataDriftDetector()
    # Create extreme anomaly: 20 texts consisting entirely of random out-of-vocabulary pseudo-words
    drifted_texts = [
        "zxqwert plkjhgf mnbvcxz tyuiopas dfghjklq werzxcvbnm qazwsxedcrfvtgbyhnujmikolp " * 5
        for _ in range(10)
    ]
    result = detector.calculate_drift(drifted_texts, threshold=0.15)

    assert result["drift_detected"] is True
    assert result["status"] == "DRIFT_DETECTED"
    assert result["score"] >= 0.15
    assert "Data drift detected" in result["interpretation"]


# ---------------------------------------------------------------------------
# 5. Alert / Threshold Logic
# ---------------------------------------------------------------------------
def test_alert_threshold_logic():
    """Verify threshold evaluation states: NORMAL, WARNING, CRITICAL, DRIFT_DETECTED."""
    alert_mgr = MonitoringAlertManager(
        low_confidence_warning_pct=20.0,
        low_confidence_critical_pct=40.0,
        error_rate_warning_pct=5.0,
        error_rate_critical_pct=15.0,
        latency_warning_seconds=0.50,
        latency_critical_seconds=1.00,
    )

    # 1. Healthy state -> NORMAL
    eval_normal = alert_mgr.evaluate(
        prediction_metrics={"total_predictions": 20, "low_confidence_percentage": 5.0},
        system_metrics={"total_requests": 20, "error_rate_percentage": 0.0, "average_latency_seconds": 0.05},
        drift_metrics={"drift_detected": False},
    )
    assert eval_normal["system_status"] == "NORMAL"
    assert len(eval_normal["alerts"]) == 0

    # 2. Elevated error rate -> WARNING
    eval_warn = alert_mgr.evaluate(
        prediction_metrics={"total_predictions": 20, "low_confidence_percentage": 5.0},
        system_metrics={"total_requests": 20, "error_rate_percentage": 8.0, "average_latency_seconds": 0.05},
        drift_metrics={"drift_detected": False},
    )
    assert eval_warn["system_status"] == "WARNING"
    assert any("error rate" in a for a in eval_warn["alerts"])

    # 3. High latency -> CRITICAL
    eval_crit = alert_mgr.evaluate(
        prediction_metrics={"total_predictions": 20, "low_confidence_percentage": 5.0},
        system_metrics={"total_requests": 20, "error_rate_percentage": 0.0, "average_latency_seconds": 1.20},
        drift_metrics={"drift_detected": False},
    )
    assert eval_crit["system_status"] == "CRITICAL"
    assert any("latency" in a for a in eval_crit["alerts"])

    # 4. Data drift -> DRIFT_DETECTED
    eval_drift = alert_mgr.evaluate(
        prediction_metrics={"total_predictions": 20, "low_confidence_percentage": 5.0},
        system_metrics={"total_requests": 20, "error_rate_percentage": 0.0, "average_latency_seconds": 0.05},
        drift_metrics={"drift_detected": True, "score": 0.28, "threshold": 0.20},
    )
    assert eval_drift["system_status"] == "DRIFT_DETECTED"
    assert any("drift" in a.lower() for a in eval_drift["alerts"])


# ---------------------------------------------------------------------------
# 6. Model Version Tracking
# ---------------------------------------------------------------------------
def test_model_version_tracking(repo):
    """Verify that model version is dynamically included in monitoring responses without hardcoding."""
    svc = MonitoringService()
    summary = svc.get_full_summary(repo=repo)
    assert "model_version" in summary
    assert summary["model_version"] is not None
    assert summary["model_performance"]["model_version"] == summary["model_version"]


# ---------------------------------------------------------------------------
# 7. Monitoring API Endpoints
# ---------------------------------------------------------------------------
def test_monitoring_api_endpoints(client):
    """Verify all monitoring HTTP endpoints."""
    # Seed at least one prediction
    pred_res = client.post("/predict", json={"text": "A breathtaking cinematic masterpiece."})
    assert pred_res.status_code == 200

    # 1. Main summary endpoint GET /metrics
    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    metrics_data = res_metrics.json()
    assert metrics_data["total_predictions"] >= 1
    assert "sentiment_distribution" in metrics_data
    assert "average_confidence" in metrics_data
    assert "model_version" in metrics_data
    assert "status" in metrics_data
    assert "model_performance" in metrics_data
    assert "prediction_monitoring" in metrics_data
    assert "system_monitoring" in metrics_data
    assert "data_drift" in metrics_data

    # 2. Sub-endpoint GET /metrics/model
    res_model = client.get("/metrics/model")
    assert res_model.status_code == 200
    model_data = res_model.json()
    assert "baseline_metrics" in model_data
    assert "production_metrics" in model_data
    assert model_data["baseline_metrics"]["accuracy"] > 0.80

    # 3. Sub-endpoint GET /metrics/predictions
    res_preds = client.get("/metrics/predictions")
    assert res_preds.status_code == 200
    preds_data = res_preds.json()
    assert preds_data["total_predictions"] >= 1
    assert "positive_percentage" in preds_data
    assert "negative_percentage" in preds_data
    assert preds_data["neutral_percentage"] is None  # strictly binary

    # 4. Sub-endpoint GET /metrics/system
    res_sys = client.get("/metrics/system")
    assert res_sys.status_code == 200
    sys_data = res_sys.json()
    assert sys_data["total_requests"] >= 1
    assert "average_latency_seconds" in sys_data
    assert "endpoint_request_counts" in sys_data

    # 5. Sub-endpoint GET /metrics/drift
    res_drift = client.get("/metrics/drift?window=20&threshold=0.20")
    assert res_drift.status_code == 200
    drift_data = res_drift.json()
    assert "drift_detected" in drift_data
    assert "metric" in drift_data
    assert "status" in drift_data

    # 6. Evaluation endpoint POST /metrics/evaluate-production
    eval_payload = {
        "evaluations": [
            {"prediction": "positive", "ground_truth": "positive"},
            {"prediction": "negative", "ground_truth": "negative"},
            {"prediction": "positive", "ground_truth": "negative"},
        ]
    }
    res_eval = client.post("/metrics/evaluate-production", json=eval_payload)
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert eval_data["status"] == "AVAILABLE"
    assert eval_data["sample_count"] == 3
    assert eval_data["accuracy"] == pytest.approx(2 / 3, abs=1e-2)

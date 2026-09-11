"""Dashboard configuration and API connectivity helpers.

Centralizes all dashboard settings, environment variable resolution,
and endpoint builders to guarantee that no URLs are hard-coded in the UI.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

# Base URL for the FastAPI backend service
DEFAULT_API_BASE_URL = "http://localhost:8000"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")

# Timeout in seconds for backend HTTP requests
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("DASHBOARD_REQUEST_TIMEOUT", "3.0"))

# Page settings
PAGE_TITLE = "SentimentOps | Production NLP Platform"
PAGE_ICON = "⚡"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Application Metadata
APP_NAME = "SentimentOps"
APP_TAGLINE = "Production NLP Sentiment Analysis & MLOps Platform"
APP_VERSION = "1.0.0"
PHASE = "Phase 3: Visual MLOps Dashboard"

# API Route Endpoints
ENDPOINTS = {
    "health": "/health",
    "model_info": "/model-info",
    "predict": "/predict",
    "predictions": "/predictions",
    "explain": "/explain",
    "metrics": "/metrics",
    "metrics_model": "/metrics/model",
    "metrics_predictions": "/metrics/predictions",
    "metrics_system": "/metrics/system",
    "metrics_drift": "/metrics/drift",
    "metrics_evaluate": "/metrics/evaluate-production",
}


def get_api_url(endpoint_key: str) -> str:
    """Build full URL for an API endpoint without hard-coding."""
    endpoint_path = ENDPOINTS.get(endpoint_key, endpoint_key)
    if not endpoint_path.startswith("/"):
        endpoint_path = f"/{endpoint_path}"
    return f"{API_BASE_URL}{endpoint_path}"


def fetch_api_health(timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Tuple[bool, Dict[str, Any], float]:
    """Ping the FastAPI /health endpoint safely.
    
    Returns:
        Tuple of (is_healthy, response_data, latency_ms)
    """
    url = get_api_url("health")
    start = time.perf_counter()
    try:
        response = requests.get(url, timeout=timeout)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if response.status_code == 200:
            data = response.json()
            is_healthy = data.get("status") in ("healthy", "ok")
            return is_healthy, data, latency_ms
        return False, {"error": f"HTTP {response.status_code}", "raw": response.text[:200]}, latency_ms
    except requests.exceptions.ConnectionError:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"error": "Connection refused - Backend service is offline", "url": url}, latency_ms
    except requests.exceptions.Timeout:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"error": f"Connection timed out after {timeout}s", "url": url}, latency_ms
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"error": str(exc), "url": url}, latency_ms


def fetch_model_info(timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Tuple[bool, Dict[str, Any]]:
    """Fetch model architecture and vectorizer metadata from /model-info.
    
    Returns:
        Tuple of (success, response_data)
    """
    url = get_api_url("model_info")
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"HTTP {response.status_code}"}
    except Exception as exc:
        return False, {"error": str(exc)}


def fetch_metrics(timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Tuple[bool, Dict[str, Any]]:
    """Fetch full MLOps telemetry from /metrics safely.
    
    Returns:
        Tuple of (success, metrics_dict)
    """
    url = get_api_url("metrics")
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"HTTP {response.status_code}"}
    except Exception as exc:
        return False, {"error": str(exc)}


def fetch_recent_predictions(
    limit: int = 5,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Fetch recent prediction history from /predictions.
    
    Returns:
        Tuple of (success, predictions_list, total_count)
    """
    url = f"{get_api_url('predictions')}?limit={limit}"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("predictions", []), data.get("total", 0)
        return False, [], 0
    except Exception:
        return False, [], 0


def make_prediction(
    text: str,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Tuple[bool, Dict[str, Any], float]:
    """Execute POST /predict endpoint to classify sentiment.
    
    Returns:
        Tuple of (success, response_dict, latency_ms)
    """
    if not text or not text.strip():
        return False, {"error": "Input text cannot be empty."}, 0.0

    url = get_api_url("predict")
    start = time.perf_counter()
    try:
        response = requests.post(
            url,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if response.status_code == 200:
            data = response.json()
            data["latency_ms"] = latency_ms
            return True, data, latency_ms
        elif response.status_code == 422:
            err_detail = response.json().get("detail", "Validation error")
            return False, {"error": f"Validation Error (422): {err_detail}"}, latency_ms
        return False, {"error": f"HTTP {response.status_code}: {response.text[:200]}"}, latency_ms
    except requests.exceptions.ConnectionError:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"error": "Backend service is offline. Please start FastAPI server."}, latency_ms
    except requests.exceptions.Timeout:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"error": f"Request timed out after {timeout}s."}, latency_ms
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"error": str(exc)}, latency_ms


def fetch_predictions_history(
    limit: int = 50,
    offset: int = 0,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Tuple[bool, Dict[str, Any], float]:
    """Fetch stored prediction records from GET /predictions.
    
    Returns:
        Tuple of (success, response_dict, latency_ms)
    """
    url = f"{get_api_url('predictions')}?limit={limit}&offset={offset}"
    start = time.perf_counter()
    try:
        response = requests.get(url, timeout=timeout)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if response.status_code == 200:
            data = response.json()
            return True, data, latency_ms
        return False, {"error": f"HTTP {response.status_code}: {response.text[:200]}"}, latency_ms
    except requests.exceptions.ConnectionError:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"error": "Backend service is offline. Please start FastAPI server."}, latency_ms
    except requests.exceptions.Timeout:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"error": f"Request timed out after {timeout}s."}, latency_ms
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, {"error": str(exc)}, latency_ms



def fetch_overview_data(timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Dict[str, Any]:
    """Consolidated aggregator for Day 2 Overview page.
    
    Extracts all metrics, system health, drift, alerts, and recent predictions
    while gracefully handling all failure modes without crashing.
    """
    api_healthy, health_data, latency_ms = fetch_api_health(timeout=timeout)

    default_payload = {
        "api_available": False,
        "api_status": "OFFLINE",
        "latency_ms": latency_ms,
        "total_predictions": 0,
        "positive_pct": 0.0,
        "negative_pct": 0.0,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "average_confidence": 0.0,
        "drift_status": "UNKNOWN",
        "drift_detected": False,
        "model_name": "LogisticRegression",
        "model_version": "N/A",
        "model_info": {},
        "recent_predictions": [],
        "recent_alerts": [],
        "system_health": {
            "api": False,
            "database_connected": False,
            "model_loaded": False,
            "vectorizer_loaded": False,
        },
        "classes": ["negative", "positive"],
        "is_binary": True,
        "error_message": health_data.get("error", "Backend service offline") if not api_healthy else None,
    }

    if not api_healthy:
        return default_payload

    # API is reachable — query other endpoints
    info_ok, model_info = fetch_model_info(timeout=timeout)
    metrics_ok, metrics_data = fetch_metrics(timeout=timeout)
    preds_ok, recent_preds, total_preds_count = fetch_recent_predictions(limit=5, timeout=timeout)

    model_name = model_info.get("model_name", "LogisticRegression") if info_ok else "LogisticRegression"
    model_version = health_data.get("model_version", "0.1.0")
    classes = model_info.get("classes", ["negative", "positive"]) if info_ok else ["negative", "positive"]
    # Ensure binary classes (negative, positive)
    is_binary = len(classes) == 2 and "neutral" not in [str(c).lower() for c in classes]

    # Metrics parsing
    pred_monitoring = metrics_data.get("prediction_monitoring", {}) if metrics_ok else {}
    total_preds = pred_monitoring.get("total_predictions", total_preds_count if preds_ok else 0)
    pos_pct = round(float(pred_monitoring.get("positive_percentage", 0.0)), 2)
    neg_pct = round(float(pred_monitoring.get("negative_percentage", 0.0)), 2)
    avg_conf = round(float(pred_monitoring.get("average_confidence", 0.0)), 4)

    # Model performance baseline metrics
    model_perf = metrics_data.get("model_performance", {}) if metrics_ok else {}
    baseline = model_perf.get("baseline_metrics", {})
    prod_metrics = model_perf.get("production_metrics", {})
    
    # Priority: genuine production metrics if available, otherwise Phase 1 baseline
    accuracy = prod_metrics.get("accuracy") if prod_metrics.get("accuracy") is not None else baseline.get("accuracy")
    precision = prod_metrics.get("precision") if prod_metrics.get("precision") is not None else baseline.get("precision")
    recall = prod_metrics.get("recall") if prod_metrics.get("recall") is not None else baseline.get("recall")
    f1 = (
        prod_metrics.get("f1_score") or prod_metrics.get("f1")
        if prod_metrics.get("f1_score") or prod_metrics.get("f1")
        else (baseline.get("f1_score") or baseline.get("f1"))
    )

    # Drift Status
    drift_data = metrics_data.get("data_drift", {}) if metrics_ok else {}
    drift_status = drift_data.get("status", "NORMAL")
    drift_detected = drift_data.get("drift_detected", False)

    # Alerts list
    recent_alerts = metrics_data.get("alerts", []) if metrics_ok else []

    # System health items
    system_health = {
        "api": health_data.get("api", True),
        "database_connected": health_data.get("database_connected", False),
        "model_loaded": health_data.get("model_loaded", False),
        "vectorizer_loaded": health_data.get("vectorizer_loaded", False),
    }

    return {
        "api_available": True,
        "api_status": "ONLINE",
        "latency_ms": latency_ms,
        "total_predictions": total_preds,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_confidence": avg_conf,
        "drift_status": drift_status,
        "drift_detected": drift_detected,
        "model_name": model_name,
        "model_version": model_version,
        "model_info": model_info if info_ok else {},
        "recent_predictions": recent_preds if preds_ok else [],
        "recent_alerts": recent_alerts,
        "system_health": system_health,
        "classes": classes,
        "is_binary": is_binary,
        "error_message": None,
    }

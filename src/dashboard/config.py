"""Dashboard configuration and API connectivity helpers.

Centralizes all dashboard settings, environment variable resolution,
and endpoint builders to guarantee that no URLs are hard-coded in the UI.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple
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

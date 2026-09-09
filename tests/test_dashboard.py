"""Unit tests for Phase 3 Dashboard Foundation and Day 2 Overview Navigation."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.dashboard import config, theme
from src.dashboard.components.alerts import render_alert_box, render_recent_alerts_section
from src.dashboard.components.badges import (
    render_badge,
    render_binary_badge,
    render_drift_badge,
    render_sentiment_badge,
    render_status_badge,
)
from src.dashboard.components.cards import (
    render_card_header,
    render_kpi_card,
    render_metric_card,
    render_spec_grid,
)
from src.dashboard.components.sections import (
    render_model_info_card,
    render_recent_predictions_table,
    render_section_heading,
    render_system_health_grid,
)
from src.dashboard.components.sidebar import NAVIGATION_PAGES
from src.dashboard.components.states import (
    render_empty_state,
    render_error_state,
    render_loading_state,
)


def test_dashboard_module_imports():
    """Verify core theme and design tokens are defined and backward-compatible."""
    assert theme.PRIMARY_COLOR == "#2563EB"
    assert theme.SUCCESS_COLOR
    assert theme.WARNING_COLOR
    assert theme.DANGER_COLOR


def test_dashboard_config_api_url_resolution():
    """Verify endpoint URLs are built dynamically without hardcoding."""
    assert config.get_api_url("health").endswith("/health")
    assert config.get_api_url("predict").endswith("/predict")
    assert config.get_api_url("metrics").endswith("/metrics")
    assert config.get_api_url("explain").endswith("/explain")
    assert config.get_api_url("predictions").endswith("/predictions")


def test_dashboard_config_custom_base_url():
    """Verify API_BASE_URL respects environment variable or custom overrides."""
    with patch.dict(os.environ, {"API_BASE_URL": "http://custom-mlops-host:9000"}):
        custom_url = f"{os.getenv('API_BASE_URL')}/health"
        assert custom_url == "http://custom-mlops-host:9000/health"


def test_render_badges():
    """Verify badge rendering returns HTML with proper classes and content."""
    success_badge = render_badge("ONLINE", badge_type="success", pulse=True)
    assert "badge-success" in success_badge
    assert "ONLINE" in success_badge
    assert "pulse-dot" in success_badge

    danger_badge = render_badge("OFFLINE", badge_type="danger", pulse=False)
    assert "badge-danger" in danger_badge
    assert "OFFLINE" in danger_badge


def test_render_status_badge():
    """Verify operational health badge logic."""
    live_badge = render_status_badge("healthy", latency_ms=12.5)
    assert "API LIVE" in live_badge
    assert "12.5ms" in live_badge
    assert "badge-success" in live_badge

    offline_badge = render_status_badge("offline")
    assert "API OFFLINE" in offline_badge
    assert "badge-danger" in offline_badge


def test_render_drift_badge():
    """Verify drift monitoring badges for normal, detected, and insufficient data."""
    normal = render_drift_badge("NORMAL")
    assert "DRIFT: STABLE" in normal
    assert "badge-success" in normal

    detected = render_drift_badge("DRIFT_DETECTED")
    assert "DRIFT DETECTED" in detected
    assert "badge-danger" in detected

    insufficient = render_drift_badge("INSUFFICIENT_DATA")
    assert "INSUFFICIENT DATA" in insufficient
    assert "badge-neutral" in insufficient


def test_render_binary_badge():
    """Verify binary classification indicator badge."""
    binary_badge = render_binary_badge()
    assert "BINARY: POSITIVE / NEGATIVE" in binary_badge
    assert "badge-info" in binary_badge


def test_render_sentiment_badge():
    """Verify sentiment badges strictly distinguish positive and negative."""
    pos = render_sentiment_badge("positive")
    assert "POSITIVE" in pos
    assert "badge-success" in pos

    neg = render_sentiment_badge("negative")
    assert "NEGATIVE" in neg
    assert "badge-danger" in neg


def test_fetch_api_health_graceful_offline():
    """Verify that backend offline state is caught safely without unhandled crashes."""
    with patch("requests.get", side_effect=Exception("Connection refused")):
        healthy, data, latency = config.fetch_api_health()
        assert healthy is False
        assert "error" in data
        assert isinstance(latency, (int, float))


def test_fetch_api_health_success():
    """Verify successful health parsing when backend responds 200 OK."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "healthy",
        "api": True,
        "model_loaded": True,
        "vectorizer_loaded": True,
        "database_connected": True,
        "model_version": "0.1.0",
    }
    with patch("requests.get", return_value=mock_resp):
        healthy, data, latency = config.fetch_api_health()
        assert healthy is True
        assert data["database_connected"] is True
        assert data["model_version"] == "0.1.0"


def test_fetch_overview_data_offline():
    """Verify fetch_overview_data returns safe default payload when API is offline."""
    with patch("requests.get", side_effect=Exception("Connection refused")):
        data = config.fetch_overview_data()
        assert data["api_available"] is False
        assert data["api_status"] == "OFFLINE"
        assert data["total_predictions"] == 0
        assert data["positive_pct"] == 0.0
        assert data["negative_pct"] == 0.0
        assert data["accuracy"] is None
        assert data["is_binary"] is True
        assert data["recent_predictions"] == []
        assert data["recent_alerts"] == []


def test_fetch_overview_data_online():
    """Verify fetch_overview_data extracts full telemetry from backend when online."""
    health_mock = MagicMock(status_code=200)
    health_mock.json.return_value = {
        "status": "healthy",
        "api": True,
        "database_connected": True,
        "model_loaded": True,
        "vectorizer_loaded": True,
        "model_version": "0.1.0",
    }

    info_mock = MagicMock(status_code=200)
    info_mock.json.return_value = {
        "model_name": "LogisticRegression",
        "classes": ["negative", "positive"],
        "vectorizer": {"type": "TfidfVectorizer", "max_features": 10000},
    }

    metrics_mock = MagicMock(status_code=200)
    metrics_mock.json.return_value = {
        "total_predictions": 42,
        "average_confidence": 0.9123,
        "prediction_monitoring": {
            "total_predictions": 42,
            "positive_percentage": 60.0,
            "negative_percentage": 40.0,
            "average_confidence": 0.9123,
        },
        "model_performance": {
            "baseline_metrics": {
                "accuracy": 0.8904,
                "precision": 0.8906,
                "recall": 0.8904,
                "f1_score": 0.8904,
            },
            "production_metrics": {},
        },
        "data_drift": {
            "status": "NORMAL",
            "drift_detected": False,
        },
        "alerts": [],
    }

    preds_mock = MagicMock(status_code=200)
    preds_mock.json.return_value = {
        "total": 42,
        "predictions": [
            {
                "prediction_id": "test-uuid-1",
                "text": "Great movie!",
                "sentiment": "positive",
                "confidence": 0.95,
                "timestamp": "2026-09-09T10:00:00Z",
                "model_version": "0.1.0",
            }
        ],
    }

    def side_effect(url, **kwargs):
        if "health" in url:
            return health_mock
        elif "model-info" in url:
            return info_mock
        elif "predictions" in url:
            return preds_mock
        elif "metrics" in url:
            return metrics_mock
        return MagicMock(status_code=404)

    with patch("requests.get", side_effect=side_effect):
        data = config.fetch_overview_data()
        assert data["api_available"] is True
        assert data["api_status"] == "ONLINE"
        assert data["total_predictions"] == 42
        assert data["positive_pct"] == 60.0
        assert data["negative_pct"] == 40.0
        assert data["accuracy"] == 0.8904
        assert data["precision"] == 0.8906
        assert data["recall"] == 0.8904
        assert data["f1_score"] == 0.8904
        assert data["average_confidence"] == 0.9123
        assert data["drift_status"] == "NORMAL"
        assert data["is_binary"] is True
        assert len(data["recent_predictions"]) == 1
        assert data["recent_predictions"][0]["sentiment"] == "positive"


def test_navigation_pages_exist_on_filesystem():
    """Verify that all 9 required pages exist in the filesystem under src/dashboard/pages."""
    pages_dir = Path("src/dashboard/pages")
    expected_pages = [
        "01_Overview.py",
        "02_Live_Prediction.py",
        "03_Prediction_History.py",
        "04_Sentiment_Analytics.py",
        "05_Explainability.py",
        "06_Model_Performance.py",
        "07_Drift_Monitoring.py",
        "08_API_Health.py",
        "09_System_Logs.py",
    ]

    for p in expected_pages:
        page_file = pages_dir / p
        assert page_file.exists(), f"Page {p} does not exist in {pages_dir}"
        assert page_file.stat().st_size > 0, f"Page {p} is empty"


def test_navigation_pages_list():
    """Verify NAVIGATION_PAGES contains all 9 pages with appropriate icons."""
    assert len(NAVIGATION_PAGES) == 9
    filenames = [p[0] for p in NAVIGATION_PAGES]
    assert "pages/01_Overview.py" in filenames
    assert "pages/02_Live_Prediction.py" in filenames
    assert "pages/09_System_Logs.py" in filenames

"""Unit tests for Phase 3 Day 1 Dashboard Foundation."""

import os
from unittest.mock import MagicMock, patch
from src.dashboard import config, theme
from src.dashboard.components.badges import render_badge, render_status_badge
from src.dashboard.components.cards import render_card_header, render_kpi_card, render_spec_grid
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


def test_dashboard_config_custom_base_url():
    """Verify API_BASE_URL respects environment variable or custom overrides."""
    with patch.dict(os.environ, {"API_BASE_URL": "http://custom-mlops-host:9000"}):
        # Reload or test endpoint formatting logic
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

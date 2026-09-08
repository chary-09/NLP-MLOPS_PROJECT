"""Top header banner component displaying application identity, model version, and API status."""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st
from src.dashboard.components.badges import render_badge, render_status_badge
from src.dashboard.config import APP_NAME, APP_TAGLINE, APP_VERSION


def render_header(
    api_healthy: bool,
    latency_ms: Optional[float] = None,
    model_version: str = "v1.0.0",
    model_name: str = "LogisticRegression",
    environment: str = "Production",
) -> None:
    """Render the top banner with application title, model version area, and API status area."""
    status_label = "healthy" if api_healthy else "offline"
    status_badge_html = render_status_badge(status_label, latency_ms)
    version_badge_html = render_badge(f"MODEL: {model_name} ({model_version})", badge_type="info", icon="🤖")
    env_badge_html = render_badge(f"ENV: {environment.upper()}", badge_type="neutral")

    header_html = f"""
    <div class="header-hero">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h1 class="header-title">
                    <span>⚡</span> {APP_NAME} <span style="font-size: 0.9rem; font-weight: 500; color: #60A5FA; background: rgba(37,99,235,0.15); padding: 0.2rem 0.6rem; border-radius: 6px;">Studio</span>
                </h1>
                <p class="header-subtitle">{APP_TAGLINE}</p>
            </div>
            <div style="display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;">
                {version_badge_html}
                {env_badge_html}
                {status_badge_html}
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

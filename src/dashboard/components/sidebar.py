"""Sidebar component with navigation structure, configuration telemetry, and backend status."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
from src.dashboard.components.badges import render_badge, render_status_badge
from src.dashboard.config import API_BASE_URL, APP_NAME, APP_VERSION, PHASE

# Navigation page specifications (filename, label, emoji)
NAVIGATION_PAGES: List[Tuple[str, str, str]] = [
    ("pages/01_Overview.py", "01 Overview", "📊"),
    ("pages/02_Live_Prediction.py", "02 Live Prediction", "🔮"),
    ("pages/03_Prediction_History.py", "03 Prediction History", "📜"),
    ("pages/04_Sentiment_Analytics.py", "04 Sentiment Analytics", "📈"),
    ("pages/05_Explainability.py", "05 Explainability", "🔍"),
    ("pages/06_Model_Performance.py", "06 Model Performance", "🎯"),
    ("pages/07_Drift_Monitoring.py", "07 Drift Monitoring", "🌊"),
    ("pages/08_API_Health.py", "08 API Health", "🩺"),
    ("pages/09_System_Logs.py", "09 System Logs", "📋"),
]


def render_sidebar_navigation() -> None:
    """Render structured sidebar navigation links across all 9 pages."""
    st.markdown('<div class="sidebar-section-title">Studio Pages</div>', unsafe_allow_html=True)
    
    for page_path, label, icon in NAVIGATION_PAGES:
        try:
            # st.page_link supports path relative to main entrypoint
            st.page_link(page_path, label=label, icon=icon)
        except Exception:
            # Fallback for headless or alternative execution contexts
            try:
                # Try relative to src/dashboard
                st.page_link(f"src/dashboard/{page_path}", label=label, icon=icon)
            except Exception:
                pass


def render_sidebar(
    api_healthy: bool,
    latency_ms: Optional[float] = None,
    health_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the sidebar with brand identity, multi-page navigation, backend telemetry, and system metadata."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-branding">
                <div class="sidebar-logo-title">
                    <span>⚡</span> {APP_NAME}
                </div>
                <div class="sidebar-logo-sub">NLP Sentiment Analysis MLOps</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation Menu
        render_sidebar_navigation()

        st.divider()

        # Backend Connection & Telemetry
        st.markdown('<div class="sidebar-section-title">Backend Gateway</div>', unsafe_allow_html=True)
        status_label = "healthy" if api_healthy else "offline"
        st.markdown(render_status_badge(status_label, latency_ms), unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="margin-top: 0.75rem; font-size: 0.8rem; color: #94A3B8;">
                <div><strong>Base URL:</strong> <code style="color: #60A5FA;">{API_BASE_URL}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Refresh Telemetry", key="sidebar_refresh_btn", use_container_width=True):
            st.rerun()

        st.divider()

        # Diagnostics summary
        st.markdown('<div class="sidebar-section-title">Diagnostics</div>', unsafe_allow_html=True)
        if api_healthy and health_data:
            db_ok = health_data.get("database_connected", False)
            model_ok = health_data.get("model_loaded", False)
            vec_ok = health_data.get("vectorizer_loaded", False)

            db_badge = render_badge("CONNECTED" if db_ok else "OFFLINE", "success" if db_ok else "danger")
            model_badge = render_badge("LOADED" if model_ok else "NOT LOADED", "success" if model_ok else "danger")
            vec_badge = render_badge("LOADED" if vec_ok else "NOT LOADED", "success" if vec_ok else "danger")

            st.markdown(
                f"""
                <div style="font-size: 0.78rem; display: flex; flex-direction: column; gap: 0.4rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #94A3B8;">Database:</span> {db_badge}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #94A3B8;">NLP Model:</span> {model_badge}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #94A3B8;">TF-IDF:</span> {vec_badge}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="font-size: 0.78rem; color: #F87171; background: rgba(239,68,68,0.1); padding: 0.5rem; border-radius: 6px; border: 1px solid rgba(239,68,68,0.2);">
                    ⚠️ FastAPI backend unreachable. Start server on port 8000.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Phase metadata
        st.markdown(
            f"""
            <div style="font-size: 0.72rem; color: #64748B;">
                <div><strong>Dashboard:</strong> v{APP_VERSION}</div>
                <div><strong>Milestone:</strong> {PHASE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

"""Sidebar component with navigation structure, configuration telemetry, and backend status."""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st
from src.dashboard.components.badges import render_badge, render_status_badge
from src.dashboard.config import API_BASE_URL, APP_NAME, APP_VERSION, PHASE


def render_sidebar(
    api_healthy: bool,
    latency_ms: Optional[float] = None,
    health_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the sidebar with brand identity, backend telemetry, and system metadata."""
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

        st.markdown('<div class="sidebar-section-title">Backend Connection</div>', unsafe_allow_html=True)
        status_label = "healthy" if api_healthy else "offline"
        st.markdown(render_status_badge(status_label, latency_ms), unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="margin-top: 0.75rem; font-size: 0.8rem; color: #94A3B8;">
                <div><strong>Endpoint:</strong> <code style="color: #60A5FA;">{API_BASE_URL}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Check API Status", use_container_width=True):
            st.rerun()

        st.divider()

        # Telemetry Summary if reachable
        st.markdown('<div class="sidebar-section-title">Backend Diagnostics</div>', unsafe_allow_html=True)
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
                        <span style="color: #94A3B8;">TF-IDF Vectorizer:</span> {vec_badge}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="font-size: 0.78rem; color: #F87171; background: rgba(239,68,68,0.1); padding: 0.5rem; border-radius: 6px;">
                    ⚠️ Backend API is currently unreachable. Launch FastAPI backend to enable live telemetry.
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

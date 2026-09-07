"""Reusable UI state components: Loading, Error, and Empty states."""

from __future__ import annotations

from typing import Optional
import streamlit as st


def render_loading_state(message: str = "Querying MLOps Telemetry & Model Pipeline...") -> None:
    """Render a sleek loading state card with animated spinner."""
    st.markdown(
        f"""
        <div class="loader-card">
            <div class="pulse-dot success" style="width: 14px; height: 14px;"></div>
            <span style="font-weight: 600; font-size: 0.95rem;">{message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_state(
    title: str = "Service Communication Error",
    detail: str = "Unable to establish connection with the FastAPI backend service.",
    suggestion: Optional[str] = "Verify that the backend API is running on the configured API_BASE_URL.",
) -> None:
    """Render an informative error state card with troubleshooting guidance."""
    sugg_html = f'<p style="margin-top: 0.5rem; font-size: 0.8rem; color: #94A3B8;">💡 <strong>Troubleshooting:</strong> {suggestion}</p>' if suggestion else ""
    st.markdown(
        f"""
        <div class="error-state-card">
            <div class="error-state-title">
                <span>⚠️</span>
                <span>{title}</span>
            </div>
            <p class="error-state-body">{detail}</p>
            {sugg_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    title: str = "No Data Records Found",
    message: str = "There are no prediction or monitoring events recorded in the database yet.",
    icon: str = "📊",
) -> None:
    """Render a clean empty state card with icon, title, and descriptive message."""
    st.markdown(
        f"""
        <div class="empty-state-box">
            <div class="empty-state-icon">{icon}</div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-desc">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

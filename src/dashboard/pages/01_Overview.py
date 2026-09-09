"""01 Overview Page — SentimentOps Studio.

Unified production dashboard presenting real-time model telemetry,
binary sentiment distribution, baseline and production evaluation metrics,
data drift status, system health, active alerts, and recent predictions.
"""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

from src.dashboard.components.alerts import render_recent_alerts_section
from src.dashboard.components.badges import (
    render_badge,
    render_binary_badge,
    render_drift_badge,
    render_status_badge,
)
from src.dashboard.components.cards import render_metric_card
from src.dashboard.components.sections import (
    render_model_info_card,
    render_recent_predictions_table,
    render_section_heading,
    render_system_health_grid,
)
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.config import (
    APP_NAME,
    LAYOUT,
    fetch_overview_data,
)

# 1. Page Configuration
st.set_page_config(
    page_title=f"01 Overview | {APP_NAME}",
    page_icon="📊",
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


# 2. Inject CSS Stylesheets safely
def load_css() -> None:
    """Load and inject external CSS files."""
    css_dir = Path(__file__).parent.parent / "styles"
    for css_file in ["main.css", "cards.css", "sidebar.css"]:
        full_path = css_dir / css_file
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# 3. Retrieve Live Data from Backend Gateway
data = fetch_overview_data()

# 4. Render Sidebar
render_sidebar(
    api_healthy=data["api_available"],
    latency_ms=data["latency_ms"],
    health_data=data["system_health"],
)

# 5. Header Section
status_badge_html = render_status_badge(data["api_status"], data["latency_ms"])
drift_badge_html = render_drift_badge(data["drift_status"])
binary_badge_html = render_binary_badge()
version_badge_html = render_badge(f"v{data['model_version']}", badge_type="info")

st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 1.25rem;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        flex-wrap: wrap;
        gap: 1rem;
    ">
        <div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.8rem;">📊</span>
                <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
                    Overview & Telemetry
                </h1>
            </div>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.88rem;">
                Real-time MLOps operational intelligence, binary sentiment metrics, and model health.
            </p>
        </div>
        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            {binary_badge_html}
            {version_badge_html}
            {drift_badge_html}
            {status_badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 6. Service Offline Notice (if backend is not running)
if not data["api_available"]:
    st.markdown(
        """
        <div style="
            background: rgba(239, 68, 68, 0.08);
            border-left: 4px solid #EF4444;
            border-top: 1px solid rgba(239, 68, 68, 0.25);
            border-right: 1px solid rgba(239, 68, 68, 0.25);
            border-bottom: 1px solid rgba(239, 68, 68, 0.25);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
        ">
            <div style="display: flex; align-items: center; gap: 8px; color: #F87171; font-weight: 700; font-size: 0.95rem;">
                <span>⚠️</span> FastAPI Backend Offline
            </div>
            <div style="color: #CBD5E1; font-size: 0.83rem; margin-top: 4px; line-height: 1.4;">
                The dashboard is currently running in decoupled offline mode. Start your FastAPI server on port 8000 using:
                <br/><code style="color: #60A5FA; background: rgba(15, 23, 42, 0.6); padding: 2px 6px; border-radius: 4px; margin-top: 4px; display: inline-block;">
                    python -m uvicorn src.api.main:app --reload --port 8000
                </code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 7. Category 1: Inference Volume & Binary Sentiment Distribution
render_section_heading(
    title="Inference Volume & Binary Sentiment",
    subtitle="Live production inference counters and binary sentiment proportion (strictly Positive / Negative)",
    icon="⚡",
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        label="Total Predictions",
        value=f"{data['total_predictions']:,}",
        subtext="Database records processed",
        accent="blue",
    )

with col2:
    render_metric_card(
        label="Positive Sentiment",
        value=f"{data['positive_pct']:.1f}%",
        subtext="Binary Class: Positive",
        accent="emerald",
    )

with col3:
    render_metric_card(
        label="Negative Sentiment",
        value=f"{data['negative_pct']:.1f}%",
        subtext="Binary Class: Negative",
        accent="rose",
    )

with col4:
    conf_val = data["average_confidence"]
    conf_display = f"{conf_val * 100:.1f}%" if 0 < conf_val <= 1.0 else f"{conf_val:.1f}%"
    render_metric_card(
        label="Average Confidence",
        value=conf_display if conf_val > 0 else "N/A",
        subtext="Mean prediction certainty",
        accent="purple",
    )

# 8. Category 2: Model Performance Metrics
render_section_heading(
    title="Model Performance Metrics",
    subtitle="Phase 1 test evaluation baselines and verified production ground-truth metrics",
    icon="🎯",
)

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    acc_val = data["accuracy"]
    acc_display = f"{acc_val * 100:.2f}%" if acc_val is not None else "N/A"
    render_metric_card(
        label="Accuracy",
        value=acc_display,
        subtext="Baseline test score (0.8904)",
        accent="blue",
    )

with m_col2:
    prec_val = data["precision"]
    prec_display = f"{prec_val * 100:.2f}%" if prec_val is not None else "N/A"
    render_metric_card(
        label="Precision",
        value=prec_display,
        subtext="Weighted positive predictive",
        accent="blue",
    )

with m_col3:
    rec_val = data["recall"]
    rec_display = f"{rec_val * 100:.2f}%" if rec_val is not None else "N/A"
    render_metric_card(
        label="Recall",
        value=rec_display,
        subtext="True positive sensitivity",
        accent="blue",
    )

with m_col4:
    f1_val = data["f1_score"]
    f1_display = f"{f1_val * 100:.2f}%" if f1_val is not None else "N/A"
    render_metric_card(
        label="F1 Score",
        value=f1_display,
        subtext="Harmonic mean of P & R",
        accent="purple",
    )

# 9. Category 3: System Health & Diagnostics
render_section_heading(
    title="System Diagnostics & Architecture",
    subtitle="Component health status across FastAPI, SQLite, model pipeline, and vectorizer",
    icon="🩺",
)
render_system_health_grid(data["system_health"], latency_ms=data["latency_ms"])

st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
render_model_info_card(data["model_info"], model_version=data["model_version"])

# 10. Category 4: Recent Predictions (Live Table)
preds_count = len(data["recent_predictions"])
render_section_heading(
    title="Recent Predictions",
    subtitle="Latest inference records stored in SQLite database with binary classification badges",
    icon="📜",
    badge_html=render_badge(f"{preds_count} Loaded", badge_type="neutral"),
)
render_recent_predictions_table(data["recent_predictions"])

# 11. Category 5: Threshold Alerts & Drift Monitoring
render_section_heading(
    title="Threshold Alerts & Drift Health",
    subtitle="Real-time alert engine evaluating error rates, latency limits, confidence drops, and NLP drift",
    icon="🚨",
)
render_recent_alerts_section(data["recent_alerts"])

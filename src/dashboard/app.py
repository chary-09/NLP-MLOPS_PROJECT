"""SentimentOps Studio - Production NLP Sentiment Analysis MLOps Dashboard.

Day 1 — Phase 3: Dashboard Foundation & Reusable Component System.
Entry point for the Streamlit visual monitoring platform.
"""

from __future__ import annotations

import streamlit as st

# Configure page layout first before any other Streamlit commands
from src.dashboard.config import (
    API_BASE_URL,
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    ENDPOINTS,
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
    PHASE,
    fetch_api_health,
    fetch_model_info,
)
from src.dashboard.theme import inject_custom_css

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

# Inject custom CSS styles for dark MLOps aesthetic
inject_custom_css()

# Import reusable UI components
from src.dashboard.components.badges import render_badge, render_status_badge
from src.dashboard.components.cards import render_card_header, render_kpi_card, render_spec_grid
from src.dashboard.components.header import render_header
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.states import (
    render_empty_state,
    render_error_state,
    render_loading_state,
)

# -----------------------------------------------------------------------------
# 1. API Health & Model Metadata Discovery (Graceful Fallback)
# -----------------------------------------------------------------------------
api_healthy, health_data, latency_ms = fetch_api_health()

model_name = "LogisticRegression"
model_version = "v1.0.0"
vectorizer_info = {"type": "TfidfVectorizer", "max_features": 5000, "vocabulary_size": 5000}
classes = ["negative", "positive"]

if api_healthy:
    model_version = health_data.get("model_version", "v1.0.0")
    info_ok, info_data = fetch_model_info()
    if info_ok:
        model_name = info_data.get("model_name", "LogisticRegression")
        vectorizer_info = info_data.get("vectorizer", vectorizer_info)
        classes = info_data.get("classes", classes)

# -----------------------------------------------------------------------------
# 2. Header & Sidebar Rendering
# -----------------------------------------------------------------------------
render_header(
    api_healthy=api_healthy,
    latency_ms=latency_ms if api_healthy else None,
    model_version=model_version,
    model_name=model_name,
    environment="Production",
)

render_sidebar(
    api_healthy=api_healthy,
    latency_ms=latency_ms if api_healthy else None,
    health_data=health_data if api_healthy else None,
)

# -----------------------------------------------------------------------------
# 3. Main Dashboard Body: Telemetry & Foundation Overview
# -----------------------------------------------------------------------------
st.markdown("### 🖥️ MLOps System & Inference Engine Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    api_status_text = "OPERATIONAL" if api_healthy else "OFFLINE"
    delta_text = f"{latency_ms} ms" if api_healthy else "No Response"
    render_kpi_card(
        label="FastAPI Gateway",
        value=api_status_text,
        subtext="Backend Health Endpoint",
        delta=delta_text,
        delta_positive=api_healthy,
    )

with col2:
    render_kpi_card(
        label="Inference Engine",
        value=model_name.capitalize(),
        subtext=f"Model Release {model_version}",
        delta="Phase 1 Active",
        delta_positive=True,
    )

with col3:
    db_text = "ONLINE" if (api_healthy and health_data.get("database_connected")) else ("STANDBY" if not api_healthy else "ERROR")
    render_kpi_card(
        label="Persistence Layer",
        value=db_text,
        subtext="SQLite (data/sentiment.db)",
        delta="ACID Compliant",
        delta_positive=(db_text == "ONLINE"),
    )

with col4:
    render_kpi_card(
        label="Explainable AI",
        value="SHAP + LIME",
        subtext="Local Feature Attribution",
        delta="Phase 2 Ready",
        delta_positive=True,
    )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Backend Connectivity & Architecture Details
# -----------------------------------------------------------------------------
tab_overview, tab_components, tab_endpoints, tab_roadmap = st.tabs([
    "📊 Architecture & Health",
    "🎨 Reusable UI Components",
    "🔌 API Gateway Endpoints",
    "🗺️ Phase 3 Roadmap",
])

with tab_overview:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            render_card_header(
                "Inference Engine Specifications",
                "Metadata fetched dynamically from backend `/model-info`",
                badge_html=render_badge("Active Model", "info"),
            ),
            unsafe_allow_html=True,
        )

        specs = {
            "Model Algorithm": model_name,
            "Model Version": model_version,
            "Vectorizer Type": vectorizer_info.get("type", "TfidfVectorizer"),
            "Vocabulary Size": f"{vectorizer_info.get('vocabulary_size', 5000):,} terms",
            "Max Features": f"{vectorizer_info.get('max_features', 5000):,}",
            "N-Gram Range": "(1, 2) Unigram + Bigram",
            "Classes": ", ".join(classes).title(),
            "Backend Base URL": API_BASE_URL,
        }
        render_spec_grid(specs)

    with col_right:
        st.markdown(
            render_card_header(
                "Backend Health Diagnostics",
                f"Real-time ping via {API_BASE_URL}/health",
                badge_html=render_status_badge("healthy" if api_healthy else "offline", latency_ms),
            ),
            unsafe_allow_html=True,
        )

        if api_healthy and health_data:
            st.json(health_data)
        else:
            render_error_state(
                title="FastAPI Backend Unreachable",
                detail=(
                    f"The dashboard attempted to contact <code>{API_BASE_URL}/health</code>, but the "
                    "server did not respond. The dashboard continues running in resilient offline mode."
                ),
                suggestion=(
                    "Start the FastAPI server in a separate terminal: "
                    "<code>uvicorn src.api.main:app --reload --port 8000</code>"
                ),
            )

with tab_components:
    st.markdown("#### Reusable Component Foundation Gallery")
    st.caption("Standardized MLOps building blocks designed for subsequent Phase 3 pages.")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown("##### 1. Status Badges & Pulsing Indicators")
        b1 = render_badge("SYSTEM OPTIMAL", "success", pulse=True)
        b2 = render_badge("DRIFT DETECTED", "warning", pulse=True)
        b3 = render_badge("NODE UNRESPONSIVE", "danger", pulse=False)
        b4 = render_badge("LOGISTIC REGRESSION", "info")
        b5 = render_badge("STAGING", "neutral")
        st.markdown(f"{b1} &nbsp; {b2} &nbsp; {b3} &nbsp; {b4} &nbsp; {b5}", unsafe_allow_html=True)

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### 2. Loading State Component")
        render_loading_state("Synchronizing prediction metrics from SQLite store...")

    with comp_col2:
        st.markdown("##### 3. Error State Component")
        render_error_state(
            title="Inference Pipeline Timeout",
            detail="Vectorization step exceeded 500ms timeout threshold for high-volume text payload.",
            suggestion="Check input sequence length or scale worker threads in uvicorn configuration.",
        )

        st.markdown("##### 4. Empty State Component")
        render_empty_state(
            title="No Drift Analysis History",
            message="Run at least 20 live predictions to calibrate Kolmogorov-Smirnov baseline comparison.",
            icon="📉",
        )

with tab_endpoints:
    st.markdown("#### Dynamic API Endpoint Registry")
    st.caption(f"All routes resolve dynamically via <code>API_BASE_URL={API_BASE_URL}</code>.")

    endpoint_rows = [
        {"Endpoint Key": key, "Path": path, "Full Target URL": f"{API_BASE_URL}{path}"}
        for key, path in ENDPOINTS.items()
    ]
    st.dataframe(endpoint_rows, use_container_width=True, hide_index=True)

with tab_roadmap:
    st.markdown("#### Phase 3 Dashboard Implementation Plan")
    st.markdown(
        """
        | Day | Milestone | Scope & Deliverables | Status |
        |---|---|---|---|
        | **Day 1** | **Dashboard Foundation** | Reusable UI components, theme tokens, sidebar, header, dynamic config, API health check | **✅ COMPLETE** |
        | **Day 2** | **Live Prediction & Batch Inference** | Single text prediction, confidence meter, batch CSV upload, instant latency display | ⏳ Upcoming |
        | **Day 3** | **Prediction History & Analytics** | SQLite history inspection, pagination, sentiment distribution graphs, export CSV | ⏳ Upcoming |
        | **Day 4** | **Explainable AI Studio** | Visual SHAP and LIME word contribution bar charts, comparative XAI inspection | ⏳ Upcoming |
        | **Day 5** | **MLOps Monitoring & Drift Telemetry** | 4-category telemetry (Model, Predictions, Latency, KS-Drift & OOV Vocabulary) | ⏳ Upcoming |
        """
    )

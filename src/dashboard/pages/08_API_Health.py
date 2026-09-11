"""08 API Health Page — SentimentOps Studio.

FastAPI REST endpoints health check, latency monitoring, and backend status diagnostics.
"""

from __future__ import annotations

from pathlib import Path
import time
import requests
import streamlit as st

from src.dashboard.components.alerts import render_alert_box
from src.dashboard.components.cards import render_kpi_card
from src.dashboard.components.sections import render_section_heading, render_system_health_grid
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.config import (
    API_BASE_URL,
    APP_NAME,
    ENDPOINTS,
    LAYOUT,
    fetch_api_health,
)


st.set_page_config(
    page_title=f"08 API Health | {APP_NAME}",
    page_icon="🩺",
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_dir = Path(__file__).parent.parent / "styles"
    for css_file in ["main.css", "cards.css", "sidebar.css"]:
        full_path = css_dir / css_file
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()
is_healthy, health_data, latency_ms = fetch_api_health()
render_sidebar(api_healthy=is_healthy, latency_ms=latency_ms, health_data=health_data)

st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.8rem;">🩺</span>
                <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
                    API Gateway & System Health
                </h1>
            </div>
            <p style="color: #94A3B8; font-size: 0.88rem; margin: 4px 0 0 0;">
                Live healthcheck diagnostics, backend ping latency, database connectivity, and endpoint registry.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top KPI Summary Cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    api_status_str = "HEALTHY" if is_healthy else ("DEGRADED" if health_data.get("model_loaded") else "OFFLINE")
    render_kpi_card("API Status", api_status_str, accent_color="#10B981" if is_healthy else "#EF4444", subtext="FastAPI Gateway")
with k2:
    render_kpi_card("Ping Latency", f"{latency_ms} ms", accent_color="#3B82F6", subtext="Round-trip time")
with k3:
    db_conn = "CONNECTED" if health_data.get("database_connected") else "DISCONNECTED"
    render_kpi_card("SQLite Database", db_conn, accent_color="#8B5CF6" if health_data.get("database_connected") else "#EF4444", subtext="data/sentiment.db")
with k4:
    model_ver = health_data.get("model_version", "0.1.0") if is_healthy else "N/A"
    render_kpi_card("Model Release", f"v{model_ver}", accent_color="#F59E0B", subtext="Logistic Regression")

st.markdown("<hr style='border-color: #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)

# System Health Component Grid
render_section_heading("System Health Subsystems", "Status of FastAPI, Database, ML Model, and TF-IDF Vectorizer", "⚡")

health_subsystems = {
    "api": health_data.get("api", is_healthy),
    "database_connected": health_data.get("database_connected", False),
    "model_loaded": health_data.get("model_loaded", False),
    "vectorizer_loaded": health_data.get("vectorizer_loaded", False),
}
render_system_health_grid(health_subsystems)

# Interactive Endpoint Ping Tester
st.markdown("<hr style='border-color: #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)
render_section_heading("Live Endpoint Diagnostic Inspector", "Ping individual FastAPI REST endpoints in real-time", "🔍")

endpoint_keys = list(ENDPOINTS.keys())
selected_ep = st.selectbox("Select Endpoint to Ping", options=endpoint_keys, format_func=lambda k: f"{k} ({ENDPOINTS[k]})")

if st.button("🚀 Ping Selected Endpoint", type="primary"):
    ep_path = ENDPOINTS[selected_ep]
    full_url = f"{API_BASE_URL}{ep_path}"
    start_t = time.perf_counter()
    try:
        if ep_path in ["/predict", "/metrics/evaluate-production", "/explain"]:
            resp = requests.post(full_url, json={"text": "Test ping string"}, timeout=3.0)
        else:
            resp = requests.get(full_url, timeout=3.0)
        ping_latency = round((time.perf_counter() - start_t) * 1000, 2)
        
        st.markdown(
            f"""
            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 1rem; margin-top: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div><strong>URL:</strong> <code style="color: #38BDF8;">{full_url}</code></div>
                    <div><strong>Status:</strong> <span style="color: {'#10B981' if resp.status_code == 200 else '#EF4444'}; font-weight: 700;">HTTP {resp.status_code}</span></div>
                    <div><strong>Latency:</strong> <span style="color: #60A5FA; font-weight: 700;">{ping_latency} ms</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View Raw API Response Payload"):
            st.json(resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text)
    except Exception as exc:
        render_alert_box("Ping Failure", f"Failed to ping `{full_url}`: {exc}", alert_type="danger", icon="❌")


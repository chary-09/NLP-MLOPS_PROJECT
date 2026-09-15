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
from src.dashboard.config import API_BASE_URL, APP_NAME, ENDPOINTS, LAYOUT, fetch_api_health, fetch_metrics, fetch_monitoring_endpoint


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
    model_ver = health_data.get("model_version") if is_healthy else None
    render_kpi_card("Model Release", f"v{model_ver}" if model_ver else "Unavailable", accent_color="#F59E0B", subtext="Backend health response")

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

metrics_ok, system_metrics = fetch_metrics()
system = system_metrics.get("system_monitoring", {}) if metrics_ok and isinstance(system_metrics, dict) else {}
st.markdown("### Runtime telemetry")
if metrics_ok and system:
    runtime_cols = st.columns(5)
    runtime_cols[0].metric("Requests", system.get("total_requests", "Unavailable"))
    runtime_cols[1].metric("Errors", system.get("api_error_count", "Unavailable"))
    runtime_cols[2].metric("Success rate", f"{(100 - float(system.get('error_rate_percentage', 0))):.2f}%" if system.get("error_rate_percentage") is not None else "Unavailable")
    runtime_cols[3].metric("Average latency", f"{system.get('average_latency_ms')} ms" if system.get("average_latency_ms") is not None else "Unavailable")
    runtime_cols[4].metric("Max latency", f"{system.get('max_latency_ms')} ms" if system.get("max_latency_ms") is not None else "Unavailable")
else:
    render_error_state("Runtime metrics unavailable", "The backend did not return system telemetry.", suggestion=None)

st.markdown("### Endpoint status")
endpoint_rows = []
for endpoint_key in ["predict", "health", "metrics", "predictions", "model_info"]:
    if endpoint_key == "predict":
        endpoint_ok = is_healthy and bool(health_data.get("model_loaded")) and bool(health_data.get("vectorizer_loaded"))
        endpoint_data = health_data
        endpoint_latency = latency_ms
    elif endpoint_key == "health":
        endpoint_ok, endpoint_data, endpoint_latency = is_healthy, health_data, latency_ms
    else:
        endpoint_ok, endpoint_data, endpoint_latency = fetch_monitoring_endpoint(endpoint_key)
    endpoint_rows.append({"Endpoint": ENDPOINTS[endpoint_key], "Status": "Healthy" if endpoint_ok else "Error", "Latency (ms)": endpoint_latency, "Detail": endpoint_data.get("error", "Available")})
st.dataframe(endpoint_rows, hide_index=True, use_container_width=True)

# Interactive Endpoint Ping Tester
st.markdown("<hr style='border-color: #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)
render_section_heading("Live Endpoint Diagnostic Inspector", "Ping individual FastAPI REST endpoints in real-time", "🔍")

endpoint_keys = list(ENDPOINTS.keys())
selected_ep = st.selectbox("Select Endpoint to Ping", options=endpoint_keys, format_func=lambda k: f"{k} ({ENDPOINTS[k]})")

if st.button("🚀 Ping Selected Endpoint", type="primary"):
    ep_path = ENDPOINTS[selected_ep]
    full_url = f"{API_BASE_URL}{ep_path}"
    if ep_path == "/predict":
        st.info("`/predict` accepts POST requests with review text. Readiness is reported above from the existing `/health` response; no test prediction is created by this diagnostic.")
        st.stop()
    start_t = time.perf_counter()
    try:
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


"""09 System Logs Page — SentimentOps Studio."""

from __future__ import annotations

from pathlib import Path
import streamlit as st
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.config import APP_NAME, LAYOUT, fetch_api_health, fetch_logs
from src.dashboard.components.states import render_empty_state, render_error_state

st.set_page_config(
    page_title=f"09 System Logs | {APP_NAME}",
    page_icon="📋",
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
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
        <span style="font-size: 1.8rem;">📋</span>
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
            System & Inference Logs
        </h1>
    </div>
    <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 1.5rem;">
        Real-time rotating application logs, audit trails, and monitoring exception records.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Application log stream")
level = st.selectbox("Log level", ["All", "INFO", "WARNING", "ERROR"])
component = st.text_input("Component filter", placeholder="Optional logger name")
date = st.date_input("UTC date", value=None)
date_value = date.isoformat() if date else None
ok, payload = fetch_logs(
    level=None if level == "All" else level,
    component=component.strip() or None,
    date=date_value,
)
if not ok:
    render_error_state("Logs unavailable", payload.get("error", "The logs endpoint returned an invalid response."))
else:
    logs = payload.get("logs", [])
    if not logs:
        render_empty_state("No logs available", "The backend returned no log records for the selected filters.", icon="📋")
    else:
        st.caption(f"{len(logs)} actual log record(s) returned by the backend")
        st.dataframe(logs, hide_index=True, use_container_width=True)

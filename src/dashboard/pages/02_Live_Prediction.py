"""02 Live Prediction Page — SentimentOps Studio.

Interactive inference console scheduled for Day 3.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st
from src.dashboard.components.badges import render_badge
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.config import APP_NAME, LAYOUT, fetch_api_health

st.set_page_config(
    page_title=f"02 Live Prediction | {APP_NAME}",
    page_icon="🔮",
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
        <span style="font-size: 1.8rem;">🔮</span>
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
            Live Sentiment Prediction
        </h1>
    </div>
    <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 1.5rem;">
        Real-time interactive sentiment inference powered by the trained Logistic Regression model.
    </p>
    """,
    unsafe_allow_html=True,
)

st.info("🔮 **Scheduled for Day 3**: Real-time text input area, confidence gauges, and instant classification.")

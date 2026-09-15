"""07 Drift Monitoring Page — SentimentOps Studio."""

from __future__ import annotations

from pathlib import Path
import streamlit as st
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.config import APP_NAME, LAYOUT, fetch_api_health, fetch_drift_metrics, fetch_metrics
from src.dashboard.components.states import render_empty_state, render_error_state

st.set_page_config(
    page_title=f"07 Drift Monitoring | {APP_NAME}",
    page_icon="🌊",
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
        <span style="font-size: 1.8rem;">🌊</span>
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
            NLP Data Drift Monitoring
        </h1>
    </div>
    <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 1.5rem;">
        Track Kolmogorov-Smirnov distribution shifts, Out-Of-Vocabulary rates, and token length drifts.
    </p>
    """,
    unsafe_allow_html=True,
)

ok, drift = fetch_drift_metrics()
metrics_ok, telemetry = fetch_metrics()
if not ok:
    render_error_state("Drift data unavailable", drift.get("error", "The drift endpoint did not return data."))
else:
    status = str(drift.get("status", "UNKNOWN"))
    score = drift.get("score")
    threshold = drift.get("threshold")
    columns = st.columns(4)
    columns[0].metric("Current status", status)
    columns[1].metric("Drift score", f"{float(score):.4f}" if isinstance(score, (int, float)) else "Unavailable")
    columns[2].metric("Threshold", f"{float(threshold):.4f}" if isinstance(threshold, (int, float)) else "Unavailable")
    columns[3].metric("Samples", drift.get("details", {}).get("production_sample_count", "Unavailable"))
    st.info(drift.get("interpretation", "No interpretation returned by the backend."))

    history = st.session_state.setdefault("drift_history", [])
    if isinstance(score, (int, float)) and isinstance(threshold, (int, float)):
        snapshot = {"score": score, "threshold": threshold, "status": status}
        if not history or history[-1] != snapshot:
            history.append(snapshot)
    if history:
        st.markdown("### Drift history")
        import pandas as pd
        import plotly.express as px
        history_frame = pd.DataFrame(history)
        history_frame.index = history_frame.index + 1
        st.plotly_chart(px.line(history_frame, y=["score", "threshold"], markers=True, labels={"index": "Refresh", "value": "Score"}), use_container_width=True)
        st.dataframe(history_frame, use_container_width=True)
    else:
        render_empty_state("Drift history unavailable", "The backend has not returned a numeric drift snapshot yet.", icon="⌁")

    details = drift.get("details")
    if isinstance(details, dict):
        st.markdown("### Feature drift")
        feature_rows = [
            {"Feature": "Text length (KS)", "Value": details.get("ks_statistic"), "Reference": details.get("reference_mean_char_length"), "Current": details.get("production_mean_char_length")},
            {"Feature": "Vocabulary OOV rate", "Value": details.get("oov_rate"), "Reference": "TF-IDF vocabulary", "Current": details.get("total_production_tokens_analyzed")},
        ]
        st.dataframe(pd.DataFrame(feature_rows), hide_index=True, use_container_width=True)
    else:
        render_empty_state("Feature drift unavailable", "The backend did not return drift details.", icon="⌁")

    st.markdown("### Alerts")
    alerts = telemetry.get("alerts", []) if metrics_ok and isinstance(telemetry, dict) else []
    if alerts:
        for alert in alerts:
            st.warning(str(alert))
    else:
        st.success("No active monitoring alerts returned by the backend.") if metrics_ok else st.info("Alerts unavailable because unified monitoring data could not be loaded.")

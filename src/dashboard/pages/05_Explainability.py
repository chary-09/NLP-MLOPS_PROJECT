"""05 Explainability (XAI) Page — SentimentOps Studio."""

from __future__ import annotations

from pathlib import Path
import streamlit as st
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.config import APP_NAME, LAYOUT, fetch_api_health, fetch_explanation
from src.dashboard.components.states import render_error_state

st.set_page_config(
    page_title=f"05 Explainability | {APP_NAME}",
    page_icon="🔍",
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
        <span style="font-size: 1.8rem;">🔍</span>
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
            Model Explainability (XAI)
        </h1>
    </div>
    <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 1.5rem;">
        Interpret model predictions using SHAP LinearExplainer and LIME token-level attributions.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Explain a prediction")
text = st.text_area(
    "Input text",
    value="The product was amazing but delivery was terrible.",
    height=120,
)
method = st.selectbox("Explanation method", ["both", "shap", "lime"])
top_n = st.slider("Important words", min_value=1, max_value=20, value=10)

if st.button("Generate explanation", type="primary", use_container_width=True):
    ok, result, latency_ms = fetch_explanation(text, method=method, top_n=top_n)
    if not ok:
        render_error_state("Explanation unavailable", result.get("error", "The XAI service did not return an explanation."))
    else:
        prediction = str(result.get("prediction", "unknown")).upper()
        confidence = result.get("confidence")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Prediction", prediction)
        metric_cols[1].metric("Confidence", f"{float(confidence):.1%}" if confidence is not None else "Unavailable")
        metric_cols[2].metric("XAI latency", f"{latency_ms:.1f} ms")
        st.caption(f"Model version: {result.get('model_version', 'Unavailable')} | Method: {result.get('method', method)}")

        contributions = result.get("explanation", [])
        if not contributions:
            render_error_state("Explanation unavailable", "The XAI service returned no feature contributions.", suggestion=None)
        else:
            import pandas as pd
            import plotly.express as px

            frame = pd.DataFrame(contributions)
            frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce")
            frame = frame.dropna(subset=["feature", "importance"])
            frame["direction"] = frame["importance"].map(
                lambda value: "Positive contribution" if value >= 0 else "Negative contribution"
            )
            frame = frame.sort_values("importance")
            st.markdown("#### Word contributions")
            chart = px.bar(
                frame,
                x="importance",
                y="feature",
                color="direction",
                orientation="h",
                color_discrete_map={"Positive contribution": "#10B981", "Negative contribution": "#F87171"},
                labels={"importance": "Contribution", "feature": "Word"},
            )
            chart.update_layout(height=max(280, len(frame) * 32), showlegend=True)
            st.plotly_chart(chart, use_container_width=True)
            display = frame[["feature", "importance"]].copy()
            display["importance"] = display["importance"].map(lambda value: f"{value:+.3f}")
            st.dataframe(
                display.rename(columns={"feature": "Word", "importance": "Contribution"}),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown(f"**Positive words:** {', '.join(result.get('positive_words', [])) or 'None returned'}")
            st.markdown(f"**Negative words:** {', '.join(result.get('negative_words', [])) or 'None returned'}")

"""06 Model Performance Page — SentimentOps Studio."""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.config import APP_NAME, LAYOUT, fetch_api_health, fetch_metrics
from src.dashboard.components.states import render_empty_state, render_error_state

st.set_page_config(
    page_title=f"06 Model Performance | {APP_NAME}",
    page_icon="🎯",
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
        <span style="font-size: 1.8rem;">🎯</span>
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
            Model Performance & Validation
        </h1>
    </div>
    <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 1.5rem;">
        Baseline test benchmarks, production ground-truth confusion matrix, and ROC/PR curves.
    </p>
    """,
    unsafe_allow_html=True,
)

metrics_path = Path(__file__).parents[3] / "data" / "models" / "metrics.json"
try:
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        evaluation = json.load(metrics_file)
except (OSError, json.JSONDecodeError) as exc:
    render_error_state("Evaluation metrics unavailable", f"Could not load the checked-in evaluation artifact: {exc}")
    evaluation = {}

models = evaluation.get("models", {})
best_model = evaluation.get("best_model")
if not models:
    render_empty_state("No evaluation metrics", "Run the existing evaluation pipeline to produce data/models/metrics.json.")
else:
    st.markdown(f"**Selected model:** `{best_model or 'Unavailable'}`")
    selected = models.get(best_model, {}) if best_model else {}
    metric_cols = st.columns(4)
    for column, label, key in zip(
        metric_cols,
        ["Accuracy", "Precision", "Recall", "F1"],
        ["accuracy", "precision", "recall", "f1_score"],
    ):
        value = selected.get(key)
        column.metric(label, f"{float(value):.2%}" if value is not None else "Unavailable")

    st.caption("Source: existing Phase 1/Phase 2 evaluation artifact. No models were retrained or re-evaluated by this dashboard.")
    st.markdown("### Model comparison")
    import pandas as pd

    comparison = pd.DataFrame(
        [
            {
                "Model": name.replace("_", " ").title(),
                "Accuracy": values.get("accuracy"),
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1": values.get("f1_score"),
                "Selected": name == best_model,
            }
            for name, values in models.items()
        ]
    )
    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        comparison[metric] = comparison[metric].map(lambda value: f"{value:.2%}" if pd.notna(value) else "Unavailable")
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Confusion matrix")
        matrix = selected.get("confusion_matrix")
        if matrix and len(matrix) == 2:
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual negative", "Actual positive"],
                    columns=["Predicted negative", "Predicted positive"],
                ),
                use_container_width=True,
            )
        else:
            render_empty_state("Confusion matrix unavailable", "The selected model has no stored confusion matrix.", icon="▦")
    with right:
        st.markdown("### Class-wise metrics")
        report = selected.get("report", {})
        class_rows = [
            {
                "Class": label.title(),
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1": values.get("f1-score"),
                "Support": values.get("support"),
            }
            for label, values in report.items()
            if isinstance(values, dict) and label in {"negative", "positive"}
        ]
        if class_rows:
            st.dataframe(pd.DataFrame(class_rows), hide_index=True, use_container_width=True)
        else:
            render_empty_state("Class metrics unavailable", "The selected model has no stored class-wise report.", icon="▤")

    api_ok, telemetry = fetch_metrics()
    production = telemetry.get("model_performance", {}).get("production_metrics", {}) if api_ok else {}
    st.markdown("### Production validation")
    if production.get("status") == "UNAVAILABLE" or production.get("accuracy") is None:
        st.info("Production metrics unavailable: no verified ground-truth labels are recorded. Baseline metrics above are the only displayed evaluation results.")
    else:
        st.success("Verified production metrics are available from the monitoring API.")
        st.json(production)

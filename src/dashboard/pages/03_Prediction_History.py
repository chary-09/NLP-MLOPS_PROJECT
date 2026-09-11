"""03 Prediction History Page — SentimentOps Studio.

Paginated historical logs fetched directly from SQLite via GET /predictions.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.dashboard.components.alerts import render_alert_box
from src.dashboard.components.badges import render_sentiment_badge
from src.dashboard.components.cards import render_kpi_card
from src.dashboard.components.sidebar import render_sidebar

from src.dashboard.config import (
    APP_NAME,
    LAYOUT,
    fetch_api_health,
    fetch_predictions_history,
)

st.set_page_config(
    page_title=f"03 Prediction History | {APP_NAME}",
    page_icon="📜",
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
                <span style="font-size: 1.8rem;">📜</span>
                <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
                    Prediction History Log
                </h1>
            </div>
            <p style="color: #94A3B8; font-size: 0.88rem; margin: 4px 0 0 0;">
                Audit trail of persisted predictions queried directly from SQLite via GET /predictions.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not is_healthy:
    render_alert_box(
        title="FastAPI Backend Offline",
        message="Unable to load prediction history. Please start the FastAPI backend server.",
        alert_type="danger",
        icon="🚨",
    )

# Page limit control
col_limit, col_refresh = st.columns([3, 1])
with col_limit:
    records_limit = st.select_slider(
        "Records to Fetch from API",
        options=[10, 25, 50, 100],
        value=50,
        help="Controls limit parameter sent to GET /predictions endpoint.",
    )
with col_refresh:
    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Fetch predictions from API
with st.spinner("Fetching predictions history from database..."):
    success, resp_data, req_latency = fetch_predictions_history(limit=records_limit, offset=0)

if not success:
    err_detail = resp_data.get("error", "Failed to communicate with database API.")
    render_alert_box(
        title="API Communication Error",
        message=f"Error querying /predictions: {err_detail}",
        alert_type="danger",
        icon="❌",
    )
else:
    predictions_list = resp_data.get("predictions", [])
    total_db_count = resp_data.get("total", len(predictions_list))

    if not predictions_list:
        render_alert_box(
            title="No Predictions Found",
            message="The database has no recorded predictions yet. Go to <strong>02 Live Prediction</strong> to generate predictions!",
            alert_type="info",
            icon="ℹ️",
        )
    else:
        # Convert to pandas DataFrame for client-side filtering & sorting
        df = pd.DataFrame(predictions_list)

        # Filters & Search Bar
        st.markdown("### 🔍 Search & Filter Records")
        c_search, c_sent, c_conf, c_sort = st.columns([3, 2, 2, 2])

        with c_search:
            search_query = st.text_input("Search Keyword / ID", placeholder="Type keyword or UUID...").strip().lower()

        with c_sent:
            sentiment_filter = st.selectbox("Sentiment Class", options=["All Sentiments", "positive", "negative"])

        with c_conf:
            min_conf = st.slider("Min Confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

        with c_sort:
            sort_by = st.selectbox("Sort Order", options=["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence"])

        # Filter pipeline
        filtered_df = df.copy()

        if search_query:
            filtered_df = filtered_df[
                filtered_df["text"].str.lower().str.contains(search_query, na=False)
                | filtered_df["prediction_id"].str.lower().str.contains(search_query, na=False)
            ]

        if sentiment_filter != "All Sentiments":
            filtered_df = filtered_df[filtered_df["sentiment"].str.lower() == sentiment_filter]

        filtered_df = filtered_df[filtered_df["confidence"] >= min_conf]

        # Sorting logic
        if sort_by == "Newest First":
            filtered_df = filtered_df.sort_values(by="timestamp", ascending=False)
        elif sort_by == "Oldest First":
            filtered_df = filtered_df.sort_values(by="timestamp", ascending=True)
        elif sort_by == "Highest Confidence":
            filtered_df = filtered_df.sort_values(by="confidence", ascending=False)
        elif sort_by == "Lowest Confidence":
            filtered_df = filtered_df.sort_values(by="confidence", ascending=True)

        # Summary KPIs
        st.markdown("<hr style='border-color: #334155; margin: 1rem 0;'>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("Total DB Records", f"{total_db_count}", accent_color="#3B82F6", subtext="Persisted in SQLite")
        with k2:
            render_kpi_card("Filtered Records", f"{len(filtered_df)}", accent_color="#8B5CF6", subtext="Matches criteria")
        with k3:
            pos_cnt = len(filtered_df[filtered_df["sentiment"].str.lower() == "positive"]) if not filtered_df.empty else 0
            render_kpi_card("Positive Count", f"{pos_cnt}", accent_color="#10B981", subtext="Filtered set")
        with k4:
            neg_cnt = len(filtered_df[filtered_df["sentiment"].str.lower() == "negative"]) if not filtered_df.empty else 0
            render_kpi_card("Negative Count", f"{neg_cnt}", accent_color="#EF4444", subtext="Filtered set")

        # Table & Detail Views
        tab_table, tab_cards = st.tabs(["📊 Table View", "🔎 Detailed Record Inspector"])

        with tab_table:
            if filtered_df.empty:
                st.warning("No records matched your search filter criteria.")
            else:
                display_df = filtered_df.copy()
                display_df["sentiment"] = display_df["sentiment"].str.upper()
                display_df["confidence"] = (display_df["confidence"] * 100).round(2).astype(str) + "%"

                st.dataframe(
                    display_df[["prediction_id", "sentiment", "confidence", "model_version", "timestamp", "text"]],
                    column_config={
                        "prediction_id": st.column_config.TextColumn("Prediction ID", width="small"),
                        "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                        "confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "model_version": st.column_config.TextColumn("Model Version", width="small"),
                        "timestamp": st.column_config.TextColumn("Timestamp (UTC)", width="medium"),
                        "text": st.column_config.TextColumn("Input Review Text", width="large"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

                # Export CSV
                csv_export = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Filtered History (CSV)",
                    data=csv_export,
                    file_name="prediction_history_export.csv",
                    mime="text/csv",
                )

        with tab_cards:
            if filtered_df.empty:
                st.info("No records to inspect.")
            else:
                for idx, row in filtered_df.iterrows():
                    raw_sent = str(row.get("sentiment", "unknown")).lower()
                    raw_text = str(row.get("text", ""))
                    raw_time = str(row.get("timestamp", "N/A"))
                    raw_id = str(row.get("prediction_id", "N/A"))
                    raw_ver = str(row.get("model_version", "0.1.0"))
                    
                    try:
                        conf_val = float(row.get("confidence", 0.0))
                    except (ValueError, TypeError):
                        conf_val = 0.0

                    snippet = (raw_text[:80] + "...") if len(raw_text) > 80 else raw_text

                    with st.expander(f"[{raw_sent.upper()}] {snippet} ({raw_time})"):
                        c_a, c_b = st.columns([3, 1])
                        with c_a:
                            st.markdown(f"**Full Input Text:**")
                            st.info(raw_text)
                        with c_b:
                            st.markdown(f"**Class:** {render_sentiment_badge(raw_sent)}", unsafe_allow_html=True)
                            st.markdown(f"**Confidence:** `{conf_val * 100:.2f}%`")
                            st.markdown(f"**Model:** `{raw_ver}`")
                            st.markdown(f"**ID:** `{raw_id}`")





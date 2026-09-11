"""02 Live Prediction Page — SentimentOps Studio.

Interactive sentiment inference console powered by the FastAPI backend.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.dashboard.components.alerts import render_alert_box
from src.dashboard.components.badges import render_sentiment_badge
from src.dashboard.components.cards import render_kpi_card
from src.dashboard.components.sections import render_section_heading
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.config import APP_NAME, LAYOUT, fetch_api_health, make_prediction


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
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.8rem;">🔮</span>
                <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
                    Live Sentiment Prediction Studio
                </h1>
            </div>
            <p style="color: #94A3B8; font-size: 0.88rem; margin: 4px 0 0 0;">
                Real-time interactive sentiment classification consuming Phase 2 FastAPI REST API.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not is_healthy:
    render_alert_box(
        title="FastAPI Backend Offline",
        message="The prediction service is unreachable. Please ensure the backend is running at <code>http://localhost:8000</code>.",
        alert_type="danger",
        icon="🚨",
    )

tab_single, tab_batch = st.tabs(["✍️ Single Review Inference", "📁 Batch File Classification"])

# -----------------------------------------------------------------------------
# TAB 1: SINGLE REVIEW INFERENCE
# -----------------------------------------------------------------------------
with tab_single:
    st.markdown("### 📝 Enter Review Text")
    st.caption("Click a sample preset below or enter custom text to test live inference.")

    # Preset sample reviews
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_text = ""
    if col_p1.button("🟢 Positive Sample", use_container_width=True):
        st.session_state["review_input"] = (
            "This movie was an absolute masterpiece! Incredible acting, brilliant cinematography, and a deeply moving plot."
        )
    if col_p2.button("🔴 Negative Sample", use_container_width=True):
        st.session_state["review_input"] = (
            "Complete waste of time and money. Terrible dialogue, awful pacing, and the worst acting I have ever seen."
        )
    if col_p3.button("🟡 Nuanced Review", use_container_width=True):
        st.session_state["review_input"] = (
            "While the visual effects and musical score were breathtaking, the storytelling felt convoluted and dragged on for far too long."
        )

    review_text = st.text_area(
        label="Input Text",
        value=st.session_state.get("review_input", ""),
        height=140,
        placeholder="Type or paste text here (e.g., 'The service was incredible and staff was very polite')...",
        help="Input text will be processed via Phase 1 TF-IDF vectorizer and classified by Logistic Regression.",
    )

    col_btn, col_clear = st.columns([1, 4])
    with col_btn:
        predict_clicked = st.button("🚀 Classify Sentiment", type="primary", use_container_width=True)

    if predict_clicked:
        if not review_text or not review_text.strip():
            render_alert_box(
                title="Input Required",
                message="Please enter a valid non-empty review text before submitting.",
                alert_type="warning",
                icon="⚠️",
            )
        else:
            with st.spinner("Calling FastAPI POST /predict endpoint..."):
                success, result, pred_latency = make_prediction(review_text)

            if success:
                st.session_state["last_prediction"] = result
                st.success(f"Inference completed in {pred_latency} ms!")
            else:
                err_msg = result.get("error", "Failed to get prediction from backend.")
                render_alert_box(
                    title="Prediction Error",
                    message=f"API Error: {err_msg}",
                    alert_type="danger",
                    icon="❌",
                )

    # Render Prediction Output if available
    last_pred = st.session_state.get("last_prediction")
    if last_pred:
        st.markdown("<hr style='border-color: #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        render_section_heading("Inference Result Telemetry", "Details returned by FastAPI POST /predict", "📊")

        sentiment = str(last_pred.get("sentiment", "unknown")).lower()
        confidence = float(last_pred.get("confidence", 0.0))
        model_ver = last_pred.get("model_version", "0.1.0")
        pred_id = last_pred.get("prediction_id", "N/A")
        pred_time = last_pred.get("timestamp", "N/A")
        pred_latency = last_pred.get("latency_ms", 0.0)

        c1, c2, c3 = st.columns(3)
        with c1:
            render_kpi_card("Predicted Class", sentiment.upper(), accent_color="#3B82F6", subtext="IMDb Binary Model")
        with c2:
            render_kpi_card("Model Confidence", f"{confidence * 100:.2f}%", accent_color="#10B981", subtext="Probability Score")
        with c3:
            render_kpi_card("API Latency", f"{pred_latency} ms", accent_color="#8B5CF6", subtext="Round-trip time")

        st.markdown(
            f"""
            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 1.25rem; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC;">
                        Classification: {render_sentiment_badge(sentiment)}
                    </div>
                    <div style="font-size: 0.82rem; color: #94A3B8;">
                        Model Release: <span style="color: #60A5FA; font-weight: 600;">v{model_ver}</span>
                    </div>
                </div>
                <div style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Confidence Score</span>
                        <span style="font-weight: 700;">{confidence * 100:.2f}%</span>
                    </div>
                    <div style="background: #0F172A; border-radius: 9999px; height: 10px; overflow: hidden; width: 100%;">
                        <div style="background: {'#10B981' if sentiment == 'positive' else '#EF4444'}; width: {confidence * 100:.2f}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #334155; font-size: 0.82rem; color: #94A3B8;">
                    <div><strong>Prediction ID:</strong><br><code style="color: #38BDF8;">{pred_id}</code></div>
                    <div><strong>Timestamp (UTC):</strong><br><span style="color: #E2E8F0;">{pred_time}</span></div>
                    <div><strong>Text Character Length:</strong><br><span style="color: #E2E8F0;">{len(review_text)} chars</span></div>
                    <div><strong>Estimated Tokens:</strong><br><span style="color: #E2E8F0;">{len(review_text.split())} words</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# TAB 2: BATCH FILE CLASSIFICATION
# -----------------------------------------------------------------------------
with tab_batch:
    st.markdown("### 📁 Upload Batch Dataset")
    st.caption("Upload a `.csv` file (must contain a text column) or a `.txt` file (one review per line).")

    uploaded_file = st.file_uploader("Choose CSV or TXT File", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"Uploaded `{uploaded_file.name}` ({len(df_upload)} rows)")
                text_col = st.selectbox("Select Text Column", options=df_upload.columns.tolist())
                reviews_to_predict = df_upload[text_col].dropna().astype(str).tolist()
            else:
                content = uploaded_file.getvalue().decode("utf-8")
                reviews_to_predict = [line.strip() for line in content.splitlines() if line.strip()]
                st.write(f"Uploaded TXT file with {len(reviews_to_predict)} lines.")

            if st.button("⚡ Run Batch Classification", type="primary"):
                if not reviews_to_predict:
                    st.warning("No valid text entries found in file.")
                else:
                    results_list = []
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    for idx, text_item in enumerate(reviews_to_predict[:50]):  # cap batch at 50 for speed
                        status_text.text(f"Classifying {idx + 1}/{min(len(reviews_to_predict), 50)}...")
                        ok, res, _ = make_prediction(text_item)
                        if ok:
                            results_list.append({
                                "Prediction ID": res.get("prediction_id", "N/A")[:8],
                                "Input Text": text_item,
                                "Sentiment": res.get("sentiment", "unknown").upper(),
                                "Confidence": f"{float(res.get('confidence', 0.0)) * 100:.2f}%",
                                "Model Version": res.get("model_version", "0.1.0"),
                            })
                        progress_bar.progress((idx + 1) / min(len(reviews_to_predict), 50))

                    status_text.empty()
                    progress_bar.empty()

                    if results_list:
                        batch_df = pd.DataFrame(results_list)
                        st.success(f"Batch completed! Processed {len(batch_df)} reviews.")
                        st.dataframe(batch_df, use_container_width=True)

                        # CSV Export
                        csv_data = batch_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Batch Results (CSV)",
                            data=csv_data,
                            file_name="sentiment_batch_results.csv",
                            mime="text/csv",
                        )
        except Exception as exc:
            st.error(f"Failed to process file: {exc}")


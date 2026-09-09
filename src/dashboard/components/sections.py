"""Reusable section layout components for dashboard pages."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st
from src.dashboard.components.badges import render_badge, render_sentiment_badge


def render_section_heading(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    badge_html: Optional[str] = None,
) -> None:
    """Render a styled section header with optional icon and badge."""
    icon_str = f"<span style='margin-right: 8px;'>{icon}</span>" if icon else ""
    badge_str = f"<div style='margin-left: auto;'>{badge_html}</div>" if badge_html else ""
    subtitle_str = (
        f"<div style='color: #94A3B8; font-size: 0.82rem; margin-top: 2px;'>{subtitle}</div>"
        if subtitle
        else ""
    )

    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        ">
            <div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; display: flex; align-items: center;">
                    {icon_str}{title}
                </div>
                {subtitle_str}
            </div>
            {badge_str}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_health_grid(
    health_dict: Dict[str, Any],
    latency_ms: Optional[float] = None,
) -> None:
    """Render a responsive diagnostic grid for system services."""
    services = [
        ("FastAPI Gateway", health_dict.get("api", False), "HTTP API Core"),
        ("SQLite Database", health_dict.get("database_connected", False), "Prediction Repository"),
        ("NLP Model", health_dict.get("model_loaded", False), "Logistic Regression"),
        ("TF-IDF Vectorizer", health_dict.get("vectorizer_loaded", False), "Fitted 10k Vocab"),
    ]

    cols = st.columns(4)
    for col, (name, is_up, detail) in zip(cols, services):
        with col:
            status_badge = render_badge("ONLINE" if is_up else "OFFLINE", "success" if is_up else "danger")
            border_color = "rgba(16, 185, 129, 0.2)" if is_up else "rgba(239, 68, 68, 0.2)"
            bg_color = "rgba(16, 185, 129, 0.04)" if is_up else "rgba(239, 68, 68, 0.04)"
            st.markdown(
                f"""
                <div style="
                    background: {bg_color};
                    border: 1px solid {border_color};
                    border-radius: 8px;
                    padding: 0.8rem;
                    height: 100%;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #E2E8F0;">{name}</span>
                        {status_badge}
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_model_info_card(
    model_info: Dict[str, Any],
    model_version: str = "0.1.0",
) -> None:
    """Render model architecture and vectorizer metadata."""
    m_name = model_info.get("model_name", "LogisticRegression")
    vectorizer = model_info.get("vectorizer", {})
    v_type = vectorizer.get("type", "TfidfVectorizer")
    max_features = vectorizer.get("max_features", "10,000")
    ngram = vectorizer.get("ngram_range", "(1, 2)")
    vocab_size = vectorizer.get("vocabulary_size", "10,000")
    classes = model_info.get("classes", ["negative", "positive"])

    st.markdown(
        f"""
        <div style="
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 1rem;
        ">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
                <div>
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Classifier Algorithm</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin-top: 2px;">{m_name}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Model Version</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #38BDF8; margin-top: 2px;">v{model_version}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Feature Extractor</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin-top: 2px;">{v_type}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">N-Gram Range & Features</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin-top: 2px;">{ngram} | {max_features}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Active Binary Classes</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #A78BFA; margin-top: 2px;">{', '.join(classes)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recent_predictions_table(predictions: List[Dict[str, Any]]) -> None:
    """Render the recent predictions table strictly showing binary sentiment."""
    if not predictions:
        st.markdown(
            """
            <div style="
                background: rgba(30, 41, 59, 0.4);
                border: 1px dashed rgba(255, 255, 255, 0.15);
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
                color: #94A3B8;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">📭</div>
                <div style="font-weight: 600; color: #E2E8F0;">No Predictions in Database</div>
                <div style="font-size: 0.8rem; margin-top: 4px;">
                    Send inference requests to <code>/predict</code> or use the <b>02 Live Prediction</b> tab to log records.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    table_rows = []
    for pred in predictions:
        pid = str(pred.get("prediction_id", ""))[:8] + "..."
        raw_text = str(pred.get("text", ""))
        text_preview = (raw_text[:75] + "...") if len(raw_text) > 75 else raw_text
        sentiment = pred.get("sentiment", "unknown")
        badge = render_sentiment_badge(sentiment)
        confidence = pred.get("confidence", 0.0)
        conf_str = f"{confidence * 100:.1f}%" if confidence <= 1.0 else f"{confidence:.1f}%"
        ts = str(pred.get("timestamp", ""))[:19].replace("T", " ")
        m_ver = pred.get("model_version", "0.1.0")

        table_rows.append(
            f"""
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                <td style="padding: 8px 12px; font-family: monospace; font-size: 0.78rem; color: #94A3B8;">{pid}</td>
                <td style="padding: 8px 12px; font-size: 0.82rem; color: #E2E8F0;" title="{raw_text}">{text_preview}</td>
                <td style="padding: 8px 12px;">{badge}</td>
                <td style="padding: 8px 12px; font-size: 0.82rem; font-weight: 600; color: #F1F5F9;">{conf_str}</td>
                <td style="padding: 8px 12px; font-size: 0.78rem; color: #94A3B8;">{ts}</td>
                <td style="padding: 8px 12px; font-size: 0.78rem; color: #64748B;">v{m_ver}</td>
            </tr>
            """
        )

    rows_html = "\n".join(table_rows)
    st.markdown(
        f"""
        <div style="
            overflow-x: auto;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.5);
        ">
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="background: rgba(30, 41, 59, 0.7); border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                        <th style="padding: 10px 12px; font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">ID</th>
                        <th style="padding: 10px 12px; font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Input Text</th>
                        <th style="padding: 10px 12px; font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Sentiment</th>
                        <th style="padding: 10px 12px; font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Confidence</th>
                        <th style="padding: 10px 12px; font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Timestamp</th>
                        <th style="padding: 10px 12px; font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Model</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

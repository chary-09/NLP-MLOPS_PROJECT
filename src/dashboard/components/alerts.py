"""Alert helper components for dashboard telemetry and monitoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st


def render_alert_box(
    level: str,
    title: str,
    message: str,
    metric: Optional[str] = None,
    value: Optional[Any] = None,
    threshold: Optional[Any] = None,
) -> None:
    """Render a styled alert container for monitoring events."""
    lvl = level.upper()
    if lvl in ("CRITICAL", "ERROR", "DANGER"):
        border_color = "rgba(239, 68, 68, 0.4)"
        bg_color = "rgba(239, 68, 68, 0.08)"
        text_color = "#F87171"
        icon = "🚨"
    elif lvl in ("WARNING", "WARN"):
        border_color = "rgba(245, 158, 11, 0.4)"
        bg_color = "rgba(245, 158, 11, 0.08)"
        text_color = "#FBBF24"
        icon = "⚠️"
    else:
        border_color = "rgba(59, 130, 246, 0.4)"
        bg_color = "rgba(59, 130, 246, 0.08)"
        text_color = "#60A5FA"
        icon = "ℹ️"

    metric_details = ""
    if metric or value is not None or threshold is not None:
        parts = []
        if metric:
            parts.append(f"<b>Metric:</b> <code>{metric}</code>")
        if value is not None:
            parts.append(f"<b>Current:</b> <code>{value}</code>")
        if threshold is not None:
            parts.append(f"<b>Threshold:</b> <code>{threshold}</code>")
        metric_details = f"<div style='margin-top: 6px; font-size: 0.78rem; opacity: 0.85;'>{' &bull; '.join(parts)}</div>"

    html = f"""
    <div style="
        background: {bg_color};
        border-left: 4px solid {text_color};
        border-top: 1px solid {border_color};
        border-right: 1px solid {border_color};
        border-bottom: 1px solid {border_color};
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-weight: 600; color: {text_color}; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">
                <span>{icon}</span> {title}
            </div>
            <span style="font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; background: {border_color}; color: {text_color};">
                {lvl}
            </span>
        </div>
        <div style="color: #CBD5E1; font-size: 0.82rem; margin-top: 4px; line-height: 1.4;">
            {message}
        </div>
        {metric_details}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_recent_alerts_section(alerts: List[Dict[str, Any]]) -> None:
    """Render the list of recent threshold monitoring alerts, or nominal state."""
    if not alerts:
        st.markdown(
            """
            <div style="
                background: rgba(16, 185, 129, 0.06);
                border: 1px solid rgba(16, 185, 129, 0.25);
                border-radius: 8px;
                padding: 1rem 1.25rem;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <span style="font-size: 1.4rem;">✅</span>
                <div>
                    <div style="color: #34D399; font-weight: 600; font-size: 0.9rem;">
                        All Systems Nominal
                    </div>
                    <div style="color: #94A3B8; font-size: 0.78rem;">
                        No active monitoring alerts or threshold violations detected across latency, drift, confidence, or error rates.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for alert_item in alerts:
        level = alert_item.get("level", "WARNING")
        title = alert_item.get("name", alert_item.get("type", "Threshold Alert"))
        message = alert_item.get("message", "Metric threshold triggered.")
        metric = alert_item.get("metric")
        current_val = alert_item.get("current_value")
        threshold_val = alert_item.get("threshold")
        render_alert_box(
            level=level,
            title=title,
            message=message,
            metric=metric,
            value=current_val,
            threshold=threshold_val,
        )


def alert(message: str, alert_type: str = "warning") -> None:
    """Render basic alert message (backward compatibility)."""
    if alert_type == "error":
        st.error(message)
    elif alert_type == "success":
        st.success(message)
    elif alert_type == "info":
        st.info(message)
    else:
        st.warning(message)

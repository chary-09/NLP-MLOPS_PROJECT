"""Reusable MLOps card components for layout, KPI metrics, and model specifications."""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st


def render_card_header(title: str, subtitle: Optional[str] = None, badge_html: Optional[str] = None) -> str:
    """Generate HTML for a polished card header."""
    sub_html = f'<p class="mlops-card-subtitle">{subtitle}</p>' if subtitle else ""
    badge = badge_html if badge_html else ""
    return f"""
    <div class="mlops-card-header">
        <div>
            <h3 class="mlops-card-title">{title}</h3>
            {sub_html}
        </div>
        <div>{badge}</div>
    </div>
    """


def render_kpi_card(
    label: str,
    value: str,
    subtext: Optional[str] = None,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    accent: Optional[str] = None,
) -> None:
    """Render a KPI metric card with value, label, and optional delta indicator.
    
    Args:
        label: Metric title (e.g. 'TOTAL PREDICTIONS')
        value: Primary numerical or percentage display
        subtext: Context string underneath
        delta: Optional delta badge string (e.g. '+4.2%')
        delta_positive: Whether delta represents favorable trend
        accent: Optional color accent ('emerald', 'rose', 'blue', 'amber', 'purple')
    """
    delta_html = ""
    if delta:
        color = "#34D399" if delta_positive else "#F87171"
        icon = "▲" if delta_positive else "▼"
        delta_html = f'<span style="color: {color}; font-weight: 600; margin-right: 0.3rem;">{icon} {delta}</span>'

    sub_html = ""
    if subtext or delta:
        sub_html = f'<div class="kpi-subtext">{delta_html}<span>{subtext or ""}</span></div>'

    accent_border = ""
    val_color = "#FFFFFF"
    if accent == "emerald":
        accent_border = "border-top: 3px solid #10B981;"
        val_color = "#34D399"
    elif accent == "rose":
        accent_border = "border-top: 3px solid #EF4444;"
        val_color = "#F87171"
    elif accent == "blue":
        accent_border = "border-top: 3px solid #3B82F6;"
        val_color = "#60A5FA"
    elif accent == "purple":
        accent_border = "border-top: 3px solid #8B5CF6;"
        val_color = "#C084FC"
    elif accent == "amber":
        accent_border = "border-top: 3px solid #F59E0B;"
        val_color = "#FBBF24"

    html = f"""
    <div class="kpi-card" style="{accent_border}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {val_color};">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    subtext: Optional[str] = None,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    accent: Optional[str] = None,
) -> None:
    """Alias for render_kpi_card for flexible component usage."""
    render_kpi_card(
        label=label,
        value=value,
        subtext=subtext,
        delta=delta,
        delta_positive=delta_positive,
        accent=accent,
    )


def render_spec_grid(specs: Dict[str, Any]) -> None:
    """Render a clean grid of specification key-value items."""
    items_html = ""
    for k, v in specs.items():
        items_html += f"""
        <div class="spec-item">
            <div class="spec-key">{k}</div>
            <div class="spec-val">{v}</div>
        </div>
        """
    html = f'<div class="spec-grid">{items_html}</div>'
    st.markdown(html, unsafe_allow_html=True)

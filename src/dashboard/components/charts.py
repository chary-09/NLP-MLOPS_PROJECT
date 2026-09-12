"""Reusable Plotly chart components for the SentimentOps analytics dashboard.

All chart functions render directly via Streamlit and consume only real data
passed in — no fabrication, no inline model calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Shared visual constants — consistent with the dashboard dark theme
# ---------------------------------------------------------------------------
_BG_COLOR = "rgba(0,0,0,0)"
_PAPER_BG = "rgba(15,23,42,0.0)"
_FONT_COLOR = "#CBD5E1"
_GRID_COLOR = "rgba(148,163,184,0.12)"
_POSITIVE_COLOR = "#10B981"
_NEGATIVE_COLOR = "#F87171"
_NEUTRAL_COLOR = "#94A3B8"
_CONFIDENCE_HIGH = "#10B981"
_CONFIDENCE_MED = "#F59E0B"
_CONFIDENCE_LOW = "#EF4444"

_SENTIMENT_PALETTE = {
    "positive": _POSITIVE_COLOR,
    "negative": _NEGATIVE_COLOR,
    "neutral": _NEUTRAL_COLOR,
}


def _base_layout(**overrides) -> Dict[str, Any]:
    """Return a sensible Plotly layout dict for dark-themed charts."""
    layout: Dict[str, Any] = dict(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_BG_COLOR,
        font=dict(family="Inter, system-ui, sans-serif", color=_FONT_COLOR, size=12),
        margin=dict(l=16, r=16, t=40, b=24),
        legend=dict(
            bgcolor="rgba(15,23,42,0.6)",
            bordercolor="rgba(148,163,184,0.2)",
            borderwidth=1,
            font=dict(color=_FONT_COLOR, size=11),
        ),
        xaxis=dict(
            gridcolor=_GRID_COLOR,
            linecolor="rgba(148,163,184,0.2)",
            tickfont=dict(color=_FONT_COLOR, size=11),
        ),
        yaxis=dict(
            gridcolor=_GRID_COLOR,
            linecolor="rgba(148,163,184,0.2)",
            tickfont=dict(color=_FONT_COLOR, size=11),
        ),
    )
    layout.update(overrides)
    return layout


def _render_empty_chart(title: str, message: str) -> None:
    """Styled placeholder when chart data is absent."""
    st.markdown(
        f"""
        <div style="
            background: rgba(15,23,42,0.6);
            border: 1px dashed rgba(148,163,184,0.25);
            border-radius: 10px;
            padding: 2rem 1.5rem;
            text-align: center;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <div style="color: #F8FAFC; font-weight: 600; font-size: 0.95rem;">{title}</div>
            <div style="color: #64748B; font-size: 0.82rem; margin-top: 0.25rem;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 1. Sentiment Donut Chart
# ---------------------------------------------------------------------------

def render_sentiment_donut(
    sentiment_counts: Dict[str, int],
    title: str = "Sentiment Distribution",
) -> None:
    """Render a donut chart for binary sentiment distribution."""

    if not sentiment_counts or sum(sentiment_counts.values()) == 0:
        _render_empty_chart(title, "No prediction data available yet.")
        return

    labels = list(sentiment_counts.keys())
    values = [int(sentiment_counts[k]) for k in labels]
    colors = [_SENTIMENT_PALETTE.get(lbl.lower(), "#64748B") for lbl in labels]

    fig = go.Figure(
        go.Pie(
            labels=[lbl.capitalize() for lbl in labels],
            values=values,
            hole=0.58,
            marker=dict(colors=colors, line=dict(color="rgba(15,23,42,0.8)", width=2)),
            textinfo="percent+label",
            textfont=dict(size=12, color="#F8FAFC"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )
    )
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total:,}</b><br>total",
        x=0.5, y=0.5,
        font=dict(size=14, color="#F8FAFC"),
        showarrow=False,
        align="center",
    )
    layout = _base_layout(title=dict(text=title, font=dict(color="#F8FAFC", size=14), x=0.02))
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 2. Sentiment Horizontal Bar
# ---------------------------------------------------------------------------

def render_sentiment_bar(
    sentiment_counts: Dict[str, int],
    title: str = "Prediction Volume by Sentiment",
) -> None:
    """Horizontal bar chart for sentiment counts."""

    if not sentiment_counts or sum(sentiment_counts.values()) == 0:
        _render_empty_chart(title, "No prediction data available yet.")
        return

    labels = [lbl.capitalize() for lbl in sentiment_counts.keys()]
    values = list(sentiment_counts.values())
    colors = [_SENTIMENT_PALETTE.get(lbl.lower(), "#64748B") for lbl in sentiment_counts.keys()]

    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[f"{v:,}" for v in values],
            textposition="inside",
            textfont=dict(color="#F8FAFC", size=12),
            hovertemplate="<b>%{y}</b>: %{x:,} predictions<extra></extra>",
        )
    )
    layout = _base_layout(title=dict(text=title, font=dict(color="#F8FAFC", size=14), x=0.02))
    layout["xaxis"]["title"] = dict(text="Predictions", font=dict(color=_FONT_COLOR))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 3. Prediction Trend — area chart
# ---------------------------------------------------------------------------

def render_prediction_trend(
    trend_data: List[Dict[str, Any]],
    grouping: str = "daily",
    title: str = "Prediction Volume Over Time",
) -> None:
    """Area chart showing prediction count per time bucket.

    Args:
        trend_data: list of dicts with "period", "count", and optionally
                    "positive" / "negative" sub-counts.
        grouping:   "hourly" | "daily" | "weekly"
        title:      Chart title
    """

    if not trend_data:
        _render_empty_chart(title, "Not enough time-series data to plot.")
        return

    periods = [d.get("period", "") for d in trend_data]
    totals = [int(d.get("count", 0)) for d in trend_data]
    positives = [int(d.get("positive", 0)) for d in trend_data]
    negatives = [int(d.get("negative", 0)) for d in trend_data]

    fig = go.Figure()
    has_breakdown = any(p > 0 for p in positives) or any(n > 0 for n in negatives)

    if has_breakdown:
        fig.add_trace(go.Scatter(
            x=periods, y=positives, mode="lines",
            name="Positive",
            line=dict(color=_POSITIVE_COLOR, width=2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.15)",
            hovertemplate="<b>%{x}</b><br>Positive: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=periods, y=negatives, mode="lines",
            name="Negative",
            line=dict(color=_NEGATIVE_COLOR, width=2),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.13)",
            hovertemplate="<b>%{x}</b><br>Negative: %{y:,}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=periods, y=totals, mode="lines+markers",
            name="Total",
            line=dict(color="#60A5FA", width=2.5),
            marker=dict(color="#60A5FA", size=5),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.12)",
            hovertemplate="<b>%{x}</b><br>Predictions: %{y:,}<extra></extra>",
        ))

    layout = _base_layout(
        title=dict(text=f"{title} ({grouping})", font=dict(color="#F8FAFC", size=14), x=0.02),
    )
    layout["xaxis"]["title"] = dict(text=grouping.capitalize(), font=dict(color=_FONT_COLOR))
    layout["yaxis"]["title"] = dict(text="Predictions", font=dict(color=_FONT_COLOR))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 4. Confidence Distribution — histogram
# ---------------------------------------------------------------------------

def render_confidence_histogram(
    confidence_values: List[float],
    title: str = "Confidence Score Distribution",
) -> None:
    """Histogram of raw confidence values with Low / Medium / High bands."""

    if not confidence_values:
        _render_empty_chart(title, "No confidence data available.")
        return

    fig = go.Figure()
    bands = [
        (0.0, 0.50, "rgba(239,68,68,0.08)", "Low"),
        (0.50, 0.80, "rgba(245,158,11,0.08)", "Medium"),
        (0.80, 1.01, "rgba(16,185,129,0.08)", "High"),
    ]
    for x0, x1, color, label in bands:
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=color, layer="below", line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(size=10, color=_FONT_COLOR),
        )

    fig.add_trace(go.Histogram(
        x=confidence_values, nbinsx=20,
        marker=dict(color="#60A5FA", line=dict(color="rgba(15,23,42,0.8)", width=1)),
        name="Predictions",
        hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    layout = _base_layout(title=dict(text=title, font=dict(color="#F8FAFC", size=14), x=0.02))
    layout["xaxis"].update(title=dict(text="Confidence Score (0 → 1)", font=dict(color=_FONT_COLOR)))
    layout["yaxis"].update(title=dict(text="Prediction Count", font=dict(color=_FONT_COLOR)))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 5. Confidence Range Bar — Low / Medium / High bucket counts
# ---------------------------------------------------------------------------

def render_confidence_range_bar(
    confidence_values: List[float],
    title: str = "Confidence Range Breakdown",
) -> None:
    """Bar chart of Low / Medium / High confidence bucket counts."""

    if not confidence_values:
        _render_empty_chart(title, "No confidence data available.")
        return

    low = sum(1 for c in confidence_values if c < 0.50)
    med = sum(1 for c in confidence_values if 0.50 <= c < 0.80)
    high = sum(1 for c in confidence_values if c >= 0.80)

    labels = ["Low (<50%)", "Medium (50–80%)", "High (≥80%)"]
    values = [low, med, high]
    colors = [_CONFIDENCE_LOW, _CONFIDENCE_MED, _CONFIDENCE_HIGH]

    fig = go.Figure(
        go.Bar(
            x=labels, y=values,
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[f"{v:,}" for v in values],
            textposition="outside",
            textfont=dict(color=_FONT_COLOR, size=12),
            hovertemplate="<b>%{x}</b>: %{y:,} predictions<extra></extra>",
        )
    )
    layout = _base_layout(title=dict(text=title, font=dict(color="#F8FAFC", size=14), x=0.02))
    layout["yaxis"]["title"] = dict(text="Count", font=dict(color=_FONT_COLOR))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Legacy shim
# ---------------------------------------------------------------------------

def chart_data(values: list) -> dict:
    """Legacy helper retained for backward compatibility."""
    return {"values": values}


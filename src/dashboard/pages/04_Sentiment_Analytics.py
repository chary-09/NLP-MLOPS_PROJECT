"""04 Sentiment Analytics — SentimentOps Studio.

Consumes REAL stored prediction data exclusively from the FastAPI backend
via GET /predictions.  No NLP model is loaded or called here.

Charts:
    1. Sentiment Distribution (Donut)
    2. Sentiment Volume Bar
    3. Prediction Trend / Volume Over Time (Area)
    4. Confidence Distribution (Histogram)
    5. Confidence Range Breakdown (Bar)

Filters:
    - Date range
    - Sentiment class
    - Model version
    - Confidence range

Summary metrics:
    - Total predictions
    - Positive %
    - Negative %
    - Average confidence
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.dashboard.components.alerts import render_alert_box
from src.dashboard.components.cards import render_kpi_card
from src.dashboard.components.charts import (
    render_confidence_histogram,
    render_confidence_range_bar,
    render_prediction_trend,
    render_sentiment_bar,
    render_sentiment_donut,
)
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.config import (
    APP_NAME,
    LAYOUT,
    fetch_analytics_data,
    fetch_api_health,
)

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"04 Sentiment Analytics | {APP_NAME}",
    page_icon="📈",
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


# ─── CSS loader ────────────────────────────────────────────────────────────────
def _load_css() -> None:
    css_dir = Path(__file__).parent.parent / "styles"
    for css_file in ["main.css", "cards.css", "sidebar.css"]:
        full_path = css_dir / css_file
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as fh:
                st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


_load_css()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
is_healthy, health_data, latency_ms = fetch_api_health()
render_sidebar(api_healthy=is_healthy, latency_ms=latency_ms, health_data=health_data)

# ─── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.4rem;">
        <span style="font-size: 2rem;">📈</span>
        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC;">
            Sentiment Analytics
        </h1>
    </div>
    <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 1.5rem;">
        Temporal trends, confidence distributions, and sentiment breakdowns — driven entirely
        by real stored prediction data from the backend database.
    </p>
    """,
    unsafe_allow_html=True,
)

# ─── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Fetching prediction records from backend…"):
    success, raw_records, error_msg = fetch_analytics_data(limit=500)

if not success:
    render_alert_box(
        title="Backend Unavailable",
        message=error_msg or "Cannot reach the FastAPI backend. Start the server and refresh.",
        alert_type="danger",
        icon="❌",
    )
    st.stop()

if not raw_records:
    render_alert_box(
        title="No Predictions Yet",
        message=(
            "The prediction database is empty. "
            "Make at least one prediction via the Live Prediction page, then refresh."
        ),
        alert_type="warning",
        icon="⚠️",
    )
    st.stop()


# ─── Helper: safe timestamp parse ──────────────────────────────────────────────

def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string safely, returning None on failure."""
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f+00:00",
                "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    return None


# ─── Pre-process records ───────────────────────────────────────────────────────

def _process_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich records with parsed datetime for filtering/grouping."""
    processed = []
    for rec in records:
        ts = _parse_ts(rec.get("timestamp", ""))
        processed.append({**rec, "_dt": ts})
    return processed


all_records = _process_records(raw_records)

# Collect filter option values from actual data
all_sentiments = sorted({r.get("sentiment", "").lower() for r in all_records if r.get("sentiment")})
all_versions = sorted({r.get("model_version", "unknown") for r in all_records if r.get("model_version")})
all_dts = [r["_dt"] for r in all_records if r["_dt"] is not None]
min_date = min(all_dts).date() if all_dts else None
max_date = max(all_dts).date() if all_dts else None

# ─── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='color:#60A5FA; font-weight:700; font-size:0.78rem; letter-spacing:0.08em;'>ANALYTICS FILTERS</div>",
    unsafe_allow_html=True,
)

# Date range
if min_date and max_date:
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="analytics_date_range",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filter_start, filter_end = date_range
    else:
        filter_start, filter_end = min_date, max_date
else:
    filter_start, filter_end = None, None

# Sentiment
selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=[s.capitalize() for s in all_sentiments],
    default=[s.capitalize() for s in all_sentiments],
    key="analytics_sentiment",
)
selected_sentiments_lower = [s.lower() for s in selected_sentiments]

# Model version
selected_versions = st.sidebar.multiselect(
    "Model Version",
    options=all_versions,
    default=all_versions,
    key="analytics_model_version",
)

# Confidence range
conf_min, conf_max = st.sidebar.slider(
    "Confidence Range",
    min_value=0.0, max_value=1.0,
    value=(0.0, 1.0),
    step=0.05,
    key="analytics_confidence",
)

# Trend grouping
grouping = st.sidebar.selectbox(
    "Time Grouping",
    options=["hourly", "daily", "weekly"],
    index=1,
    key="analytics_grouping",
)


# ─── Apply filters ─────────────────────────────────────────────────────────────

def _apply_filters(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = []
    for rec in records:
        sentiment = rec.get("sentiment", "").lower()
        version = rec.get("model_version", "unknown")
        confidence = float(rec.get("confidence", 0.0))
        dt = rec.get("_dt")

        # Sentiment filter
        if selected_sentiments_lower and sentiment not in selected_sentiments_lower:
            continue
        # Version filter
        if selected_versions and version not in selected_versions:
            continue
        # Confidence filter
        if not (conf_min <= confidence <= conf_max):
            continue
        # Date range filter
        if filter_start and filter_end and dt is not None:
            if not (filter_start <= dt.date() <= filter_end):
                continue
        filtered.append(rec)
    return filtered


filtered_records = _apply_filters(all_records)

if not filtered_records:
    render_alert_box(
        title="No Matching Records",
        message="The current filter combination returned 0 predictions. Adjust the filters in the sidebar.",
        alert_type="warning",
        icon="⚠️",
    )
    st.stop()

# ─── Derive analytics from filtered records ────────────────────────────────────
total_count = len(filtered_records)
sentiment_counts: Dict[str, int] = Counter(
    r.get("sentiment", "unknown").lower() for r in filtered_records
)
positive_count = sentiment_counts.get("positive", 0)
negative_count = sentiment_counts.get("negative", 0)
positive_pct = round(positive_count / total_count * 100, 1) if total_count else 0.0
negative_pct = round(negative_count / total_count * 100, 1) if total_count else 0.0
confidence_values: List[float] = [
    float(r.get("confidence", 0.0)) for r in filtered_records
]
avg_confidence = round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0.0


# ─── Section divider helper ────────────────────────────────────────────────────
def _section(title: str, subtitle: str = "") -> None:
    sub = f"<p style='color:#64748B; font-size:0.82rem; margin:0;'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div style="margin: 1.5rem 0 0.75rem 0;">
            <h3 style="margin:0; font-size:1.05rem; font-weight:700; color:#F8FAFC;">{title}</h3>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── 1. Summary KPI Row ────────────────────────────────────────────────────────
_section("Summary Metrics", f"Showing {total_count:,} filtered prediction(s)")

kpi_cols = st.columns(4)
with kpi_cols[0]:
    render_kpi_card(label="TOTAL PREDICTIONS", value=f"{total_count:,}", accent_color="blue")
with kpi_cols[1]:
    render_kpi_card(label="POSITIVE", value=f"{positive_pct}%",
                    subtext=f"{positive_count:,} records", accent_color="emerald")
with kpi_cols[2]:
    render_kpi_card(label="NEGATIVE", value=f"{negative_pct}%",
                    subtext=f"{negative_count:,} records", accent_color="rose")
with kpi_cols[3]:
    render_kpi_card(label="AVG CONFIDENCE", value=f"{avg_confidence:.1%}",
                    subtext="across filtered set", accent_color="purple")

st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

# ─── 2. Sentiment Distribution ─────────────────────────────────────────────────
_section(
    "Sentiment Distribution",
    "Binary sentiment breakdown — Positive vs Negative based on stored model outputs.",
)
dist_col1, dist_col2 = st.columns(2, gap="medium")
with dist_col1:
    render_sentiment_donut(
        sentiment_counts=sentiment_counts,
        title="Sentiment Distribution (Donut)",
    )
with dist_col2:
    render_sentiment_bar(
        sentiment_counts=sentiment_counts,
        title="Prediction Volume by Sentiment",
    )

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

# ─── 3. Prediction Trend ───────────────────────────────────────────────────────
_section(
    "Prediction Trend",
    "Volume of predictions over time grouped by the selected time interval.",
)


def _build_trend(records: List[Dict[str, Any]], grouping: str) -> List[Dict[str, Any]]:
    """Aggregate predictions into time buckets for the trend chart."""
    bucket_totals: Dict[str, int] = defaultdict(int)
    bucket_pos: Dict[str, int] = defaultdict(int)
    bucket_neg: Dict[str, int] = defaultdict(int)

    for rec in records:
        dt = rec.get("_dt")
        if dt is None:
            continue
        if grouping == "hourly":
            key = dt.strftime("%Y-%m-%d %H:00")
        elif grouping == "weekly":
            key = dt.strftime("%Y-W%W")
        else:  # daily (default)
            key = dt.strftime("%Y-%m-%d")

        bucket_totals[key] += 1
        sentiment = rec.get("sentiment", "").lower()
        if sentiment == "positive":
            bucket_pos[key] += 1
        elif sentiment == "negative":
            bucket_neg[key] += 1

    if not bucket_totals:
        return []

    sorted_keys = sorted(bucket_totals.keys())
    return [
        {
            "period": k,
            "count": bucket_totals[k],
            "positive": bucket_pos.get(k, 0),
            "negative": bucket_neg.get(k, 0),
        }
        for k in sorted_keys
    ]


trend_data = _build_trend(filtered_records, grouping)
render_prediction_trend(trend_data=trend_data, grouping=grouping)

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

# ─── 4. Confidence Distribution ────────────────────────────────────────────────
_section(
    "Confidence Distribution",
    "Distribution of model confidence scores across all filtered predictions.",
)
conf_col1, conf_col2 = st.columns(2, gap="medium")
with conf_col1:
    render_confidence_histogram(
        confidence_values=confidence_values,
        title="Confidence Score Histogram",
    )
with conf_col2:
    render_confidence_range_bar(
        confidence_values=confidence_values,
        title="Confidence Range Breakdown",
    )

st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)

# ─── 5. Filtered Records Table ─────────────────────────────────────────────────
with st.expander("📋 View Filtered Prediction Records", expanded=False):
    if filtered_records:
        display_rows = []
        for rec in filtered_records[:200]:  # cap display for performance
            dt = rec.get("_dt")
            display_rows.append({
                "ID": rec.get("prediction_id", "—")[:12] + "…",
                "Sentiment": rec.get("sentiment", "—").capitalize(),
                "Confidence": f"{float(rec.get('confidence', 0)):.1%}",
                "Model Ver.": rec.get("model_version", "—"),
                "Timestamp": dt.strftime("%Y-%m-%d %H:%M:%S UTC") if dt else rec.get("timestamp", "—"),
                "Text": str(rec.get("text", ""))[:80] + ("…" if len(str(rec.get("text", ""))) > 80 else ""),
            })
        st.dataframe(display_rows, use_container_width=True, height=300)
        if len(filtered_records) > 200:
            st.caption(f"Showing first 200 of {len(filtered_records):,} filtered records.")
    else:
        st.info("No records to display.")

# ─── Footer note ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="margin-top:2rem; padding:0.75rem 1rem;
         background: rgba(15,23,42,0.5); border-radius:8px;
         border: 1px solid rgba(148,163,184,0.12);
         color:#64748B; font-size:0.78rem; text-align:center;">
        All data sourced exclusively from the backend database via
        <code>GET /predictions</code> · {total_count:,} record(s) in current view
        · No predictions are recalculated on this page
    </div>
    """,
    unsafe_allow_html=True,
)


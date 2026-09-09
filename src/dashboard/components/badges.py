"""Status badge and indicator UI components."""

from __future__ import annotations

from typing import Optional


def render_badge(
    text: str,
    badge_type: str = "neutral",
    pulse: bool = False,
    icon: Optional[str] = None,
) -> str:
    """Generate HTML string for a stylized MLOps pill badge.

    Args:
        text: Badge label
        badge_type: 'success' | 'warning' | 'danger' | 'info' | 'neutral'
        pulse: Whether to include a live pulsing dot
        icon: Optional prefix emoji or icon
    """
    pulse_html = f'<span class="pulse-dot {badge_type}"></span>' if pulse else ""
    icon_html = f"<span>{icon}</span> " if icon else ""
    return (
        f'<span class="badge-pill badge-{badge_type}">'
        f'{pulse_html}{icon_html}{text}'
        f'</span>'
    )


def render_status_badge(
    status: str,
    latency_ms: Optional[float] = None,
) -> str:
    """Render operational health badge (Live/Degraded/Offline) with latency."""
    s = str(status).lower()
    if s in ("healthy", "ok", "online"):
        badge_type = "success"
        label = "API LIVE"
        pulse = True
    elif s in ("degraded", "warning"):
        badge_type = "warning"
        label = "API DEGRADED"
        pulse = True
    else:
        badge_type = "danger"
        label = "API OFFLINE"
        pulse = False

    latency_str = f" ({latency_ms}ms)" if latency_ms is not None else ""
    return render_badge(f"{label}{latency_str}", badge_type=badge_type, pulse=pulse)


def render_drift_badge(status: str) -> str:
    """Render drift monitoring badge (NORMAL, DRIFT_DETECTED, INSUFFICIENT_DATA)."""
    s = str(status).upper()
    if s in ("NORMAL", "NO_DRIFT"):
        return render_badge("DRIFT: STABLE", badge_type="success", icon="🌊")
    elif s in ("DRIFT_DETECTED", "WARNING", "DRIFT"):
        return render_badge("DRIFT DETECTED", badge_type="danger", pulse=True, icon="⚠️")
    else:
        return render_badge("DRIFT: INSUFFICIENT DATA", badge_type="neutral", icon="ℹ️")


def render_binary_badge() -> str:
    """Render indicator that classification model is binary (Positive/Negative)."""
    return render_badge("BINARY: POSITIVE / NEGATIVE", badge_type="info", icon="⚖️")


def render_sentiment_badge(sentiment: str) -> str:
    """Render badge for sentiment class (strictly positive or negative)."""
    s = str(sentiment).strip().lower()
    if s == "positive":
        return render_badge("POSITIVE", badge_type="success", icon="▲")
    elif s == "negative":
        return render_badge("NEGATIVE", badge_type="danger", icon="▼")
    else:
        return render_badge(sentiment.upper(), badge_type="neutral")


def badge(status: str) -> str:
    """Backward compatibility alias."""
    return render_badge(status.upper(), badge_type="info")

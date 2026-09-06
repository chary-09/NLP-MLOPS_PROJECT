"""Status badge and indicator UI components."""

from __future__ import annotations

from typing import Optional
import streamlit as st


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


def badge(status: str) -> str:
    """Backward compatibility alias."""
    return render_badge(status.upper(), badge_type="info")

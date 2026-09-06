"""Theme definitions, color tokens, and styling helpers for the MLOps Dashboard.

Maintains visual consistency with a sleek, high-tech MLOps aesthetic:
deep slates, vibrant status badges, subtle glassmorphic borders, and high contrast typography.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

# Core Brand Palette (Preserve PRIMARY_COLOR for backward compatibility with existing tests)
PRIMARY_COLOR = "#2563EB"       # Royal Electric Blue
SECONDARY_COLOR = "#3B82F6"     # Bright Blue
BACKGROUND_COLOR = "#0B0F19"    # Deep Space Navy
SURFACE_COLOR = "#111827"       # Slate Surface Card
SURFACE_HOVER = "#1E293B"       # Slightly lighter surface
BORDER_COLOR = "#1E293B"        # Subtle slate border
BORDER_GLOW = "rgba(37, 99, 235, 0.25)"

# Semantic & Status Colors
SUCCESS_COLOR = "#10B981"       # Emerald (Healthy / Positive)
WARNING_COLOR = "#F59E0B"       # Amber (Degraded / Drift Alert)
DANGER_COLOR = "#EF4444"        # Crimson (Unhealthy / Error)
INFO_COLOR = "#06B6D4"          # Cyan (Information / Telemetry)
NEUTRAL_COLOR = "#64748B"       # Slate Muted Text

# Typography
FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, "
    "sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji'"
)
FONT_MONO = "SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace"

# Path to stylesheet assets
STYLES_DIR = Path(__file__).resolve().parent / "styles"


def load_stylesheet(filename: str = "main.css") -> str:
    """Read CSS stylesheet content from styles directory."""
    css_file = STYLES_DIR / filename
    if css_file.exists():
        try:
            return css_file.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


def inject_custom_css() -> None:
    """Inject all dashboard stylesheets into the active Streamlit app."""
    main_css = load_stylesheet("main.css")
    cards_css = load_stylesheet("cards.css")
    sidebar_css = load_stylesheet("sidebar.css")
    combined = f"{main_css}\n{cards_css}\n{sidebar_css}"
    if combined.strip():
        st.markdown(f"<style>{combined}</style>", unsafe_allow_html=True)


def get_status_color(status: str) -> str:
    """Map operational or sentiment status string to color token."""
    s = str(status).lower()
    if s in ("healthy", "positive", "online", "active", "up"):
        return SUCCESS_COLOR
    elif s in ("degraded", "warning", "drift_detected", "moderate"):
        return WARNING_COLOR
    elif s in ("unhealthy", "negative", "offline", "error", "down"):
        return DANGER_COLOR
    elif s in ("info", "neutral"):
        return INFO_COLOR
    return NEUTRAL_COLOR

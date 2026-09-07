"""Backward compatible KPI metric card helper."""

from src.dashboard.components.cards import render_kpi_card


def metric_card(label: str, value, delta=None) -> None:
    """Render metric card using MLOps theme styling."""
    render_kpi_card(label=label, value=str(value), delta=str(delta) if delta is not None else None)

"""Alert helper component."""

import streamlit as st


def alert(message: str, alert_type: str = "warning") -> None:
    """Render basic alert message."""
    if alert_type == "error":
        st.error(message)
    elif alert_type == "success":
        st.success(message)
    elif alert_type == "info":
        st.info(message)
    else:
        st.warning(message)

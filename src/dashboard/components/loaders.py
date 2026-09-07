"""Backward compatible loader spinner component."""

import streamlit as st
from src.dashboard.components.states import render_loading_state


def loader(message: str = "Loading..."):
    """Provide loading spinner context."""
    return st.spinner(message)

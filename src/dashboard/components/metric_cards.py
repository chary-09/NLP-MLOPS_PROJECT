import streamlit as st


def metric_card(label: str, value, delta=None) -> None:
    st.metric(label, value, delta)

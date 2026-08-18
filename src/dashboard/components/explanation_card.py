import streamlit as st


def explanation_card(features: list[dict]) -> None:
    st.dataframe(features, use_container_width=True)

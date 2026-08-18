import streamlit as st


def render_table(data) -> None:
    st.dataframe(data, use_container_width=True)

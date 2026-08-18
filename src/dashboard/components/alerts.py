import streamlit as st


def alert(message: str) -> None:
    st.warning(message)

import streamlit as st


def loader(message: str = "Loading..."):
    return st.spinner(message)

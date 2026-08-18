import streamlit as st


def prediction_card(result: dict) -> None:
    st.success(f"{result['sentiment']} ({result['confidence']:.1%})")

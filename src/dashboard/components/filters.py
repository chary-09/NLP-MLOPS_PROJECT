import streamlit as st


def sentiment_filter() -> str:
    return st.selectbox("Sentiment", ["All", "positive", "neutral", "negative"])

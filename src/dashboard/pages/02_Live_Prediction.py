import streamlit as st
st.title("Live Prediction")
text = st.text_area("Text to analyze")
if st.button("Analyze") and text:
    st.caption("Connect this page to /api/v1/predict after training the model.")

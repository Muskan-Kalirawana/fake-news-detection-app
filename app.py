import streamlit as st
from model import predict_news

st.title("📰 Fake News Detection App")

news_input = st.text_area("Enter News Article")

if st.button("Predict"):
    label, confidence = predict_news(news_input)
    st.write("Prediction:", "Fake ❌" if label == 0 else "Real ✅")
    st.write("Confidence:", round(confidence * 100, 2), "%")

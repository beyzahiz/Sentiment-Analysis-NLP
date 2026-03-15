import streamlit as st
import requests

st.title("Sentiment Analysis App")

text = st.text_area("Enter a movie review")

if st.button("Predict Sentiment"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": text}
    )

    result = response.json()

    st.write("Sentiment:", result["sentiment"])
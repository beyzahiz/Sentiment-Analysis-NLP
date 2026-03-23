import streamlit as st
import requests

st.title("Sentiment Analysis App")

text = st.text_area("Enter a movie review: ")

if st.button("Analyze"):
    response = requests.post(
        "http://api:8000/predict", 
        json={"text": user_input}
    )

    result = response.json()

    st.write("Sentiment:", result["sentiment"])
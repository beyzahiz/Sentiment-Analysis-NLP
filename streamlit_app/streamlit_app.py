import streamlit as st
import requests
import os

st.title("Sentiment Analysis App")
API_URL = os.getenv("https://sentiment-analysis-nlp-jt8o.onrender.com", "http://localhost:8000")

text = st.text_area("Yorumunuzu girin:")

if st.button("Analiz Et"):
    if text:
        # Değişken ismini 'text' olarak düzelttik
        response = requests.post(f"{API_URL}/predict", json={"text": text})
        
        if response.status_code == 200:
            result = response.json()
            sentiment = result["sentiment"]
            
            # Görselleştirme ekleyelim (CV'de şık durur)
            if sentiment == "positive":
                st.success(f"Sonuç: {sentiment} 😊")
            else:
                st.error(f"Sonuç: {sentiment} 😡")
        else:
            st.error("API hatası oluştu.")
    else:
        st.warning("Lütfen bir metin girin.")
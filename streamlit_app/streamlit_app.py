import streamlit as st
import requests
import os

st.title("Sentiment Analysis App")

# DOĞRU KULLANIM:
# İlk parametre Render'daki 'Key' ismi olmalı. 
# Eğer o isimde bir değişken bulamazsa ikinci parametreyi (localhost) kullanır.
API_URL = os.getenv("API_URL", "https://sentiment-analysis-nlp-jt8o.onrender.com")

text = st.text_area("Yorumunuzu girin:")

if st.button("Analiz Et"):
    if text:
        try:
            # URL'nin sonuna /predict eklediğimizden emin oluyoruz
            target_url = f"{API_URL.rstrip('/')}/predict"
            response = requests.post(target_url, json={"text": text})
            
            if response.status_code == 200:
                result = response.json()
                sentiment = result["sentiment"]
                
                if sentiment == "positive":
                    st.success(f"Sonuç: {sentiment} 😊")
                else:
                    st.error(f"Sonuç: {sentiment} 😡")
            else:
                st.error(f"API Hatası: {response.status_code}")
        except Exception as e:
            st.error(f"Bağlantı Hatası: API henüz hazır olmayabilir veya URL yanlış. (Hata: {e})")
    else:
        st.warning("Lütfen bir metin girin.")
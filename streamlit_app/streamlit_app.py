import streamlit as st
import requests

st.title("Sentiment Analysis App")

API_BASE_URL = "https://beyzahiz-sentiment-analysis-api.hf.space"

text = st.text_area("Yorumunuzu girin:")

if st.button("Analiz Et"):
    if text:
        try:
            # URL'yi tek bir yerden ve temiz bir şekilde birleştiriyoruz
            target_url = f"{API_BASE_URL.strip('/')}/predict"
            
            with st.spinner('Analiz ediliyor, lütfen bekleyin...'):
                response = requests.post(target_url, json={"text": text})
            
            if response.status_code == 200:
                result = response.json()
                sentiment = result["sentiment"]
                
                if sentiment == "positive":
                    st.success(f"Sonuç: {sentiment} 😊")
                else:
                    st.error(f"Sonuç: {sentiment} 😡")
            elif response.status_code == 404:
                st.error("API Hatası: 404 (Adres bulunamadı). Lütfen API URL'sini kontrol edin.")
            else:
                st.error(f"API Hatası: {response.status_code}")
        except Exception as e:
            st.error(f"Bağlantı Hatası: {e}")
    else:
        st.warning("Lütfen bir metin girin.")
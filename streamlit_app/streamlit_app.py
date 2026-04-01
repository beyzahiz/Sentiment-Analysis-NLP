import streamlit as st
import requests
import time

# Sayfa ayarları
st.set_page_config(page_title="Sentify - AI Duygu Analizi", page_icon="🧠")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Sentify: AI Duygu Analizi")
st.subheader("Metinlerin ardındaki duyguyu yapay zeka ile keşfedin.")

API_BASE_URL = "https://beyzahiz-sentiment-analysis-api.hf.space"

text = st.text_area("Analiz edilecek film yorumunuzu buraya yazın:", placeholder="Örn: Bu harika bir filmdi!", height=150)

if st.button("Duyguyu Analiz Et 🚀"):
    if text.strip():
        try:
            target_url = f"{API_BASE_URL.strip('/')}/predict"
            
            with st.spinner('Yapay zeka modeli analiz ediyor...'):
                # Modelin yüklenmesi için zaman tanıyoruz
                response = requests.post(target_url, json={"text": text})
            
            if response.status_code == 200:
                result = response.json()
                sentiment = result["sentiment"]
                
                st.divider()
                if sentiment == "positive":
                    st.balloons() # KONFETİ! 🎉
                    st.success(f"### Sonuç: Pozitif ")
                    st.info("Model bu metinde iyimser ve yapıcı bir ton saptadı.")
                elif sentiment == "neutral":
                    st.warning("### Sonuç: Nötr 😐")
                    st.info("Model bu metinde dengeli veya tarafsız bir ton saptadı.")
                else:
                    st.error("### Sonuç: Negatif 😡")
                    st.warning("Model bu metinde eleştirel bir ton saptadı.")
            else:
                st.error(f"API Hatası: {response.status_code}")
        except Exception as e:
            st.error(f"Bağlantı Hatası: API şu an uykuda olabilir. Lütfen 30 saniye sonra tekrar deneyin. (Hata: {e})")
    else:
        st.warning("⚠️ Lütfen analiz için bir metin girin.")

st.sidebar.markdown("---")
st.sidebar.info("Bu proje BERT mimarisi kullanılarak Dockerize edilmiş ve dağıtık sistem mimarisi (Render + HF) ile canlıya alınmıştır.")
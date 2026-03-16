# 🎬 Movie Sentiment Analysis: From Classical ML to BERT

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-BERT-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)

Bu proje, 50.000 IMDB film yorumu üzerinde duygu analizi (sentiment analysis) gerçekleştirmek amacıyla geliştirilmiştir. Geleneksel Makine Öğrenmesi yöntemlerinden modern Transformer mimarilerine uzanan geniş bir yelpazede model karşılaştırmaları içerir ve eğitilen en iyi modelin **FastAPI** & **Streamlit** ile uçtan uca canlıya alınmasını kapsar.

---

## 📌 Proje Genel Bakış 

Bu proje, film yorumları üzerinde **duygu analizi (sentiment analysis)** yapmak için klasik makine öğrenmesi yöntemlerinden modern transformer mimarilerine kadar uzanan farklı yaklaşımları keşfeder.

Temel amaç, farklı Doğal Dil İşleme (NLP) modellerinin performansını karşılaştırmak ve en iyi sonuç veren modeli **FastAPI** ve **Streamlit** kullanarak gerçek zamanlı bir tahmin servisi olarak yayına almaktır.

**Öne Çıkanlar:**
- Kapsamlı NLP ön işleme hattı (Normalizasyon, Temizleme, Tokenization)
- TF-IDF + Lojistik Regresyon (Baseline model)
- Önceden eğitilmiş GloVe kelime gömme (embeddings) ile Bi-LSTM mimarisi
- Hugging Face Transformers kütüphanesi ile BERT Fine-tuning
- FastAPI ile modelin servis edilmesi (Deployment)
- Streamlit ile etkileşimli web arayüzü

---

## 📚 Veri Seti

Bu projede, duygu analizi için dünya çapında standart kabul edilen **IMDB Movie Reviews Dataset** kullanılmıştır.

**Veri Seti Özellikleri:**
- 50.000 film yorumu.
- İkili (binary) duygu etiketleri (pozitif / negatif).
- Dengeli dağılım: 25.000 eğitim, 25.000 test örneği.

---

## 📊 Veri Analizi ve Görselleştirme (EDA)

Veri seti üzerinde yapılan ilk incelemeler, modelin öğrenme sürecini optimize etmek için kullanılmıştır.

* **Sınıf Dengesi:** 25.000 pozitif ve 25.000 negatif yorum ile tam dengeli bir veri seti üzerinde çalışılmıştır.
* **Kelime Analizi:** Yorumlardaki en sık geçen kelimeler ve metin uzunlukları analiz edilerek modellerin giriş parametreleri (max_length) belirlenmiştir.

| Sınıf Dağılımı | Metin Uzunluğu Dağılımı |
| :---: | :---: |
| ![Sentiment Distribution](sentiment-distribution.png) | ![Review Length Distribution](review-length-distribution.png) |

### Kelime Bulutları (WordClouds)
Metin temizleme öncesi ve sonrası kelime yoğunlukları incelenmiştir.

![Most Frequent Words](most-words.jpg)
![Negative Word Cloud](negative-wc.jpg)

---

## 🛠️ Veri Ön İşleme (Preprocessing)

Modellerin başarısını artırmak için ham metin verileri şu aşamalardan geçirilmiştir:
* **Normalization:** Küçük harfe dönüştürme.
* **Cleaning:** Regex ile HTML etiketlerinin (`<br />`) ve noktalama işaretlerinin temizlenmesi.
* **Tokenization:** NLTK kullanılarak kelimelerin ayrıştırılması.
* **Stop-words:** Anlamsal ağırlığı olmayan kelimelerin (the, is, in vb.) elenmesi.

---

## 🧠 Geliştirilen Modeller ve Teknik Detaylar

### A. Baseline: TF-IDF & Logistic Regression
Geleneksel bir yaklaşım olan Lojistik Regresyon ile **%88.74** doğruluk elde edilmiştir. Modelin kararlarını hangi kelimelere dayanarak verdiği katsayı analizleri ile görselleştirilmiştir.

![Logistic Regression CM](logistic-regression-cm.png)

| En Pozitif Kelimeler | En Negatif Kelimeler |
| :---: | :---: |
| ![Top Positive Words](top-positive-words.png) | ![Top Negative Words](top-negative-words.png) |

### B. Deep Learning: Bi-LSTM & GloVe
Metinlerin ardışık yapısını kavramak için çift yönlü LSTM mimarisi kullanılmıştır. **Stanford GloVe** önceden eğitilmiş kelime vektörleri ile transfer learning uygulanmıştır.

### C. State-of-the-Art: BERT Fine-Tuning
Hugging Face `Transformers` kütüphanesi kullanılarak `bert-base-uncased` modeli bu veri setine özel olarak eğitilmiştir.
* **Optimizer:** AdamW
* **Learning Rate:** 2e-5
* **Başarı:** **%92+ Accuracy** ile en iyi performansı göstermiştir.

![BERT Confusion Matrix](bert_confusion_matrix.png)

---

## 📈 Model Performans Karşılaştırması

Proje kapsamında eğitilen tüm modellerin başarı oranları aşağıda karşılaştırılmıştır. Modern Transformer mimarilerinin (BERT) klasik ve LSTM tabanlı yöntemlere üstünlüğü net bir şekilde gözlenmektedir.

### Model Başarı Karşılaştırması

| Model | Doğruluk (Accuracy) |
| :--- | :---: |
| Lojistik Regresyon (TF-IDF) | 0.887 |
| Bi-LSTM (GloVe) | 0.881 |
| **BERT (Fine-tuned)** | **0.924** |

![Model Performance Comparison](model_comparison.png)

---

## 🛠  Kullanılan Teknolojiler

| Kategori | Araçlar |
| :--- | :--- |
| **Dil ve Temel Kütüphaneler** | Python, Pandas, NumPy |
| **NLP** | NLTK, spaCy |
| **Makine Öğrenmesi** | Scikit-learn |
| **Derin Öğrenme** | TensorFlow / Keras, PyTorch |
| **Transformers** | Hugging Face Transformers (`bert-base-uncased`) |
| **Deployment** | FastAPI, Streamlit |

---

## 🚀 Deployment

Eğitilen BERT modeli, gerçek zamanlı tahminler yapabilmek için iki aşamalı bir mimari ile ayağa kaldırılmıştır. Model, Streamlit arayüzü üzerinden yazılan film yorumlarını gerçek zamanlı olarak analiz edebilmektedir.

1.  **Backend (FastAPI):** Model, `/predict` endpoint'i üzerinden JSON tabanlı tahminler sunan yüksek performanslı bir API haline getirilmiştir.
2.  **Frontend (Streamlit):** Kullanıcının yorum yazıp anlık duygu durumunu görebildiği sade ve şık bir arayüz geliştirilmiştir.

![Sentiment Analysis App UI](Ekran Resmi 2026-03-15 17.18.00.png)


---

## ⚙️ Kurulum ve Çalıştırma

Projeyi yerelinizde çalıştırmak için:

1. Depoyu klonlayın:
```bash
git clone [https://github.com/beyzahiz/sentiment-analysis-nlp.git](https://github.com/beyzahiz/sentiment-analysis-nlp.git)
cd sentiment-analysis-nlp
```
2. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```
3. FastAPI sunucusunu başlatın:
```bash
cd api
uvicorn app:app --reload
```
4. Streamlit uygulamasını çalıştırın:
```bash
streamlit run streamlit_app.py
```
5. Tarayıcıda açın:
```bash
http://localhost:8501
```


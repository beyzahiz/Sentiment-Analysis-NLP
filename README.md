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
* **Kelime Analizi:** Yorumlardaki en sık geçen kelimeler ve metin uzunlukları analiz edilerek modellerin giriş parametreleri belirlenmiştir.

| Sınıf Dağılımı | Metin Uzunluğu Dağılımı |
| :---: | :---: |
| ![Sentiment Distribution](assets/sentiment-distribution.png) | ![Review Length Distribution](assets/review-length-distribution.png) |

<br>

### Kelime Bulutları (WordClouds)
En sık geçen kelimeler ile pozitif ve negatif kelime bulutları incelenmiştir.

<br>
<p align="center">
  <img src="assets/most-words.png" width="500">
</p>

<br>

| Positive Word Cloud | Negative Word Cloud |
| :---: | :---: |
| ![Positive Word Cloud](assets/positive-wc.png) | !![Negative Word Cloud](assets/negative-wc.png) |

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

<br>
<p align="center">
  <img src="assets/logistic-regression-cm.png" width="500">
</p>

<br>

| En Pozitif Kelimeler | En Negatif Kelimeler |
| :---: | :---: |
| ![Top Positive Words](assets/top-positive-words.png) | ![Top Negative Words](assets/top-negative-words.png) |

### B. Deep Learning: Bi-LSTM & GloVe
Metinlerin ardışık yapısını kavramak için çift yönlü LSTM mimarisi kullanılmıştır. **Stanford GloVe** önceden eğitilmiş kelime vektörleri ile transfer learning uygulanmıştır.

### C. State-of-the-Art: BERT Fine-Tuning
Hugging Face `Transformers` kütüphanesi kullanılarak `bert-base-uncased` modeli bu veri setine özel olarak eğitilmiştir.
* **Optimizer:** AdamW
* **Learning Rate:** 2e-5
* **Başarı:** **%92+ Accuracy** ile en iyi performansı göstermiştir.

<br>
<p align="center">
  <img src="assets/bert_confusion_matrix.png" width="600">
</p>

<br>

---

## 📈 Model Performans Karşılaştırması

Proje kapsamında eğitilen tüm modellerin başarı oranları aşağıda karşılaştırılmıştır. Modern Transformer mimarilerinin (BERT) klasik ve LSTM tabanlı yöntemlere üstünlüğü net bir şekilde gözlenmektedir.

### Model Başarı Karşılaştırması

| Model | Doğruluk (Accuracy) |
| :--- | :---: |
| Lojistik Regresyon + TF-IDF | 0.887 |
| Bi-LSTM (GloVe) | 0.881 |
| **BERT (Fine-tuned)** | **0.924** |


<br>
<p align="center">
  <img src="assets/model_comparison.png" width="600">
</p>

<br>

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


| Pozitif Deneme | Negatif Deneme |
| :---: | :---: |
| ![Sentiment Analysis Positive](assets/sentiment-analysis-positive.png) | ![Sentiment Analysis Negative](assets/sentiment-analysis-negative.png) |

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


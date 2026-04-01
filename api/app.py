import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

tokenizer = None
model = None
# Bu model 100+ dilde (Türkçe dahil) 1-5 yıldız arası duygu analizi yapar
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment" 

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print(f"🔄 Model Yükleniyor: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        print("✨ BAŞARILI: Multilingual BERT hazır!")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    # model_loaded durumunu kontrol ederken model nesnesinin varlığına bakıyoruz
    return {"status": "API is Running on HF Spaces", "model_loaded": model is not None}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    load_model()
    
    # Metni işle
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Model 5 sınıf döndürür (0, 1, 2, 3, 4) -> (1, 2, 3, 4, 5 Yıldız)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Karar Mantığı (Mapping)
    if prediction <= 1: # 0 ve 1 (1 ve 2 yıldız) -> Negatif
        sentiment = "negative"
    elif prediction == 2: # 2 (3 yıldız) -> Nötr
        sentiment = "neutral"
    else: # 3 ve 4 (4 ve 5 yıldız) -> Pozitif
        sentiment = "positive"
        
    # Streamlit'in beklediği JSON formatı
    return {"sentiment": sentiment}
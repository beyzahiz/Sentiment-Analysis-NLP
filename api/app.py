import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

tokenizer = None
model = None
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print(f"🔄 Model yükleniyor: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        print("✨ BAŞARILI: Model HF Spaces üzerinde hazır!")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "API is Running on HF Spaces", "model_loaded": model is not None}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    load_model()
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    # 0 -> 1 Yıldız (Çok Negatif)
    # 1 -> 2 Yıldız (Negatif)
    # 2 -> 3 Yıldız (Nötr)
    # 3 -> 4 Yıldız (Pozitif)
    # 4 -> 5 Yıldız (Çok Pozitif)
    
    if prediction <= 1: 
        sentiment = "negative"
    elif prediction == 2:
        sentiment = "neutral"
    else:
        sentiment = "positive"
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Global değişkenler
tokenizer = None
model = None

# BURASI KRİTİK: Kullandığın modelin Hugging Face adını buraya yaz
# Eğer Türkçe model kullanıyorsan: "dbmdz/bert-base-turkish-cased"
# Eğer İngilizce model kullanıyorsan: "bert-base-uncased"
MODEL_NAME = "bert-base-uncased" 

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            print(f"🔄 Model Hugging Face üzerinden indiriliyor: {MODEL_NAME}...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            print("✨ BAŞARILI: Model buluttan yüklendi ve hazır!")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise e

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "API is alive", "model_loaded": model is not None}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    load_model()
    
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs).item()
    sentiment = "positive" if prediction == 1 else "negative"
    
    return {"sentiment": sentiment}
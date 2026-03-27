import os
from pathlib import Path
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        # 1. Adım: app.py'nin olduğu tam klasörü bul
        base_path = Path(__file__).parent.resolve()
        # 2. Adım: Model klasörünün tam yolunu Path objesi olarak oluştur
        model_path = base_path / "sentiment_model"
        
        # Logda tam yolu görelim ki emin olalım
        print(f"🔄 Model yükleniyor (Tam Yol): {model_path}")
        
        # Klasör var mı kontrol edelim (Loglarda görürüz)
        if not model_path.exists():
            print(f"❌ HATA: {model_path} dizini bulunamadı!")
            return

        try:
            # Path objesini doğrudan veriyoruz, Transformers bunu lokal dizin olarak tanır
            tokenizer = AutoTokenizer.from_pretrained(str(model_path.absolute()), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path.absolute()), local_files_only=True)
            print("✅ Model başarıyla belleğe alındı!")
        except Exception as e:
            print(f"❌ KRİTİK HATA: Yükleme sırasında sorun çıktı: {e}")
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
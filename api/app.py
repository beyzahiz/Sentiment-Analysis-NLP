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
        # app.py neredeyse orayı baz al
        current_file_path = Path(__file__).resolve()
        base_path = current_file_path.parent
        
        # Önce aynı klasörde ara, bulamazsan bir üst klasörde ara
        model_path = base_path / "sentiment_model"
        
        if not model_path.exists():
            model_path = base_path.parent / "sentiment_model"

        print(f"🔄 Model aranıyor: {model_path}")

        if not model_path.exists():
            print(f"❌ KRİTİK HATA: sentiment_model klasörü hiçbir yerde bulunamadı!")
            # Dosya yapısını loglara basalım ki sorunu kökten görelim
            print(f"Mevcut dizin içeriği: {os.listdir(base_path)}")
            return

        try:
            print(f"✅ Model bulundu, yükleniyor: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
            print("✨ TEBRİKLER: Model ve Tokenizer başarıyla yüklendi!")
        except Exception as e:
            print(f"❌ Yükleme hatası: {e}")
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
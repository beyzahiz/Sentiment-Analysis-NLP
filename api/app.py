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
        # 1. Strateji: app.py dosyasının olduğu gerçek konumu bul
        current_dir = Path(__file__).parent.resolve()
        
        # 2. Strateji: Aday yolları belirle
        candidate_paths = [
            current_dir / "sentiment_model",           # Yanında mı?
            current_dir / "api" / "sentiment_model",   # api klasörü içinde mi?
            Path("/app/api/sentiment_model"),          # Docker kök dizininde mi?
            Path("/app/sentiment_model")               # Direkt /app altında mı?
        ]
        
        model_path = None
        for path in candidate_paths:
            print(f"🔍 Kontrol ediliyor: {path}")
            if path.exists() and path.is_dir():
                model_path = path
                break

        if model_path is None:
            print(f"❌ KRİTİK HATA: sentiment_model klasörü bulunamadı!")
            # Sorunu teşhis etmek için etrafa bakalım
            print(f"📍 Şu anki konum: {current_dir}")
            try:
                print(f"📂 Klasör içeriği: {os.listdir(current_dir)}")
                if current_dir.parent.exists():
                    print(f"📂 Üst klasör içeriği: {os.listdir(current_dir.parent)}")
            except:
                pass
            return

        try:
            print(f"✅ Model bulundu! Yükleniyor: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
            print("✨ BAŞARILI: Model ve Tokenizer yüklendi.")
        except Exception as e:
            print(f"❌ Yükleme sırasında teknik hata: {e}")
            raise e

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "API is alive", "model_loaded": model is not None}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    load_model()
    
    if tokenizer is None:
        return {"error": "Model could not be loaded. Check server logs."}

    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs).item()
    sentiment = "positive" if prediction == 1 else "negative"
    
    return {"sentiment": sentiment}
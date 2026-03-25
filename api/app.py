import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Global değişkenler
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        # Docker konteyner yapısına göre lokal yolu en sade haliyle veriyoruz
        # transformers kütüphanesi './klasor_adi' formatını gördüğünde lokal olduğunu anlar.
        model_path = "./sentiment_model"
        
        print(f"🔄 Model yükleniyor: {model_path}...")
        
        try:
            # local_files_only=True ve trust_remote_code=False güvenliği artırır
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            print("✅ Model başarıyla belleğe alındı!")
        except Exception as e:
            print(f"❌ KRİTİK HATA: Model yüklenirken klasör bulunamadı: {e}")
            raise e

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "API is alive", "model_loaded": model is not None}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    load_model() # İlk istekte yüklemeyi tetikler
    
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs).item()
    sentiment = "positive" if prediction == 1 else "negative"
    
    return {"sentiment": sentiment}
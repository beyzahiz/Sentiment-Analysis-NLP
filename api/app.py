import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Global değişkenler (başlangıçta boş)
tokenizer = None
model = None

def load_model():
    """Modeli sadece ihtiyaç duyulduğunda yükleyen fonksiyon (Lazy Loading)"""
    global tokenizer, model
    if tokenizer is None or model is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "sentiment_model")
        print(f"🔄 Model yükleniyor: {model_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        print("✅ Model başarıyla belleğe alındı!")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    # Sunucu anında cevap verecek, Render "Live" diyecek
    return {"status": "API is alive", "model_loaded": model is not None}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    # Tahmin isteği geldiğinde model yüklü değilse yükle
    load_model()
    
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
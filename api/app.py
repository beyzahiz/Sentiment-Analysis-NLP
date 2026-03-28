import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

tokenizer = None
model = None

# DistilBERT zaten hafiftir ama biz onu daha da hafifleteceğiz 🚀
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" 

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            print(f"🔄 Ultra-Hafif modda model yükleniyor: {MODEL_NAME}...")
            
            # Bellek kullanımını minimize eden parametreler ekliyoruz
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                low_cpu_mem_usage=True, # RAM'i sömürme 🛡️
                torch_dtype=torch.float32 # En stabil veri tipi
            )
            print("✨: Model kısıtlı kaynaklarla başarıyla yüklendi!")
        except Exception as e:
            print(f"❌ Kritik Hata: {e}")
            raise e

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "API is online - LIGHT MODE", "model_loaded": model is not None}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    load_model()
    
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128) # Max length'i kısalttık
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs).item()
    sentiment = "positive" if prediction == 1 else "negative"
    
    return {"sentiment": sentiment}
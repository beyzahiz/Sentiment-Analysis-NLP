import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

tokenizer = None
model = None

# Standart BERT yerine DistilBERT kullanarak RAM sorununu çözüyoruz 🚀
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" 

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            print(f"🔄 Optimize model (DistilBERT) yükleniyor: {MODEL_NAME}...")
            # DistilBERT çok daha hafif olduğu için Render'da takılmaz
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            print("✨ ZAFER: Optimize model başarıyla yüklendi!")
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
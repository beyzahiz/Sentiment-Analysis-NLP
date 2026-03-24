import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Docker içindeki tam yolu (absolute path) kullanmak her zaman daha güvenlidir
# app.py api klasörünün içindeyse, sentiment_model de aynı yerdedir.
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "sentiment_model")

# Modelin yüklenip yüklenmediğini kontrol etmek için bir log
print(f"Model yükleniyor: {model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {e}")

class TextRequest(BaseModel):
    text: str

@app.get("/") # Render'ın sağlıklı olduğunu anlaması için bir ana sayfa
def read_root():
    return {"status": "API is running"}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad(): # Tahmin yaparken hafızayı yormayalım
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs).item()
    sentiment = "positive" if prediction == 1 else "negative"
    
    return {"sentiment": sentiment}
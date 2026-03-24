import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# --- MODEL YÜKLEME BÖLÜMÜ ---
# Mevcut dosyanın (app.py) bulunduğu tam dizini alıyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))
# Model klasörünün tam yolunu oluşturuyoruz
model_path = os.path.join(current_dir, "sentiment_model")

print(f"Model aranıyor: {model_path}")

# Tokenizer ve Modeli yüklüyoruz
# local_files_only=True parametresi, internete bakmamasını, sadece klasöre bakmasını sağlar
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
# ----------------------------

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "API is alive!"}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
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
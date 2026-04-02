import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googletrans import Translator # Çeviri için 

app = FastAPI()
translator = Translator()

tokenizer = None
model = None

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" 

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    load_model()
    
    # Eğer metin Türkçe ise İngilizceye çevir 
    try:
        translated = translator.translate(request.text, dest='en').text
    except:
        translated = request.text # Çeviri patlarsa orijinaliyle devam et
        
    # İngilizce model ile analiz et
    inputs = tokenizer(translated, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs).item()
    confidence = torch.max(probs).item()
    
    if confidence < 0.90: 
        sentiment = "neutral"
    else:
        sentiment = "positive" if prediction == 1 else "negative"
        
    return {"sentiment": sentiment}
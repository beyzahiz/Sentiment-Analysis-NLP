import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googletrans import Translator

app = FastAPI()
translator = Translator()

tokenizer = None
model = None

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

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
    
    # Çeviri Katmanı
    try:
        translated = translator.translate(request.text, dest='en').text
    except:
        translated = request.text
        
    # RoBERTa ile Analiz
    inputs = tokenizer(translated, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Model Çıktıları: 0 -> Negative, 1 -> Neutral, 2 -> Positive
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = mapping[prediction]
        
    return {"sentiment": sentiment}
import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Render/Docker için en sağlam yol belirleme yöntemi
current_dir = os.path.dirname(os.path.abspath(__file__))
# Klasör yolunun sonuna '/' eklemek, kütüphaneye bunun bir repo değil dizin olduğunu anlatır.
model_path = os.path.join(current_dir, "sentiment_model") + os.sep

print(f"Model aranıyor: {model_path}")

try:
    # 'local_files_only=True' internete gitmesini engeller.
    # 'model_path' artık tam yol (absolute path) olarak besleniyor.
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    print("✅ BAŞARILI: Model ve Tokenizer yüklendi.")
except Exception as e:
    print(f"❌ HATA: Model yüklenemedi. Detay: {e}")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "API is alive!"}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    # Model yüklenemediyse hata dönmesi için kontrol (Safe-guard)
    if 'tokenizer' not in globals() or 'model' not in globals():
        return {"error": "Model is not loaded on the server."}

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
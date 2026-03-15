from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

model_path = "/Users/beyzahiz/Desktop/Sentiment-Analysis-NLP/sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
def predict_sentiment(request: TextRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probs).item()

    sentiment = "positive" if prediction == 1 else "negative"

    return {
        "sentiment": sentiment
    }
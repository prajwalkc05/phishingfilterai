from fastapi import FastAPI
from app.models.model_loader import tokenizer, model
import torch

app = FastAPI()

@app.post("/predict")
def predict(message: str):
    inputs = tokenizer(message, return_tensors="pt", truncation=True)
    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    label = probs.argmax().item()
    confidence = probs.max().item()

    return {"label": label, "confidence": confidence}
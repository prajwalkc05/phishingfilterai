import requests
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()

model_name = "prajwalkc/phishing-bert"
tokenizer = None
model = None

LABELS = {
    0: "safe",
    1: "spam",
    2: "phishing"
}

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=os.getenv('HF_TOKEN'))
        model.eval()

def predict_sms(text: str):
    try:
        load_model()
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label_id = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][label_id].item()
        
        return {
            "label": LABELS.get(label_id, "unknown"),
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e), "label": "unknown", "confidence": 0.0}
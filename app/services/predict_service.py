from app.models import model_loader

LABELS = {
    0: "safe",
    1: "spam",
    2: "phishing"
}

def predict_sms(message: str):
    model_loader.load_model()
    
    inputs = model_loader.tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = model_loader.model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    confidence, pred = probs.max(dim=1)

    return {
        "label": LABELS[pred.item()],
        "confidence": round(confidence.item(), 2)
    }
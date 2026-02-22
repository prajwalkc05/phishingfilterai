from app.models.model_loader import model, tokenizer

LABELS = {
    0: "safe",
    1: "spam",
    2: "phishing"
}

def predict_sms(message: str):
    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    confidence, pred = probs.max(dim=1)

    return {
        "label": LABELS[pred.item()],
        "confidence": round(confidence.item(), 2)
    }
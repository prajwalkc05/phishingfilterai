from app.models.model_loader import model, vectorizer

LABELS = {
    0: "safe",
    1: "spam",
    2: "phishing"
}

def predict_sms(message: str):
    vec = vectorizer.transform([message])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max()

    return {
        "label": LABELS[pred],
        "confidence": round(float(prob), 2)
    }
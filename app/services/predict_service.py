import random

# Simple phishing keywords
PHISHING_KEYWORDS = [
    "bank", "account", "otp", "verify", "click",
    "urgent", "blocked", "password", "loan", "link"
]


def predict_sms(message: str):
    message_lower = message.lower()

    score = 0
    for word in PHISHING_KEYWORDS:
        if word in message_lower:
            score += 1

    # confidence simulation
    confidence = min(score * 0.2, 0.95)

    if score > 2:
        label = "phishing"
    elif score > 0:
        label = "spam"
    else:
        label = "safe"

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }

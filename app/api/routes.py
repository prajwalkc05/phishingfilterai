from fastapi import APIRouter, Body
from app.schemas.request import SMSRequest, PredictionResponse
from app.services.predict_service import predict_sms_wrapper
from app.core.database import feedback_collection
from datetime import datetime, timezone

router = APIRouter()

@router.get("/")
@router.head("/")
def root():
    return {
        "message": "Phishing SMS Detection API",
        "version": "1.0",
        "endpoints": {
            "predict": "POST /predict",
            "feedback": "POST /feedback",
            "check": "POST /check"
        }
    }

@router.post("/predict", response_model=PredictionResponse)
def predict(request: SMSRequest):
    result = predict_sms_wrapper(request.message)
    return result

@router.post("/check")
def check_user_label(data: dict = Body(...)):
    """Check if user has already verified this message or sender."""
    if feedback_collection is None:
        return {"user_label": None}

    message = data.get("message")
    sender = data.get("sender")

    # 1. Exact message match with user-verified label
    record = feedback_collection.find_one(
        {"message": message, "user_label": {"$ne": None}},
        sort=[("timestamp", -1)]
    )

    # 2. Fallback: same sender has a verified label
    if not record:
        record = feedback_collection.find_one(
            {"sender": sender, "user_label": {"$ne": None}},
            sort=[("timestamp", -1)]
        )

    if record and record.get("user_label"):
        return {"user_label": record["user_label"]}

    return {"user_label": None}

@router.post("/feedback")
def store_feedback(data: dict = Body(...)):
    if feedback_collection is None:
        return {"status": "error", "message": "Database not configured"}

    message = data.get("message")
    sender = data.get("sender")
    predicted_label = data.get("predicted_label") or data.get("predicted")
    user_label = data.get("user_label") or data.get("correct")

    feedback_collection.update_one(
        {"message": message},
        {"$set": {
            "message": message,
            "sender": sender,
            "predicted_label": predicted_label,
            "user_label": user_label,
            "timestamp": datetime.now(timezone.utc)
        }},
        upsert=True
    )

    return {"status": "saved"}

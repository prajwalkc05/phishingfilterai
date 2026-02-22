from fastapi import APIRouter, Body
from app.schemas.request import SMSRequest, PredictionResponse
from app.services.predict_service import predict_sms
from app.core.database import feedback_collection
from datetime import datetime, timezone

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(request: SMSRequest):
    result = predict_sms(request.message)
    return result

@router.post("/feedback")
def store_feedback(data: dict = Body(...)):
    feedback_collection.insert_one({
        "message": data["message"],
        "predicted_label": data["predicted"],
        "user_label": data["correct"],
        "timestamp": datetime.now(timezone.utc)
    })

    return {"status": "saved"}

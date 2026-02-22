from fastapi import APIRouter, Body
from app.schemas.request import SMSRequest, PredictionResponse
from app.services.predict_service import predict_sms_wrapper
from app.core.database import feedback_collection
from datetime import datetime, timezone

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(request: SMSRequest):
    result = predict_sms_wrapper(request.message)
    return result

@router.post("/feedback")
def store_feedback(data: dict = Body(...)):
    if feedback_collection is None:
        return {"status": "error", "message": "Database not configured"}
    
    feedback_collection.insert_one({
        "message": data.get("message"),
        "predicted_label": data.get("predicted") or data.get("predicted_label"),
        "user_label": data.get("correct") or data.get("user_label"),
        "timestamp": datetime.now(timezone.utc)
    })

    return {"status": "saved"}

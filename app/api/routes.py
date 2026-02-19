from fastapi import APIRouter
from app.schemas.request import SMSRequest, PredictionResponse
from app.services.predict_service import predict_sms

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(request: SMSRequest):
    result = predict_sms(request.message)
    return result

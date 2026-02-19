from pydantic import BaseModel

class SMSRequest(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float

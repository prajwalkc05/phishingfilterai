from app.models.model_loader import predict_sms

def predict_sms_wrapper(message: str):
    return predict_sms(message)
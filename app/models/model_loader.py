from gradio_client import Client
import os
from dotenv import load_dotenv

load_dotenv()

SPACE_URL = "prajwalkc/phishing-bert-api"
client = None

def get_client():
    global client
    if client is None:
        client = Client(SPACE_URL)
    return client

def predict_sms(text: str):
    try:
        result = get_client().predict(text, api_name="/predict")
        return {"label": result[0], "confidence": result[1]}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e), "label": "unknown", "confidence": 0.0}
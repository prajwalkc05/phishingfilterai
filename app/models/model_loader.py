import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()

SPACE_URL = "https://prajwalkc-phishing-bert-api.hf.space"

LABELS = {0: "safe", 1: "spam", 2: "phishing"}

def predict_sms(text: str):
    try:
        # Use Gradio's queue system
        join_response = requests.post(f"{SPACE_URL}/queue/join", json={
            "data": [text],
            "fn_index": 0
        })
        
        if join_response.status_code != 200:
            return {"error": "Model unavailable", "label": "unknown", "confidence": 0.0}
        
        event_id = join_response.json().get("event_id")
        
        # Poll for result
        for _ in range(30):
            status_response = requests.get(f"{SPACE_URL}/queue/status?event_id={event_id}")
            data = status_response.json()
            
            if data.get("status") == "COMPLETE":
                result = data.get("data")
                if result and len(result) >= 2:
                    return {"label": result[0], "confidence": result[1]}
            elif data.get("status") == "FAILED":
                break
            
            time.sleep(0.5)
        
        return {"error": "Timeout", "label": "unknown", "confidence": 0.0}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e), "label": "unknown", "confidence": 0.0}
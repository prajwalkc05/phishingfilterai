import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://router.huggingface.co/models/prajwalkc/phishing-bert"

headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
    "Content-Type": "application/json"
}

LABELS = {
    0: "safe",
    1: "spam",
    2: "phishing"
}

def predict_sms(text: str):
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return {"error": "Model unavailable", "label": "unknown", "confidence": 0.0}

    result = response.json()
    print(f"API Response: {result}")
    
    # Handle [[{label, score}]] format
    if isinstance(result, list) and len(result) > 0:
        predictions = result[0] if isinstance(result[0], list) else result
        
        if isinstance(predictions, list) and len(predictions) > 0:
            max_pred = max(predictions, key=lambda x: x['score'])
            
            # Extract label number from 'LABEL_0', 'LABEL_1', etc.
            label_str = max_pred['label']
            if 'LABEL_' in label_str:
                label_id = int(label_str.split('_')[-1])
            else:
                label_id = int(label_str)
            
            return {
                "label": LABELS.get(label_id, "unknown"),
                "confidence": round(max_pred['score'], 2)
            }
    
    print(f"Unexpected format: {result}")
    return {"error": "Invalid response", "label": "unknown", "confidence": 0.0}
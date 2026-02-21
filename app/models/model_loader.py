import joblib

# Load trained model and vectorizer
model = joblib.load("app/models/baseline_model.pkl")
vectorizer = joblib.load("app/models/vectorizer.pkl")
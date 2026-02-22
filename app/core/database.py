from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")

db = client["phishing_ai"]

feedback_collection = db["feedback"]

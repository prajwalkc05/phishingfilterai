from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGO_URI"))
db = client["phishing_ai"]
feedback_collection = db["feedback"]
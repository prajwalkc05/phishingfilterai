import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
train = pd.read_csv("data/train.csv").dropna(subset=["cleaned"])
test = pd.read_csv("data/test.csv").dropna(subset=["cleaned"])

# Features and labels
X_train = train["cleaned"]
y_train = train["label"]

X_test = test["cleaned"]
y_test = test["label"]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()

model.fit(X_train_vec, y_train)

# Prediction
preds = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "app/models/baseline_model.pkl")
joblib.dump(vectorizer, "app/models/vectorizer.pkl")

print("Model saved successfully")
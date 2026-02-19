import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv("data/SMSSpamCollection.csv", sep="\t", header=None, names=["label", "message"], encoding="latin-1")

# Convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "url", text)
    text = re.sub(r"\d+", "number", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()
    words = [w for w in words if w not in STOPWORDS]

    return " ".join(words)

df["cleaned"] = df["message"].apply(clean_text)

# Balance dataset
spam = df[df["label"] == 1]
ham = df[df["label"] == 0].sample(len(spam))

balanced_df = pd.concat([spam, ham])

# Train-test split
train, test = train_test_split(
    balanced_df,
    test_size=0.2,
    random_state=42
)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Preprocessing completed")
print("Train size:", len(train))
print("Test size:", len(test))

import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

# Load original dataset
df1 = pd.read_csv("data/SMSSpamCollection.csv", sep="\t", header=None, names=["label", "message"], encoding="latin-1")
df1["label"] = df1["label"].map({"ham": 0, "spam": 1})

# Load modern phishing
df2 = pd.read_csv("data/modern_phishing_large.csv")
df2["label"] = 2

# Load generated phishing
df3 = pd.read_csv("data/generated_phishing.csv")
df3["label"] = 2

# Load spam promotional
df4 = pd.read_csv("data/spam_promotional.csv")
df4["label"] = 1

# Load spam large
df5 = pd.read_csv("data/spam_large.csv")
df5["label"] = 1

# Load generated spam
df6 = pd.read_csv("data/generated_spam.csv")
df6["label"] = 1

# Load modern phishing
df7 = pd.read_csv("data/modern_phishing.csv")
df7["label"] = 2

# Load promotional spam
df8 = pd.read_csv("data/promotional_spam.csv")

# Combine datasets
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])

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

# Balance
safe = df[df["label"] == 0].sample(2000, replace=True)
spam = df[df["label"] == 1].sample(2000, replace=True)
phishing = df[df["label"] == 2].sample(2000, replace=True)

balanced_df = pd.concat([safe, spam, phishing])

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
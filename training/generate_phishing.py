import pandas as pd
import random

banks = ["SBI", "HDFC", "ICICI", "Axis", "Kotak"]
actions = [
    "account blocked", "OTP expired", "KYC update",
    "PAN verification", "unauthorized login",
    "card suspended"
]

messages = []

for _ in range(300):
    bank = random.choice(banks)
    action = random.choice(actions)

    msg = f"{bank} alert: Your {action}. Click link to verify immediately."
    messages.append([1, msg])

df = pd.DataFrame(messages, columns=["label", "message"])
df.to_csv("data/generated_phishing.csv", index=False)

print("Generated phishing dataset")
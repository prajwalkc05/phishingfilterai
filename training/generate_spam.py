import pandas as pd
import random

offers = [
    "50% discount", "exclusive deal", "limited offer",
    "cashback", "festival sale", "free voucher",
    "buy 1 get 1", "special price"
]

products = [
    "shopping", "electronics", "clothing",
    "mobile", "groceries", "fashion"
]

messages = []

for _ in range(300):
    offer = random.choice(offers)
    product = random.choice(products)

    msg = f"Get {offer} on {product}. Shop now."
    messages.append([1, msg])

df = pd.DataFrame(messages, columns=["label", "message"])
df.to_csv("data/generated_spam.csv", index=False)

print("Generated spam dataset")
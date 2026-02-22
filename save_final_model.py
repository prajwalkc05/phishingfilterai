from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint_path = "bert_model/checkpoint-1800"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")

print("Final model saved successfully")
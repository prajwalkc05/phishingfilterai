import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "prajwalkc/phishing-bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()
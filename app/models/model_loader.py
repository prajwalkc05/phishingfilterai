import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "prajwalkc/phishing-bert"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        model.eval()
        torch.set_num_threads(1)
    return tokenizer, model
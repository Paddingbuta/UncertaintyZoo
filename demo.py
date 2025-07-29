# demo.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from uncertainty import Quantifier

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

uq = Quantifier(model, tokenizer, methods=["mc_dropout_var", "predictive_entropy"])
code_str = "def add(x, y): return x + y"

scores = uq.quantify(code_str)
print(scores)

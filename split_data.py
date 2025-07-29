import json
import random

with open("data/function.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)
n = len(data)
train_data = data[:int(0.8*n)]
val_data = data[int(0.8*n):int(0.9*n)]
test_data = data[int(0.9*n):]

with open("data_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
with open("data_val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)
with open("data_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm

# 1. 加载模型和tokenizer
model_path = "./model/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 创建自定义数据集
class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        encoding = self.tokenizer(
            sample["func"], 
            truncation=True, 
            max_length=self.max_length, 
            padding='max_length', 
            return_tensors="pt"
        )
        label = torch.tensor(sample["target"], dtype=torch.long)
        return {**encoding, 'label': label}

# 3. 加载数据集
with open("data/data_train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("data/data_val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

train_dataset = CodeDataset(train_data, tokenizer)
val_dataset = CodeDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 4. 加载模型并添加分类头
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
model.train()

# 5. 设置优化器
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 6. 定义训练函数
def train_epoch(model, data_loader, optimizer):
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()

        inputs = {key: value.squeeze(1).to(device) for key, value in batch.items() if key != 'label'}
        labels = batch['label'].to(device)

        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits

        # 计算损失
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算准确率
        preds = torch.argmax(logits, dim=-1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# 7. 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

# 8. 保存微调后的模型
model.save_pretrained("./model_finetuned")
tokenizer.save_pretrained("./model_finetuned")


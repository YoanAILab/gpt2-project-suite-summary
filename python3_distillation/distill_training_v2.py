# distill_training_v2.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader, Dataset
import os

'''
| 方面 | v1 | v2 |
|------|----|----|
| student 构建方式 | 直接加载 `gpt2` | 用 `GPT2Config` 构建小模型 |
| transformer 层数 | 12层（默认） | 6层 |
| 模型大小 | 与 teacher 一样 | 比 teacher 小一半 |
| 应用场景 | 简单蒸馏演示 | 模型压缩 + 蒸馏 |
'''

# 1. 加载 teacher 模型（还是你之前finetune过的完整gpt2）
teacher_model = GPT2LMHeadModel.from_pretrained("../python1_basic_training/gpt2_finetune").eval()

# 2. 定义小一点的 student 模型（比如6层Transformer）
small_config = GPT2Config(
    n_embd=768,        # 保持embedding维度一致，确保可以装载参数
    n_layer=6,         # transformer block从12层减少到6层！！
    n_head=12,         # 多头注意力头数量保持一致，确保可以装载参数
    n_positions=1024,  # 位置编码
    n_ctx=1024,
    vocab_size=50257   # 跟teacher一样大（词表大小不能改，否则 tokenizer 会报错）
)

student_model = GPT2LMHeadModel(small_config).train()

# 3. 加载 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("../python1_basic_training/gpt2_finetune")

# 4. 移动到GPU（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

# 5. 准备训练数据（小量demo版）
train_texts = [
    "Hello world!",
    "How are you today?",
    "The sky is blue.",
    "I love machine learning.",
    "AI is changing the world."
]

# 自定义Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}

# 用上面定义的 Dataset 创建 DataLoader，每批 2 个样本，打乱顺序。
dataset = TextDataset(train_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 6. 配置 optimizer，使用 AdamW 优化器更新 student 模型参数
optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

# 7. 定义 loss 函数（KL散度）
# 采用 KL 散度作为 distillation 的核心 loss， batchmean 模式会对每个 batch 中的所有 token 的 KL loss 取平均。
loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

# 8. 训练主循环
num_epochs = 5

for epoch in range(num_epochs):
    print(f"=== Epoch {epoch+1}/{num_epochs} ===")
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Teacher输出（不需要梯度）
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # Student输出
        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # 计算 KL 散度 Loss
        # 先对 student 输出取 log_softmax（符合 KL 要求）
        # teacher 输出取 softmax 得到概率分布
        loss = loss_fn(
            student_logits.log_softmax(dim=-1),
            teacher_logits.softmax(dim=-1)
        )

        # 反向传播与更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加总损失
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished, avg_loss={avg_loss:.4f}")

# 9. 保存小 student 模型
save_path = "./gpt2_student_v2"
os.makedirs(save_path, exist_ok=True)
student_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ 小模型 Student 成功保存到 {save_path}")

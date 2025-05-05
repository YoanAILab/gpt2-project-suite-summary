# distill_training.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset

# 1. 加载 teacher 和 student
teacher_model = GPT2LMHeadModel.from_pretrained("../python1_basic_training/gpt2_finetune").eval()
student_model = GPT2LMHeadModel.from_pretrained("gpt2").train()  # student可以选小一号的模型
tokenizer = GPT2Tokenizer.from_pretrained("../python1_basic_training/gpt2_finetune")

# 移动到GPU（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

# 2. 准备训练数据（简单demo版，可以换成自己数据）
train_texts = [
    "Hello world!",
    "How are you today?",
    "The sky is blue.",
    "I love machine learning.",
    "AI is changing the world."
]

# 自定义Dataset
# 最后的return，将文本转为 k(token ID) 和 v(attention mask)，返回一个字典。.squeeze(0)是为了移除 [1, seq_len] 中的 batch 维度。
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
        return {k: v.squeeze(0) for k, v in inputs.items()}  # 字典推导式

# 使用自定义 Dataset 构造 DataLoader，批大小为 2，打乱数据顺序。
dataset = TextDataset(train_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. 配置 optimizer 定义优化器，使用 AdamW 优化器训练 student 模型。
optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

# 4. 定义 loss 函数  使用 KL 散度（Kullback–Leibler divergence）作为损失函数，衡量 student 输出分布与 teacher 输出分布的差异。
loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

# 5. 训练主循环
num_epochs = 5

for epoch in range(num_epochs):
    print(f"=== Epoch {epoch+1}/{num_epochs} ===")
    total_loss = 0

    for batch in dataloader:  # 将每个 batch 的输入 ID 和 attention mask 移到对应设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Teacher输出（不需要梯度），teacher 推理 前向传播  logits（就是每个token的得分）
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # Student输出，student 推理 前向传播
        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # 计算 KL 散度 Loss，比较 小模型预测的概率 和 大模型给出的概率 差距有多大
        # 学生模型（小模型）学习老师模型（大模型），学的“方向”和“速度”都来自于 loss
        loss = loss_fn(
            student_logits.log_softmax(dim=-1),
            teacher_logits.softmax(dim=-1)
        )

        # 清空梯度，反向传播，更新参数。
        # trainer.train()，是 Hugging Face 的 Trainer 封装类的方法，它内部确实包含了这三步，并且还包括更多内容
        # 根据 KL Loss 的差距，计算出小模型每个参数应该怎么调整
        # 大模型负责“出题 + 给答案”，小模型通过 loss 学习这个答案，loss.backward() 就是让小模型慢慢变得更像大模型的关键一步。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加损失
        total_loss += loss.item()

    # 打印平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished, avg_loss={avg_loss:.4f}")

# 6. 保存 student 模型， 将训练后的 student 模型和 tokenizer 保存到本地目录中，便于后续加载使用。
save_path = "./gpt2_student"
student_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Student 模型保存到 {save_path}")

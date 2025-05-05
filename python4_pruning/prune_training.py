# prune_training.py

import os
import torch
from torch.nn.utils import prune
from transformers import GPT2LMHeadModel

# ==== 配置 ====
model_path = "../python3_distillation/gpt2_student_v2"
save_path = "./gpt2_student_v2_pruned"
prune_ratio = 0.3  # 剪枝比例（30%）

# ==== 创建保存目录 ====
os.makedirs(save_path, exist_ok=True)

# ==== 加载小模型 ====
print("🚀 加载小模型 student_v2 ...")
model = GPT2LMHeadModel.from_pretrained(model_path)

'''
| 部分 | 含义 |
|------|------|
| **模型中所有的线性层** | 指 Transformer 模型中所有的 `torch.nn.Linear` 层，分布在 Attention 模块和 FeedForward 模块里，贯穿整个模型的每一层（Layer）。 |
| **执行** | 对这些 Linear 层**逐个进行处理**。 |
| **不规则（Unstructured）剪枝** | 剪掉单个权重，而不是整个神经元或通道。剪的位置是零散的，没有结构限制。 |
| **L1 剪枝** | 按照每个权重的**绝对值大小（L1范数）排序**，剪掉绝对值最小的部分，认为它们对模型不重要。 |
'''
# ==== 对所有 Linear 层应用 L1 Unstructured 剪枝 ====
print(f"✂️ 开始对 Linear 层进行 L1 剪枝，剪枝比例: {prune_ratio}")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=prune_ratio)  # 每次操作都乘以权重
        # 也可以选择不保留mask，永久剪掉
        prune.remove(module, "weight") # 把对应的那个数直接写为 0，以后也不用乘权重了，永久改写，彻底瘦身

print("✅ 剪枝完成")

# ==== 保存剪枝后模型 ====
print(f"💾 保存剪枝后模型到: {save_path}")
model.save_pretrained(save_path)

print("🏁 剪枝流程结束！")

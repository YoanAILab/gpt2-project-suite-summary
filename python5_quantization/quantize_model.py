# quantize_model.py

import os
import torch
from transformers import GPT2LMHeadModel

# ==== 路径配置 ====
model_path = "../python3_distillation/gpt2_student_v2"
save_path = "./gpt2_student_v2_quantized"
os.makedirs(save_path, exist_ok=True)

# ==== 加载原始小模型 ====
print("🚀 加载小模型 student_v2...")
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# ==== 动态量化（Linear 层） ====
print("⚙️ 开始动态量化模型...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # 仅量化 Linear 层
    dtype=torch.qint8   # 转为 INT8 权重
)

print("✅ 量化完成")

# ==== 保存量化模型（PyTorch 原生方式） ====
print(f"💾 保存量化模型到: {save_path}")

# torch.save(model.state_dict()) 保存的是权重参数（字典形式），不包含结构信息
torch.save(quantized_model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

# 拷贝原始 config.json（否则 transformers 无法加载）
import shutil
shutil.copy(os.path.join(model_path, "config.json"), os.path.join(save_path, "config.json"))

print("🏁 量化流程结束！")


# prune_infer.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ==== 配置 ====
model_path = "./gpt2_student_v2_pruned"
tokenizer_path = "../python3_distillation/gpt2_student_v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 加载模型和分词器 ====
print("🚀 加载剪枝后 student_v2 模型中...")
model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
print("✅ 模型加载完成")

# ==== 推理函数 ====
def infer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_token_id = int(logits[0, -1].argmax())
    last_token = tokenizer.decode([last_token_id], clean_up_tokenization_spaces=True).strip()
    return last_token_id, last_token

# ==== 多组测试 ====
prompts = [
    "Hello world",
    "The sky is",
    "I love",
    "Artificial intelligence is"
]

for i, prompt in enumerate(prompts, 1):
    token_id, token_text = infer(prompt)
    print(f"\n📝 Prompt {i}: {prompt}")
    print(f"🔹 Predicted Token ID: {token_id}")
    print(f"🔹 Predicted Token Text: {token_text}")

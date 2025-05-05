# distill_infer_v2.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. 加载小模型 student
MODEL_DIR = "./gpt2_student_v2"  # 小模型目录
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 加载小 Student 模型中...")
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
print(f"✅ 小 Student 模型加载完成")

# 2. 推理函数（支持单条 or 多条 prompt）
def infer(prompts, max_length=30):
    if isinstance(prompts, str):
        prompts = [prompts]  # 单条转列表

    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        last_logits = logits[:, -1, :]
        top1_token_id = torch.argmax(last_logits, dim=-1).item()
        top1_token = tokenizer.decode([top1_token_id], clean_up_tokenization_spaces=True).strip()

        results.append({
            "prompt": prompt,
            "top1_token_id": top1_token_id,
            "top1_token": top1_token
        })

    return results

# 3. 测试入口
if __name__ == "__main__":
    test_prompts = [
        "Hello world",
        "The sky is",
        "I love",
        "Artificial intelligence is"
    ]

    outputs = infer(test_prompts)

    for i, output in enumerate(outputs):
        print(f"\n📝 Prompt {i+1}: {output['prompt']}")
        print(f"🔹 Predicted Token ID: {output['top1_token_id']}")
        print(f"🔹 Predicted Token Text: {output['top1_token']}")

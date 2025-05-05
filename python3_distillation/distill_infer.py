# distill_infer.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. 加载 student 模型
MODEL_DIR = "./gpt2_student"  # 训练后保存的student模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 加载 student 模型中...")
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
print(f"✅ student 模型加载完成")


# 2. 推理函数（支持单条 or 多条prompt）
def infer(prompts, max_length=30):
    if isinstance(prompts, str):
        prompts = [prompts]  # 转成列表，统一处理

    results = []

    # 对每个 prompt 进行分词，并转为 PyTorch 格式 tensor, 移动到 GPU/CPU 上。
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 得到每个 token 的预测 logits 分数。
        logits = outputs.logits  # [batch_size, sequence_length, vocab_size]

        # 取最后一个位置的预测, argmax 找出得分最高的 token ID
        last_logits = logits[:, -1, :]  # 只取最后一层
        top1_token_id = torch.argmax(last_logits, dim=-1).item()

        # 将 token ID 解码成人类可读的 token 文本
        top1_token = tokenizer.decode([top1_token_id], clean_up_tokenization_spaces=True).strip()

        # 将当前 prompt 的推理结果保存下来（包含原始 prompt、token id、token 文字）。
        results.append({
            "prompt": prompt,
            "top1_token_id": top1_token_id,
            "top1_token": top1_token
        })

    return results


# 3. 测试入口
if __name__ == "__main__":  # 表明下面的代码只在直接运行脚本时生效（不会在被 import 时执行）。
    test_prompts = [
        "Hello world",
        "The sky is",
        "I love",
        "Artificial intelligence is"
    ]

    outputs = infer(test_prompts)

    for i, output in enumerate(outputs):
        print(f"\n📝 Prompt {i + 1}: {output['prompt']}")  # 输入 prompt；
        print(f"🔹 Predicted Token ID: {output['top1_token_id']}") # 模型预测的下一个 token 的 ID；
        print(f"🔹 Predicted Token Text: {output['top1_token']}")  # 该 token 对应的文本。

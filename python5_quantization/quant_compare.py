import torch
import time
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = "cpu"

# 路径配置
orig_model_path = "../python3_distillation/gpt2_student_v2"
quant_model_path = "./gpt2_student_v2_quantized"
tokenizer = GPT2Tokenizer.from_pretrained(orig_model_path)

# 加载原始模型
model_orig = GPT2LMHeadModel.from_pretrained(orig_model_path).to(device).eval()

# 加载量化模型
model_quant = GPT2LMHeadModel.from_pretrained(quant_model_path)
model_quant = torch.quantization.quantize_dynamic(model_quant, {torch.nn.Linear}, dtype=torch.qint8)
model_quant.load_state_dict(torch.load(os.path.join(quant_model_path, "pytorch_model.bin"), map_location=device))
model_quant.to(device).eval()

# 对比函数
def benchmark(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        out = model(**inputs)
    end = time.time()
    token_id = int(out.logits[0, -1].argmax())
    token_text = tokenizer.decode([token_id]).strip()
    return token_id, token_text, (end - start) * 1000

prompts = [
    "Hello world",
    "The sky is",
    "I love",
    "Artificial intelligence is",
    "Python is a popular"
]

# 输出对比结果
for i, prompt in enumerate(prompts, 1):
    id1, text1, time1 = benchmark(model_orig, prompt)
    id2, text2, time2 = benchmark(model_quant, prompt)
    match = "✔️ 一致" if id1 == id2 else "❌ 不同"
    speedup = time1 - time2
    print(f"\n📝 Prompt {i}: {prompt}")
    print(f"  原始模型: {text1:<10} ⏱ {time1:.2f} ms")
    print(f"  量化模型: {text2:<10} ⏱ {time2:.2f} ms")
    print(f"  ✅ 输出是否一致: {match} | 加速: {speedup:.2f} ms")

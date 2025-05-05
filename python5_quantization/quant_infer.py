# quant_infer.py

import torch
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ==== 配置 ====
model_path = "./gpt2_student_v2_quantized"
tokenizer_path = "../python3_distillation/gpt2_student_v2"
device = "cpu"  # INT8 动态量化一般在 CPU 上推理效果更佳, GPU 不支持 INT8 动态量化

# ==== 加载 tokenizer ====
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

'''
Transformers 的 from_pretrained() 会做什么？
加载 config.json 构建模型结构（默认 float32，未量化结构）；
加载 pytorch_model.bin 权重（你保存的是 INT8 值）；
不会修改结构，即使权重是 INT8。
也就是说，权重虽然变了，但结构还是原始的 GPT2 模型，没有任何量化层的替换！

| 场景                         | 输入模型的权重 | 输入模型的结构 | `quantize_dynamic` 做了什么               |
| -------------------------- | ------- | ------- | ------------------------------------- |
| ✅ 第一次（`quantize_model.py`） | float32 | 原始结构    | ✅ 量化权重 + ✅ 替换结构                       |
| ✅ 第二次（`quant_infer.py`）    | INT8 权重 | 原始结构    | ❌ 不再量化（权重已是 INT8）<br>✅ 只是替换结构（让结构对得上） |

torch.save(model.state_dict()) 保存的是权重参数（字典形式），不包含结构信息。
所以推理时你需要：
用 GPT2LMHeadModel.from_pretrained(...) 先恢复原始结构（非量化的 Linear）；
再用 quantize_dynamic(...) 替换结构；
然后 load_state_dict(...) 恢复 INT8 权重。
'''
# ==== 初始化模型结构 + 加载权重 ====
print("🚀 加载 INT8 量化模型中...")
model = GPT2LMHeadModel.from_pretrained(model_path)
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=device))  # 加载权重
model.to(device).eval()
print("✅ 模型加载完成")

# ==== 推理函数 ====
def infer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()
    logits = outputs.logits
    token_id = int(logits[0, -1].argmax())
    token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True).strip()
    elapsed = (end - start) * 1000
    return token_id, token_text, elapsed

# ==== 多组测试 ====
prompts = [
    "Hello world",
    "The sky is",
    "I love",
    "Artificial intelligence is",
    "Python is a popular"
]

for i, prompt in enumerate(prompts, 1):
    token_id, token_text, elapsed = infer(prompt)
    print(f"\n📝 Prompt {i}: {prompt}")
    print(f"🔹 Predicted Token ID: {token_id}")
    print(f"🔹 Predicted Token Text: {token_text}")
    print(f"⏱️ 推理耗时: {elapsed:.2f} ms")

# quant_api_server.py

from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# ==== 配置路径 ====
model_path = "./gpt2_student_v2_quantized"
tokenizer_path = "../python3_distillation/gpt2_student_v2"
device = "cpu"  # 动态量化模型推荐用 CPU 推理

# ==== 初始化 Flask 应用 ====
app = Flask(__name__, static_folder="static", template_folder="templates")

# ==== 加载 tokenizer ====
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# ==== 初始化模型结构 + 加载 INT8 权重 ====
print("🚀 加载量化 student_v2 模型中...")
model = GPT2LMHeadModel.from_pretrained(model_path)
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
model.to(device).eval()
print("✅ 模型加载完成")

# ==== 网页首页 ====
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ==== 推理接口 ====
@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        token_id = int(logits[0, -1].argmax())
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True).strip()
        if not token_text:
            token_text = f"[token_id={token_id}]"

        return jsonify({
            "response": f"量化模型生成词: {token_text}",
            "token_id": token_id
        })

    except Exception as e:
        print("❌ 推理失败:", e)
        return jsonify({"error": str(e)}), 500

# ==== 启动服务 ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6006))
    app.run(host="0.0.0.0", port=port)

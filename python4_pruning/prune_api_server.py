# prune_api_server.py

from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# ==== 模型与tokenizer路径 ====
model_path = "./gpt2_student_v2_pruned"
tokenizer_path = "../python3_distillation/gpt2_student_v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 初始化 Flask 应用 ====
app = Flask(__name__, static_folder="static", template_folder="templates")

# ==== 加载模型 ====
print("🚀 加载剪枝后 student_v2 模型中...")
model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
print("✅ 模型加载完成")

# ==== 网页入口 ====
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
            "response": f"剪枝模型生成词: {token_text}",
            "token_id": token_id
        })

    except Exception as e:
        print("❌ 推理失败:", e)
        return jsonify({"error": str(e)}), 500

# ==== 启动服务 ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6006))
    app.run(host="0.0.0.0", port=port)

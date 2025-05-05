from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

app = Flask(__name__)

# ✅ 模型路径（使用量化版模型目录）
MODEL_PATH = "./gpt2_student_v2_quantized"

print("🚀 加载量化 student_v2 模型中...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()
print("✅ 模型加载完成")

@app.route("/", methods=["GET"])
def index():
    return "Welcome to the GPT-2 Quantized API Server"

@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        print(f"📥 收到 prompt: {prompt}")

        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        pred_id = int(torch.argmax(logits[0, -1]))
        pred_token = tokenizer.decode([pred_id], clean_up_tokenization_spaces=True).strip()

        return jsonify({"response": f"模型生成词: {pred_token}", "token_id": pred_id})

    except Exception as e:
        print("❌ 推理失败:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # ✅ 启动服务监听所有地址（用于容器暴露）
    app.run(host="0.0.0.0", port=6006)

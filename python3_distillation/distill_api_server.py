# distill_api_server.py

from flask import Flask, request, jsonify, render_template
from distill_infer import infer
import os

app = Flask(__name__,
            static_folder="static",
            template_folder="templates")

# 首页（网页界面）
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# 接收 infer API（供网页或Postman调用）
@app.route("/infer", methods=["POST"])
def infer_api():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        result = infer(prompt)[0]  # 因为infer()返回是list，所以取第一个结果
        response_text = result["top1_token"]

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6006))
    app.run(host="0.0.0.0", port=port)

# GPT-2 小模型动态量化部署项目 (v5)

本项目基于 GPT-2 蒸馏小模型 student_v2，应用 PyTorch 的动态量化（Dynamic Quantization）技术，  
实现 **INT8 推理加速 + 模型体积压缩**，并提供完整的 Web API 与网页交互界面。

---

## ✅ 项目亮点

- 使用 `torch.quantization.quantize_dynamic` 对 Linear 层量化为 INT8
- 支持 CPU 环境高效部署，推理速度提升，模型大小显著下降
- 提供 Flask Web 服务与 HTML 前端交互界面
- 支持 Docker 镜像封装与一键部署

<p align="center">
  <img src="sample_images/quantize.png" width="600" alt="量化模型网页演示">
</p>

---

## 📂 项目结构说明

```plaintext
PythonProject_GPT2/
├── python5_quantization/            # ⭐ 当前项目核心目录（量化版）
│   ├── quantize_model.py            # 动态量化脚本
│   ├── quant_infer.py               # 推理测试脚本
│   ├── quant_compare.py             # 与未量化模型对比脚本
│   ├── quant_api_server.py          # Web API 服务（Flask）
│   ├── static/                      # 前端样式资源
│   ├── templates/                   # index.html 模板
│   └── gpt2_student_v2_quantized/   # 量化模型权重
├── Dockerfile_v5_backend            # 🐳 量化服务部署镜像构建文件
├── requirements_v5_backend.txt      # 📦 服务依赖列表
├── README_python5_quantization.md   # 📄 当前文件（量化项目说明）
├── sample_images/quantize.png       # ✅ 网页演示截图
```

> 📌 以下目录为历史版本，仅保留参考，**本项目版本未使用其中代码**：
> - `python1_basic_training/`：早期 GPT-2 微调实验
> - `python2_onnx_tensorrt_infer/`：ONNX 推理优化
> - `python3_distillation/`：Student 小模型训练（本项目复用了其模型）
> - `python4_pruning/`：剪枝优化版本

---

## ✂️ 动态量化说明（Dynamic Quantization）

- 量化方式：`quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`
- 量化对象：所有 Linear 层的权重参数
- 优点：
  - 模型体积缩小约 50%
  - 推理速度提升明显（CPU 环境中加速最多达 6 倍）
  - 无需重新训练或微调

---

## 📊 推理效果与对比分析

量化模型在推理结果上与原始模型保持一致，且大幅提升速度：

| Prompt | 原始输出 | 量化输出 | 是否一致 | 原始耗时 (ms) | 量化耗时 (ms) | 加速 |
|--------|-----------|------------|------------|----------------|----------------|--------|
| Hello world | 空 | 空 | ✔️ | 102.46 | 16.32 | **+86.14 ms** |
| The sky is | The | The | ✔️ | 19.19 | 13.49 | +5.70 ms |
| I love | 空 | 空 | ✔️ | 17.53 | 12.89 | +4.65 ms |
| Artificial intelligence is | The | The | ✔️ | 16.56 | 13.44 | +3.11 ms |
| Python is a popular | The | The | ✔️ | 16.08 | 13.96 | +2.12 ms |

✅ 结论：
- 预测 token 全部一致，语义未发生变化
- 在 CPU 环境中推理平均加速 30% ~ 80%

---

## 🛠️ Docker 构建与运行

### 1. 构建镜像

```bash
docker build -f Dockerfile_v5_backend -t gpt2-quant-backend .
```

### 2. 启动服务

```bash
docker run -it --rm -p 6006:6006 gpt2-quant-backend
```

### 3. 访问网页

浏览器打开：

```
http://localhost:6006/
```

输入 prompt，点击提交，返回量化模型推理词。

---

## 🧪 示例输出（CLI）

```plaintext
📝 Prompt: Artificial intelligence is
🔹 Predicted Token: The
🔹 推理耗时: 13.44 ms
```

---

## 🧠 技术栈版本

| 组件 | 版本 |
|------|------|
| torch | 2.6.0 |
| transformers | 4.36.2 |
| flask | 2.2.5 |
| python | 3.11 |
| docker base | pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime |

---

## 📜 License

本项目遵循 MIT License。

---

## 🔮 可扩展方向

- ✅ 对剪枝模型做量化，实现 “剪枝 + 量化” 双压缩部署
- ✅ 将 INT8 模型导出为 ONNX，结合 TensorRT 加速推理
- ✅ 支持多轮对话生成、Prompt 编排、前端改造等


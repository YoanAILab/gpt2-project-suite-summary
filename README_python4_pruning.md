# GPT-2 小模型剪枝部署项目 (v4)

本项目基于 GPT-2 小模型 student_v2（已蒸馏）进行结构外剪枝，  
通过 `torch.nn.utils.prune` 对模型权重进行稀疏化处理，  
并封装为 API 服务和网页交互形式，支持轻量化部署与演示。

✅ 项目特点：
- 使用 L1 Unstructured 剪枝压缩小模型参数
- 支持剪枝比例可调（默认 30%）
- 封装 Flask 接口服务 + 网页 UI
- 可通过 Docker 一键部署
- 页面演示截图如下：

<p align="center">
  <img src="sample_images/prune.png" width="600" alt="剪枝版演示页面">
</p>

---

## 📂 项目结构说明

```plaintext
PythonProject_GPT2/
├── python4_pruning/               # ⭐ 本项目核心目录（剪枝版）
│   ├── prune_training.py          # 剪枝训练脚本（从小模型加载并剪枝）
│   ├── prune_infer.py             # 剪枝后模型推理测试
│   ├── prune_compare.py           # 剪枝 vs 原始 推理速度与一致性对比
│   ├── prune_api_server.py        # 剪枝模型API服务（Flask）
│   ├── gpt2_student_v2_pruned/    # 剪枝后模型权重
│   ├── static/                    # 网页样式
│   └── templates/                 # 网页模板
├── Dockerfile_v4_backend          # 🚀 剪枝服务镜像构建文件
├── requirements_v4_backend.txt    # 📦 剪枝服务依赖文件
├── README_python4_pruning.md      # 📄 当前文件（剪枝项目说明）
├── sample_images/prune.png        # ✅ 剪枝版演示截图
```

> 📌 以下目录为项目历史版本，仅保留参考，**本项目版本未使用**：
> - `python1_basic_training/`：最早训练实验
> - `python2_onnx_tensorrt_infer/`：ONNX + TensorRT 加速实验
> - `python3_distillation/`：student_v2 蒸馏训练（本项目仅使用其模型权重）

---

## ✂️ 剪枝说明

- 剪枝方式：`torch.nn.utils.prune.l1_unstructured`
- 剪枝层：所有 `nn.Linear` 层的 `weight`
- 剪枝比例：默认 30%
- 剪枝完成后保存模型至 `gpt2_student_v2_pruned/`

---

## 🛠️ Docker 构建与运行

### 1. 构建镜像

```bash
docker build -f Dockerfile_v4_backend -t gpt2-prune-backend .
```

### 2. 启动服务

```bash
docker run -it --rm -p 6006:6006 gpt2-prune-backend
```

### 3. 访问网页

浏览器打开：

```
http://localhost:6006/
```

输入 prompt，点击提交，查看剪枝模型的推理结果。

---

## 🧪 示例输出（CLI版）

```plaintext
📝 Prompt: The sky is
🔹 Predicted Token Text: The
```

> 剪枝会影响模型预测能力，但可极大压缩参数、提升推理效率

---

## 📊 剪枝前后对比结果

我们使用相同的 prompt，对比了剪枝前后模型的推理输出和耗时：

| Prompt | 原始输出 | 剪枝输出 | 是否一致 | 原始耗时 (ms) | 剪枝耗时 (ms) | 加速比 |
|--------|-----------|------------|------------|----------------|----------------|--------|
| Hello world | [] | [] | ✔️ | 110.08 | 3.99 | **+96.4%** |
| The sky is | The | The | ✔️ | 4.14 | 5.00 | -20.6% |
| I love | [] | [] | ✔️ | 4.00 | 4.02 | -0.5% |
| Artificial intelligence is | The | The | ✔️ | 9.08 | 4.00 | **+55.9%** |
| Python is a popular | The | The | ✔️ | 4.01 | 3.01 | **+24.9%** |

✅ 总结：
- 预测一致性达 **100%**
- 推理速度平均提升明显，个别 prompt 提升 **50%+**
- 部署剪枝模型后，在资源受限场景下更具实用性

---

## 🛠️ 技术栈版本

| 组件 | 版本 |
|:---|:---|
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

- ✂️ 更高比例剪枝 + 误差对比分析
- 🧪 剪枝后微调恢复性能
- 🧊 结合 INT8 量化做双重压缩
- ⚙️ 使用 ONNX 导出剪枝模型，结合 TensorRT 加速

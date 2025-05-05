#!/bin/bash

# 启动 PyTorch 多进程分布式推理（通信版）
echo "🚀 启动多进程推理（dist_infer_demo_v2.py）..."

# 在单 GPU 场景下，gloo 更通用，不依赖 GPU 数量
# nccl 主要用于多 GPU 通信，特别是分布式训练时

torchrun --nproc_per_node=4 dist_infer_demo_v2.py

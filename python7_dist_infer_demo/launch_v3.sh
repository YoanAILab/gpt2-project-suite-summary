#!/bin/bash
echo "🚀 启动多进程推理（dist_infer_demo_v3.py）..."
torchrun --nproc_per_node=4 dist_infer_demo_v3.py

import os
import torch
import torch.distributed as dist
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

'''
RANK=0            # 第一个进程是 rank 0，第二个是 1，以此类推
WORLD_SIZE=4      # 启动了 4 个进程
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
'''
def setup_distributed():
    dist.init_process_group(
        backend="gloo",  # Windows 和 CPU 环境推荐 gloo，Linux GPU 多卡建议用 nccl
        init_method="env://"  # 告诉 PyTorch 使用环境变量方式建立通信连接，运行 torchrun 系统自动会设置上边的环境变量
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

def main():
    rank, world_size = setup_distributed()
    torch.manual_seed(42) # 设置随机数生成器的种子（seed），以确保程序每次运行结果一致、可复现

    # === 设备分配策略 ===
    if torch.cuda.is_available() and rank == 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # === 加载模型与 tokenizer ===
    model_path = "./gpt2_student_v2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()

    # 总共有 4 个 prompt，分别分配给 4 个进程
    prompts = [
        "Hello world",
        "The sky is",
        "I love",
        "Artificial intelligence is"
    ]

    # 如果某个进程的 rank 超出了 prompt 数量，直接退出
    if rank >= len(prompts):
        print(f"[Rank {rank}] 🚫 无任务，直接退出")
        cleanup()
        return

    prompt = prompts[rank]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()

    logits = outputs.logits
    pred_token_id = int(logits[0, -1].argmax())
    pred_token = tokenizer.decode([pred_token_id]).strip()
    latency_ms = (end - start) * 1000

    result = {
        "rank": rank,
        "prompt": prompt,
        "pred_token_id": pred_token_id,
        "pred_token": pred_token,
        "latency_ms": latency_ms
    }

    # === 进程间通信同步结果 ===
    # 使用 all_gather_object 将每个进程的 result 同步到所有进程。
    # gathered 最终为一个包含所有进程推理信息的列表。
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, result)

    if rank == 0:
        print("\n🚀 所有进程推理结果：")
        # 将所有进程收集到的推理结果（gathered），按 rank 顺序排列，并依次打印每个进程的结果
        for r in sorted(gathered, key=lambda x: x["rank"]):
            print(f"\n[Rank {r['rank']}] 📝 Prompt: {r['prompt']}")
            print(f"[Rank {r['rank']}] 🔹 Predicted Token ID: {r['pred_token_id']}")
            print(f"[Rank {r['rank']}] 🔹 Predicted Token Text: {r['pred_token']}")
            print(f"[Rank {r['rank']}] ⏱ 推理耗时: {r['latency_ms']:.2f} ms")

    cleanup()

if __name__ == "__main__":
    main()

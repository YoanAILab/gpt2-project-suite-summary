import os
import torch
import torch.distributed as dist
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

def setup_distributed():
    dist.init_process_group(
        backend="gloo",  # CPU/单GPU 推荐 gloo，NCCL 用于多 GPU
        init_method="env://"
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def cleanup():
    dist.destroy_process_group()

def main():
    rank, world_size = setup_distributed()
    torch.manual_seed(42)

    # 设备分配策略
    if torch.cuda.is_available() and rank == 0:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 模型加载
    model_path = "./gpt2_student_v2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()

    prompts = [
        "Hello world",
        "The sky is",
        "I love",
        "Artificial intelligence is"
    ]

    if rank >= len(prompts):
        print(f"[Rank {rank}] 🚫 无任务，直接退出")
        cleanup()
        return

    prompt = prompts[rank]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        start = time.time()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False, # 默认推荐部署环境设置为 False（稳定输出、可复现），而创意生成则设为 True
            use_cache=True  # ✅ 开启 KV Cache，在“逐 token 多轮生成”或“长文本+多轮交互”的场景中，KV Cache 会显著提升推理效率。
        )
        end = time.time()

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    latency_ms = (end - start) * 1000

    result = {
        "rank": rank,
        "prompt": prompt,
        "generated_text": generated_text,
        "latency_ms": latency_ms
    }

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, result)

    if rank == 0:
        print("\n🚀 所有进程推理结果（使用 KV Cache）:")
        for r in sorted(gathered, key=lambda x: x["rank"]):
            print(f"\n[Rank {r['rank']}] 📝 Prompt: {r['prompt']}")
            print(f"[Rank {r['rank']}] 🔹 Generated: {r['generated_text']}")
            print(f"[Rank {r['rank']}] ⏱ 推理耗时: {r['latency_ms']:.2f} ms")

    cleanup()

if __name__ == "__main__":
    main()

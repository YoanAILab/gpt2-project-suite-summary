# dist_infer_demo.py
import os
import time
import torch
import torch.distributed as dist  # PyTorch 分布式训练/推理模块
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 初始化分布式环境，使用 gloo 通信后端（CPU上最常用）
def setup_distributed():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()  # 当前进程编号
    world_size = dist.get_world_size()  # 总进程数
    return rank, world_size

# 清理分布式进程组，释放资源
def cleanup_distributed():
    dist.destroy_process_group()

def load_model_and_tokenizer(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# 推理函数（未在 main 中使用）
def infer(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()
    logits = outputs.logits
    pred_token_id = int(torch.argmax(logits[0, -1]))
    pred_token = tokenizer.decode([pred_token_id])
    elapsed_ms = (end - start) * 1000
    return pred_token_id, pred_token.strip(), elapsed_ms

def main():
    rank, world_size = setup_distributed()

    prompts = [
        "Hello world",
        "The sky is",
        "I love",
        "Artificial intelligence is",
        "Python is a popular"
    ]
    prompt = prompts[rank % len(prompts)]

    # rank 0 进程作为“主进程”，打印总体信息
    if rank == 0:
        print(f"🚀 使用 GPT2 Student v2 进行多进程推理（共 {world_size} 个进程）")

    # 仅 rank 0 用 GPU，其余进程使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() and rank == 0 else "cpu")
    model_path = "./gpt2_student_v2"
    tokenizer, model = load_model_and_tokenizer(model_path)
    model.to(device)

    # 将输入 prompt 转换为张量，并移动到设备
    '''
    假如
    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors="pt")
    上面一句执行后，inputs 便是
    {
        'input_ids': tensor([[15496, 995]], device='cuda:0'),
        'attention_mask': tensor([[1, 1]], device='cuda:0')
    }
    '''
    inputs = tokenizer(prompt, return_tensors="pt")
    '''
    下面的字典推导式等价于
    new_inputs = {}
    for k, v in inputs.items():
        new_inputs[k] = v.to(device)
    inputs = new_inputs
    新的 inputs 只是把原来键值对中值的 device 等于 cpu 或 gpu 了
    '''
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()

    logits = outputs.logits
    pred_id = int(torch.argmax(logits[0, -1]))
    pred_token = tokenizer.decode([pred_id])
    elapsed = (end - start) * 1000

    time.sleep(rank * 0.1)  # 加入轻微延迟，错开多进程的打印输出顺序，防止它们在控制台抢输出而导致混乱
    print("\n" + "=" * 60)
    print(f"[Rank {rank}] 📝 Prompt: {prompt}")
    print(f"[Rank {rank}] 🔹 Predicted Token ID: {pred_id}")
    print(f"[Rank {rank}] 🔹 Predicted Token Text: {pred_token.strip()}")
    print(f"[Rank {rank}] ⏱ 推理耗时: {elapsed:.2f} ms")
    print("=" * 60 + "\n")

    cleanup_distributed()

if __name__ == "__main__":
    main()

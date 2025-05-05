import time
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

'''
做 PyTorch 和 TensorRT 推理效果对比的测试脚本。

这个 infer_utils.py 完全没有用到之前的 trt_infer.py。
它是自己重新写了一套推理流程，而且是独立运行的。
🔵 更具体来说：
infer_utils.py 里面自己重新实现了：
TensorRT引擎构建（build_trt_engine）
TensorRT推理（infer_tensorrt）
而不是去 import trt_infer 直接用你之前定义好的 infer_tensorrt 函数。
它连 from trt_infer import infer_tensorrt 这种导入都没有。
✅ 所以整个 infer_utils.py 其实是自给自足的，跟 trt_infer.py 完全没关系，也不会调用它里面的函数。
'''

# 创建一个 TensorRT 的日志对象
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# ====== 通用配置 ======
model_path = "../python1_basic_training/gpt2_finetune"
onnx_path = "model/gpt2.onnx"
prompt = "Hello world"

# 返回 logits + 推理用时（毫秒ms）
# ====== PyTorch 推理 ======
def infer_pytorch(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).eval()
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()

    logits = outputs.logits.detach().numpy()
    elapsed_ms = (end - start) * 1000
    return logits, elapsed_ms


# ====== TensorRT 推理 ======
def build_trt_engine(onnx_path):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        profile = builder.create_optimization_profile()
        profile.set_shape("input_ids", (1, 8), (1, 10), (1, 16))
        profile.set_shape("attention_mask", (1, 8), (1, 10), (1, 16))
        config.add_optimization_profile(profile)

        with open(onnx_path, "rb") as model_file:
            parser.parse(model_file.read())

        return builder.build_engine(network, config)


def infer_tensorrt(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=10)
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)

    engine = build_trt_engine(onnx_path)
    context = engine.create_execution_context()

    input_ids_bind = engine.get_binding_index("input_ids")
    attention_bind = engine.get_binding_index("attention_mask")
    output_bind = engine.get_binding_index("logits")

    context.set_binding_shape(input_ids_bind, input_ids.shape)
    context.set_binding_shape(attention_bind, attention_mask.shape)

    d_input_ids = cuda.mem_alloc(input_ids.nbytes)
    d_attention = cuda.mem_alloc(attention_mask.nbytes)
    output_shape = (1, 10, engine.get_binding_shape(output_bind)[-1])
    output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_input_ids, input_ids)
    cuda.memcpy_htod(d_attention, attention_mask)

    start = time.time()
    context.execute_v2([int(d_input_ids), int(d_attention), int(d_output)])
    end = time.time()

    cuda.memcpy_dtoh(output, d_output)

    elapsed_ms = (end - start) * 1000
    return output, elapsed_ms

'''
在分类、生成类模型推理时，比如 GPT-2 输出 logits（就是每个token的得分），
我们通常关心的是：得分最高的那个 token 是哪个？
这个得分最高的，就是所谓的：
🔵 Top-1（Top-1 Prediction）
'''
# ====== 对比函数 ======
def compare_infer(prompt):
    print(f"\n🚀 对比推理结果（prompt: '{prompt}'）")

    logits_pt, time_pt = infer_pytorch(prompt)
    logits_trt, time_trt = infer_tensorrt(prompt)

    print(f"🔹 PyTorch 耗时：{time_pt:.2f} ms")
    print(f"🔹 TensorRT 耗时：{time_trt:.2f} ms")

    top1_pt = logits_pt[0, -1].argmax()
    top1_trt = logits_trt[0, -1].argmax()
    print(f"🔹 Top-1 Token (PyTorch)   : {top1_pt}")
    print(f"🔹 Top-1 Token (TensorRT) : {top1_trt}")
    print(f"✅ 输出是否一致：{'✅ 是' if top1_pt == top1_trt else '❌ 否'}")

'''
那么如果你直接写 compare_infer(prompt)，
import 也会直接开始跑推理，那就会很尴尬（人家只是想引用个小工具，不是要跑你的测试）。

在每一个 Python 文件中，Python 默认会给你提供一个内置变量 __name__
直接运行这个文件（python xxx.py）  内置变量 __name__ ==	"__main__"
这个文件被别人 import 了（比如 import xxx）	"模块名"，也就是文件名（不带 .py） 内置变量 __name__ == "模块名"
'''
# ====== 运行测试 ======
if __name__ == "__main__":
    compare_infer(prompt)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from transformers import GPT2Tokenizer

# TensorRT 是专门为 ONNX 或 TensorRT Engine 格式 设计的, 原来在 gpt2_finetune 中的模型无法处理
# 初始化 Logger 和路径 TRT_LOGGER：TensorRT 日志对象，只输出 ERROR 级别及以上的日志
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
onnx_path = "model/gpt2.onnx"
model_path = "../python1_basic_training/gpt2_finetune"

'''
程序启动
 ↓
创建上下文（make_context）
 ↓
推理时 push 进入
 ↓
执行 TensorRT 推理
 ↓
推理完 pop 出来

显式管理 CUDA 上下文是因为：
TensorRT 推理中，显卡资源分配要精准控制，否则多线程或不同程序容易出错。
'''
# === 初始化显式 CUDA context ===
cuda.init()  # 初始化CUDA
DEVICE = cuda.Device(0)  # 选第0号显卡
CTX = DEVICE.make_context()  # 创建一个显式上下文
# 如果不手动创建，PyTorch、TensorFlow这类框架内部会帮你隐式创建，不好控制

'''
定义构建 TensorRT 引擎的函数
创建 builder、network、parser，三件套。
EXPLICIT_BATCH：明确指定 batch 维度（必要的，否则 ONNX 和 TensorRT理解的维度对不上）。
同时打开三样东西：
一个 Builder（建造器） ➔ builder
一个 Network（神经网络结构） ➔ network
一个 Parser（解析器） ➔ parser
用来后续构建 TensorRT 引擎。

trt.Builder(TRT_LOGGER)	新建一个 TensorRT 的"建造器"，可以建引擎、建配置
builder.create_network(flags)	创建一个新的神经网络（Network），flags参数指定是否使用显式 batch（很重要）
trt.OnnxParser(network, TRT_LOGGER)	基于刚才的 network，新建一个 ONNX 解析器，可以把 ONNX 文件解析到 Network 里面
✅ 建 Builder ➔ ✅ 建 Network ➔ ✅ 用 Parser 把 ONNX 导入 Network
1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 意思是把1往左移动 EXPLICIT_BATCH 位
1 << 1 → 00000010
1 << 2 → 00000100
1 << 3 → 00001000

0000 0000   # 所有功能都关
0000 0010   # 开了 EXPLICIT_BATCH（第1位）
0000 0100   # 开了 FP16（第2位）
0000 1000   # 开了 INT8（第3位）
0000 0010 | 0000 0100 = 0000 0110  （= 6）同时开EXPLICIT_BATCH和FP16

"with" 是Python的语法糖，用来
👉 自动管理资源的申请和释放！
TensorRT对象（builder、network、parser）都是C++底层对象
如果你自己忘了 .destroy()，就会造成显存泄露、程序崩溃
用 with，Python会自动帮你销毁资源，非常安全！

with 开启
  ↓
Builder (负责建引擎)
  ↓
Network (搭网络)
  ↓
Parser (读ONNX文件到Network里)
  ↓
with 退出时自动释放资源
'''
# === 构建 TensorRT 引擎 ===
def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # 创建一个 TensorRT 构建配置对象（config） config 里面可以配置很多选项
        # 位运算，表示 2³⁰ = 1073741824 字节，即 1GB，设置 TensorRT 的中间计算临时空间（workspace）最大只能用 1GB 内存
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # 配置动态 Shape Profile 输入尺寸的范围定义
        '''
        profile.set_shape("input_ids", (1, 8), (1, 16), (1, 32))
        最小输入	(1, 8)	batch=1, sequence_length=8（最短输入8个token）
        最优输入	(1, 16)	batch=1, sequence_length=16（常见情况）
        最大输入	(1, 32)	batch=1, sequence_length=32（最长支持32个token）
        '''
        profile = builder.create_optimization_profile()
        profile.set_shape("input_ids", (1, 8), (1, 16), (1, 32))
        profile.set_shape("attention_mask", (1, 8), (1, 16), (1, 32))
        config.add_optimization_profile(profile)

        # 加载并解析 ONNX
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("❌ ONNX 解析失败")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # 返回编译好的 Engine 真正开始编译成 TensorRT 二进制执行计划
        return builder.build_engine(network, config)


# 模块级的全局变量，只要 trt_infer 这个模块被 import，这些对象就会在第一次加载时初始化好并常驻内存。
print("🚀 初始化 TensorRT 引擎和分词器...")
ENGINE = build_engine(onnx_path)  # TensorRT执行引擎, 轻则几十秒，重则几分钟, 我的跑了70s
# 如果保存成 .plan 文件（这是 TensorRT引擎的后缀），后面每次部署时，直接反序列化加载 .plan 文件，几乎是秒级启动的
# 别的文件导入 该文件 trt_infer 的时候，Python 会自动执行 ENGINE = build_engine(onnx_path)！
CONTEXT = ENGINE.create_execution_context()  # 执行时用的 context
TOKENIZER = GPT2Tokenizer.from_pretrained(model_path)  # 文本分词器

# trt_infer.py 作为一个模块来使用的，如果直接运行 python trt_infer.py	初始化完就退出了（因为没有调用推理）
# import trt_infer	把 ENGINE、CONTEXT、infer_tensorrt 加载到内存，而不会直接执行任何函数调用，等你需要的时候才调用
# 调用 infer_tensorrt(prompt)	手动执行推理，显式 push / pop 上下文，推理结束后资源释放
# === 推理函数 ===
def infer_tensorrt(prompt):
    CTX.push()  # ✅ 显式进入上下文
    # 手动 "激活" CUDA 上下文，保证所有显存申请和内存拷贝都在自己开的 context 下执行

    # 把 prompt 编成 input_ids 和 attention_mask，返回 numpy 格式。
    # 并转为 int32（TensorRT 要求的精度）。
    # numpy格式，指的是 用numpy这个库表示的数据格式 可以高效地在内存里连续存储，非常适合拿来做数值计算、机器学习、深度学习推理等。
    try:
        inputs = TOKENIZER(prompt, return_tensors="np", padding="max_length", truncation=True, max_length=16)
        input_ids = inputs["input_ids"].astype(np.int32)
        attention_mask = inputs["attention_mask"].astype(np.int32)

        # 根据模型里绑定的名字，查到对应索引，推理时要用到。
        input_ids_bind = ENGINE.get_binding_index("input_ids")
        attention_bind = ENGINE.get_binding_index("attention_mask")
        output_bind = ENGINE.get_binding_index("logits")

        # 明确告诉 TensorRT：这次推理用的 input shape 是多少（动态 batch 支持）。
        CONTEXT.set_binding_shape(input_ids_bind, input_ids.shape)
        CONTEXT.set_binding_shape(attention_bind, attention_mask.shape)

        # 申请显存，还在 CPU 上建了个空的 output 数组，用来拿回推理结果
        d_input_ids = cuda.mem_alloc(input_ids.nbytes)
        d_attention = cuda.mem_alloc(attention_mask.nbytes)
        output_shape = (1, 16, ENGINE.get_binding_shape(output_bind)[-1])
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        cuda.memcpy_htod(d_input_ids, input_ids)
        cuda.memcpy_htod(d_attention, attention_mask)

        # 执行推理，计时
        start = time.time()
        CONTEXT.execute_v2([int(d_input_ids), int(d_attention), int(d_output)])
        end = time.time()

        # 拷贝输出回主机内存
        cuda.memcpy_dtoh(output, d_output)
        elapsed = (end - start) * 1000

        return output, elapsed

    finally:
        CTX.pop()  # ✅ 推理完成后释放上下文，无论成功失败，最后都要 pop()，释放掉 CUDA 上下文，防止资源泄露！

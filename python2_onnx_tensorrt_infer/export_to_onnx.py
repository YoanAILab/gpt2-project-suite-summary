import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

model_path = "../python1_basic_training/gpt2_finetune"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# 默认就会去加载 tokenizer_config.json + vocab.json + merges.txt + （可选）special_tokens_map.json
model = GPT2LMHeadModel.from_pretrained(model_path)
# 默认就会去加载 配置文件 config.json 模型结构配置（层数、头数、隐藏层等），权重文件 model.safetensors 模型的全部参数（就是原生模型）
model.eval()

# 构造包装类，避免 trace 时访问 past_key_values 报错
# 在 PyTorch 里，模型(x) 就是调用 模型的 forward(x)，会自动调用的
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.config.use_cache = False  # 显式关闭缓存，ONNX 不支持这种动态缓存，所以必须关掉

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return outputs.logits  # logits 是模型给每个词预测输出打的原始分数，大的更可能，softmax(logits) 变成真正的百分比概率

wrapped_model = Wrapper(model)  # 创建包装后的模型 此时可以安全导出 ONNX 的模型了

# 构造 dummy 输入，来告诉系统：将来模型推理时，输入的数据是什么样子的，ONNX 导出是静态图，要提前确定输入结构。
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]    #这两句就是把字典里面的两个元素取出来，分别赋值给变量，方便后面使用
attention_mask = inputs["attention_mask"]

onnx_path = "model/gpt2.onnx"
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

# 导出 ONNX 模型
torch.onnx.export(
    wrapped_model,
    (input_ids, attention_mask),  # 输入示例，告诉导出工具输入结构。
    onnx_path,
    input_names=["input_ids", "attention_mask"],  # 给输入和输出起名字
    output_names=["logits"],
    dynamic_axes={  # 动态 batch size、动态 sequence length（✅ 允许输入长度不固定！很重要）。
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    },
    opset_version=14  # ✅ 指定 ONNX 版本，这里用14版是因为支持更多新操作，比如 Flash Attention
)

print(f"✅ 模型成功导出为 ONNX：{onnx_path}")

import torch

from transformers import AutoTokenizer

def average_dict_list(dict_list):
    """将具有相同键的字典列表合并为新字典，每个键的值为平均值"""
    if not dict_list:
        return {}
    
    # 获取所有键（假设所有字典键相同）
    keys = dict_list[0].keys()
    result = {}
    
    for key in keys:
        # 收集所有字典中该键的值并计算平均值
        values = [d[key] for d in dict_list if key in d]
        # 处理数值类型（整数/浮点数）
        if all(isinstance(v, (int, float)) for v in values):
            result[key] = sum(values) / len(values)
        else:
            # 非数值类型可根据需求处理（如保留第一个值/拼接）
            result[key] = values[0]
    
    return result


dict_list = [
    {"a": 10, "b": 20, "c": 30},
    {"a": 20, "b": 30, "c": 40},
    {"a": 30, "b": 40, "c": 50}
]

# 计算平均值字典
avg_dict = average_dict_list(dict_list)
print("平均值字典：", avg_dict)
# 输出：平均值字典： {'a': 20.0, 'b': 30.0, 'c': 40.0}

exit(0)

# 加载预训练tokenizer（示例使用GPT-2，可替换为其他模型如Llama、BERT等）
tokenizer = AutoTokenizer.from_pretrained("/path/to/models/DeepSeek-R1-Distill-Qwen-1.5B")
y=tokenizer.encode("the positive x-axis. \n\nFirst,")

print(y)
for token in y:
    t=tokenizer.convert_ids_to_tokens([token])
    print(t)
    print(token)


# Create two tensors with 1GB memory footprint each, initialized randomly, in fp16 format
# For a tensor of float16 (2 bytes), 1GB of memory can hold 1GB / 2B = 500M elements
tensor_size = 512 * 256 * 256 
x = torch.randn(tensor_size, dtype=torch.float16, device='cuda')
y = torch.randn(tensor_size, dtype=torch.float16, device='cuda')

# Record current memory footprint, and reset max memory counter
current_memory = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()

def compute(x, y):
    return (x + 1) * (y + 1)

z = compute(x, y)

# Record the additional memory (both peak memory and persistent memory) after calculating the resulting tensor
additional_memory = torch.cuda.memory_allocated() - (current_memory + 1e9)
peak_memory = torch.cuda.max_memory_allocated()
additional_peak_memory = peak_memory - (current_memory + 1e9)

print(f"Additional memory used: {additional_memory / (1024 ** 3)} GB")
print(f"Additional peak memory used: {additional_peak_memory / (1024 ** 3)} GB")

# git rm -r --cached /path/to/SAGE/verl/workers/rollout/vllm_rollout/rollout_outputs
# git rm -r --cached /path/to/SAGE/verl/workers/rollout/vllm_rollout/val_rollout_outputs


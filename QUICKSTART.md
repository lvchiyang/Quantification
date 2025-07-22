# 快速入门指南

这个指南将帮助你快速上手 Decoder-only Transformer with MLA 项目。

## 前置要求

- Python 3.8+
- PyTorch 2.1+
- 8GB+ RAM (推荐)
- GPU (可选，但推荐用于训练)

## 5分钟快速体验

### 1. 安装依赖

```bash
pip install torch einops transformers datasets numpy tqdm matplotlib
```

### 2. 验证安装

```bash
python simple_test.py
```

如果看到 "All tests passed!" 说明安装成功！

### 3. 查看功能演示

```bash
python examples/demo.py --demo creation
```

这将展示不同大小的模型配置。

## 核心功能体验

### 创建和测试模型

```python
# 导入必要模块
from src.config import ModelConfigs
from src.transformer import Transformer
import torch

# 创建一个小型模型
args = ModelConfigs.tiny()
model = Transformer(args)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 测试前向传播
input_ids = torch.randint(0, args.vocab_size, (2, 32))
outputs = model(input_ids)
print(f"输出形状: {outputs['logits'].shape}")

# 测试损失计算
outputs = model(input_ids, labels=input_ids)
print(f"损失值: {outputs['loss'].item():.4f}")
```

### 文本生成示例

```python
# 设置模型为评估模式
model.eval()

# 创建输入
input_ids = torch.randint(0, args.vocab_size, (1, 10))

# 生成文本
with torch.no_grad():
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        temperature=0.8,
        do_sample=True
    )

print(f"输入长度: {input_ids.shape[1]}")
print(f"生成长度: {generated.shape[1]}")
```

## 训练模型

### 快速训练测试

```bash
# 训练一个微型模型（几分钟完成）
python examples/train.py \
    --model_size tiny \
    --num_epochs 1 \
    --max_samples 1000 \
    --batch_size 8
```

### 标准训练

```bash
# 训练一个小型模型
python examples/train.py \
    --model_size small \
    --num_epochs 3 \
    --max_samples 10000 \
    --batch_size 16
```

训练完成后，模型会保存在 `./checkpoints/` 目录下。

## 文本生成

### 交互式生成

```bash
python examples/generate.py \
    --checkpoint ./checkpoints/best_checkpoint.pt \
    --mode interactive
```

然后输入提示词，如 "Once upon a time"。

### 批量生成

```bash
python examples/generate.py \
    --checkpoint ./checkpoints/best_checkpoint.pt \
    --mode batch \
    --prompts "Once upon a time" "The little girl" \
    --output_file results.json
```

## 自定义配置

### 创建自定义模型配置

```python
from src.config import ModelArgs

# 自定义配置
custom_args = ModelArgs(
    d_model=512,
    n_layers=6,
    n_heads=8,
    kv_lora_rank=256,
    qk_rope_head_dim=32,
    qk_nope_head_dim=32,
    v_head_dim=64,
    intermediate_size=1024,
    vocab_size=32000,
    max_seq_len=1024
)

model = Transformer(custom_args)
```

### 使用不同的 FFN

```python
from src.feedforward import get_ffn

# 使用 GeGLU 而不是 SwiGLU
ffn = get_ffn(args, ffn_type="geglu")

# 使用标准 FFN
ffn = get_ffn(args, ffn_type="standard")
```

## 测试和验证

### 运行完整验证

```bash
python final_verification.py
```

### 运行单元测试

```bash
python tests/test_model.py
```

### 模型架构验证

```python
from src.model_utils import validate_model_architecture

validation_results = validate_model_architecture(model, args)
print(f"验证通过: {validation_results['passed']}")
```

## 性能分析

### 模型信息分析

```python
from src.model_utils import save_model_info

# 保存详细的模型信息
save_model_info(model, args, "./model_analysis/")
```

这会生成：
- 配置文件
- 验证报告
- 权重分析
- 性能基准

### 速度基准测试

```python
from src.model_utils import benchmark_model_speed

results = benchmark_model_speed(model)
print(f"前向传播速度: {results['forward_pass_times']}")
```

## 高级功能

### 权重分布可视化

```python
from src.model_utils import plot_weight_distributions

plot_weight_distributions(model, save_path="weights.png")
```

### 注意力机制对比

```python
from src.attention import MLA, MultiHeadAttention

# 创建 MLA 和标准注意力
mla = MLA(args)
std_attn = MultiHeadAttention(args)

# 比较参数数量
mla_params = sum(p.numel() for p in mla.parameters())
std_params = sum(p.numel() for p in std_attn.parameters())

print(f"MLA 参数: {mla_params:,}")
print(f"标准注意力参数: {std_params:,}")
print(f"参数比例: {mla_params/std_params:.2f}")
```

## 常见问题

### Q: 运行时出现内存不足错误
A: 尝试减小批次大小或使用更小的模型配置：
```bash
python examples/train.py --model_size tiny --batch_size 4
```

### Q: 训练速度很慢
A: 确保安装了 GPU 版本的 PyTorch，或者使用更小的模型：
```bash
# 检查 GPU 可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: 生成的文本质量不好
A: 这是正常的，因为模型没有经过充分训练。尝试：
1. 增加训练数据量
2. 延长训练时间
3. 调整超参数

### Q: 如何保存和加载模型
A: 训练脚本会自动保存模型，加载方法：
```python
from src.model_utils import load_model_for_inference

model, args, tokenizer = load_model_for_inference("./checkpoints/best_checkpoint.pt")
```

## 进一步学习

1. **阅读源码**: 从 `src/config.py` 开始，了解模型配置
2. **查看测试**: `tests/test_model.py` 展示了各组件的使用方法
3. **研究论文**: 了解 MLA、RoPE、SwiGLU 等技术的原理
4. **实验改进**: 尝试修改架构或添加新功能

## 恭喜！

你现在已经掌握了 Decoder-only Transformer with MLA 的基本使用方法。这个实现包含了最新的 Transformer 架构改进，可以用于研究和实际应用。

如果遇到问题，请查看 `PROJECT_SUMMARY.md` 获取更详细的信息，或者运行 `python final_verification.py` 进行完整的功能验证。

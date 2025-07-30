# 前馈网络实现文档

专门为金融时序预测设计的前馈网络模块，提供多种前馈网络变体以满足不同的建模需求。

## 📋 目录

- [概述](#概述)
- [网络架构](#网络架构)
- [使用方法](#使用方法)
- [性能对比](#性能对比)
- [最佳实践](#最佳实践)

---

## 🎯 概述

### 设计理念

前馈网络（Feed-Forward Network, FFN）是 Transformer 架构的核心组件之一，负责提供非线性变换能力。本模块实现了多种前馈网络变体：

1. **SwiGLU**：使用 SiLU 激活的门控线性单元（推荐）
2. **GeGLU**：使用 GELU 激活的门控线性单元
3. **StandardFFN**：传统的两层前馈网络
4. **MoEFFN**：专家混合前馈网络（高级用法）

### 核心特性

- **门控机制**：SwiGLU 和 GeGLU 使用门控机制提高表达能力
- **激活函数优化**：使用现代激活函数（SiLU、GELU）
- **可配置性**：支持不同的隐藏维度和 dropout 率
- **专家混合**：MoE 支持条件计算，提高模型容量

---

## 🏗️ 网络架构

### 1. SwiGLU（推荐）

**公式**：`SwiGLU(x) = (SiLU(W1 * x) ⊙ W3 * x) * W2`

```python
class SwiGLU(nn.Module):
    def __init__(self, args: ModelArgs):
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # 门控投影
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # 输出投影  
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # 值投影
        
    def forward(self, x):
        gate = F.silu(self.w1(x))    # 门控分支
        value = self.w3(x)           # 值分支
        hidden = gate * value        # 门控机制
        return self.w2(hidden)       # 输出投影
```

**特点**：
- 使用 SiLU (Swish) 激活函数
- 门控机制提高表达能力
- 无偏置项，减少参数量
- 在大模型中表现优异

### 2. GeGLU

**公式**：`GeGLU(x) = (GELU(W1 * x) ⊙ W3 * x) * W2`

```python
class GeGLU(nn.Module):
    def forward(self, x):
        gate = F.gelu(self.w1(x))    # 使用 GELU 激活
        value = self.w3(x)
        hidden = gate * value
        return self.w2(hidden)
```

**特点**：
- 使用 GELU 激活函数
- 结构与 SwiGLU 相似
- 在某些任务上可能表现更好

### 3. StandardFFN

**公式**：`FFN(x) = W2 * ReLU(W1 * x + b1) + b2`

```python
class StandardFFN(nn.Module):
    def forward(self, x):
        hidden = F.relu(self.linear1(x))
        return self.linear2(hidden)
```

**特点**：
- 传统的两层前馈网络
- 使用 ReLU 激活函数
- 包含偏置项
- 计算简单，适合基线对比

### 4. MoEFFN（专家混合）

**原理**：根据输入动态选择专家网络

```python
class MoEFFN(nn.Module):
    def __init__(self, args, num_experts=8, top_k=2):
        self.gate = nn.Linear(d_model, num_experts)  # 门控网络
        self.experts = nn.ModuleList([               # 专家网络
            SwiGLU(args) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        gate_weights = F.softmax(self.gate(x), dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k)
        # 只激活 top-k 个专家
```

**特点**：
- 条件计算，提高模型容量
- 参数利用率高
- 适合大规模模型
- 训练复杂度较高

---

## 🚀 使用方法

### 基础使用

```python
from src.price_prediction.feedforward import get_ffn
from src.price_prediction.config import PricePredictionConfigs

# 创建配置
config = PricePredictionConfigs.base()

# 创建不同类型的前馈网络
swiglu_ffn = get_ffn(config, ffn_type="swiglu")     # 推荐
geglu_ffn = get_ffn(config, ffn_type="geglu")       # 替代选择
standard_ffn = get_ffn(config, ffn_type="standard") # 基线对比
moe_ffn = get_ffn(config, ffn_type="moe")           # 高级用法

# 前向传播
batch_size, seq_len, d_model = 4, 180, 512
x = torch.randn(batch_size, seq_len, d_model)

output = swiglu_ffn(x)  # [4, 180, 512]
```

### 在 Transformer 中使用

```python
class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = MultiHeadAttention(args)
        self.ffn = get_ffn(args, ffn_type="swiglu")  # 使用 SwiGLU
        self.norm1 = RMSNorm(args.d_model)
        self.norm2 = RMSNorm(args.d_model)
        
    def forward(self, x, freqs_cis):
        # 注意力层
        x = x + self.attn(self.norm1(x), freqs_cis)
        # 前馈网络层
        x = x + self.ffn(self.norm2(x))
        return x
```

### 配置参数

```python
# 在配置文件中设置前馈网络参数
config = PricePredictionConfigs.base()
config.d_model = 512              # 模型维度
config.intermediate_size = 2048   # 前馈网络隐藏维度（通常是 d_model 的 4 倍）
config.dropout = 0.1              # Dropout 率

# 对于 MoE
config.num_experts = 8            # 专家数量
config.top_k_experts = 2          # 激活的专家数量
```

---

## 📊 性能对比

### 计算复杂度

| 网络类型 | 参数量 | FLOPs | 内存占用 | 训练速度 |
|----------|--------|-------|----------|----------|
| StandardFFN | 2 × d_model × hidden_dim | 2 × d_model × hidden_dim | 低 | 快 |
| SwiGLU | 3 × d_model × hidden_dim | 3 × d_model × hidden_dim | 中 | 中 |
| GeGLU | 3 × d_model × hidden_dim | 3 × d_model × hidden_dim | 中 | 中 |
| MoEFFN | num_experts × 3 × d_model × hidden_dim | top_k × 3 × d_model × hidden_dim | 高 | 慢 |

### 表达能力

1. **SwiGLU** > **GeGLU** > **StandardFFN**（一般情况）
2. **MoEFFN** 在大规模数据上表现最好
3. **StandardFFN** 适合快速原型和基线对比

### 金融时序预测中的表现

根据实验结果：

```python
# 推荐配置（平衡性能和效果）
config.ffn_type = "swiglu"
config.intermediate_size = 4 * config.d_model  # 4倍扩展

# 高性能配置（追求最佳效果）
config.ffn_type = "moe"
config.num_experts = 8
config.top_k_experts = 2

# 轻量级配置（快速训练）
config.ffn_type = "standard"
config.intermediate_size = 2 * config.d_model  # 2倍扩展
```

---

## 💡 最佳实践

### 1. 选择指南

**SwiGLU**（推荐）：
- 大多数情况下的最佳选择
- 在金融时序预测中表现优异
- 训练稳定，收敛快

**GeGLU**：
- 当 SwiGLU 过拟合时的替代选择
- 某些数据集上可能表现更好

**StandardFFN**：
- 快速原型开发
- 基线对比
- 计算资源受限时

**MoEFFN**：
- 大规模数据集
- 需要最高模型容量时
- 有充足计算资源

### 2. 超参数调优

```python
# 隐藏维度选择
config.intermediate_size = 4 * config.d_model  # 标准配置
config.intermediate_size = 8 * config.d_model  # 高容量配置
config.intermediate_size = 2 * config.d_model  # 轻量级配置

# Dropout 设置
config.dropout = 0.1   # 标准设置
config.dropout = 0.0   # 大数据集，无过拟合风险
config.dropout = 0.2   # 小数据集，防止过拟合
```

### 3. 训练技巧

1. **梯度裁剪**：使用梯度裁剪防止梯度爆炸
2. **学习率调度**：使用 warmup 和 cosine 衰减
3. **权重初始化**：使用 Xavier 或 He 初始化
4. **正则化**：结合 dropout 和 weight decay

### 4. 调试建议

```python
# 检查前馈网络输出
def debug_ffn(ffn, x):
    print(f"输入形状: {x.shape}")
    output = ffn(x)
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"输出均值: {output.mean():.4f}")
    print(f"输出标准差: {output.std():.4f}")
    return output

# 使用示例
x = torch.randn(4, 180, 512)
ffn = get_ffn(config, "swiglu")
output = debug_ffn(ffn, x)
```

---

## 🔗 相关文件

- `src/price_prediction/feedforward.py` - 前馈网络实现
- `src/price_prediction/config.py` - 配置参数
- `src/price_prediction/price_transformer.py` - Transformer 主模型
- `doc/transformer.md` - Transformer 架构文档

这套前馈网络实现为金融时序预测提供了强大而灵活的非线性变换能力！

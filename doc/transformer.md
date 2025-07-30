# Transformer 架构文档

专门为金融时序预测设计的 Transformer 架构，包含 TransformerBlock 和相关组件的详细说明。

## 📋 目录

- [架构概述](#架构概述)
- [TransformerBlock](#transformerblock)
- [核心组件](#核心组件)
- [使用方法](#使用方法)
- [性能优化](#性能优化)

---

## 🏗️ 架构概述

### 整体设计理念

本 Transformer 架构专门针对金融时序预测任务优化，具有以下特点：

1. **Multi-Head Latent Attention (MLA)**：高效的注意力机制
2. **RoPE 位置编码**：旋转位置编码，适合长序列
3. **SwiGLU 前馈网络**：现代化的前馈网络设计
4. **Pre-RMSNorm**：更稳定的归一化方式
5. **金融特征嵌入**：专门的特征分组嵌入

### 数据流

```
输入: [batch, 180, 20] 金融特征
  ↓
FinancialEmbeddingLayer: 分组嵌入 + 批标准化
  ↓
[batch, 180, d_model] 嵌入特征
  ↓
TransformerBlock × n_layers:
  - RMSNorm → MLA + RoPE → 残差连接
  - RMSNorm → SwiGLU FFN → 残差连接
  ↓
[batch, 180, d_model] 编码特征
  ↓
取最后时间步: [batch, d_model]
  ↓
价格预测头: [batch, 10] 未来10个时间点预测
```

---

## 🧩 TransformerBlock

### 设计原理

TransformerBlock 是 Transformer 的核心组件，采用 Pre-RMSNorm 结构：

```python
class TransformerBlock(nn.Module):
    """
    结构：
    x -> RMSNorm -> MLA -> Add -> RMSNorm -> FFN -> Add
    """
    
    def __init__(self, args, layer_idx: int = 0):
        super().__init__()
        
        # 注意力层
        self.attn_norm = RMSNorm(args.d_model)
        self.attn = MultiHeadLatentAttention(args)
        
        # 前馈网络层
        self.ffn_norm = RMSNorm(args.d_model)
        self.ffn = get_ffn(args, ffn_type="swiglu")
```

### 前向传播

```python
def forward(self, x, freqs_cis, attn_mask=None, is_causal=False):
    # Pre-RMSNorm + MLA + 残差连接
    attn_input = self.attn_norm(x)
    attn_output = self.attn(attn_input, freqs_cis, attn_mask, is_causal)
    x = x + attn_output
    
    # Pre-RMSNorm + FFN + 残差连接
    ffn_input = self.ffn_norm(x)
    ffn_output = self.ffn(ffn_input)
    x = x + ffn_output
    
    return x
```

### 关键特性

1. **Pre-RMSNorm**：
   - 在注意力和FFN之前进行归一化
   - 比 Post-Norm 更稳定，训练更容易

2. **残差连接**：
   - 缓解梯度消失问题
   - 允许更深的网络结构

3. **层索引**：
   - 支持层特定的配置
   - 便于分析不同层的作用

---

## 🔧 核心组件

### 1. Multi-Head Latent Attention (MLA)

**传统注意力 vs MLA**：

```python
# 传统注意力
Traditional: Q@K^T@V  # O(n²d) 复杂度

# MLA
MLA: Q@(W_kv@X)^T@(W_kv@X)  # O(nd) 复杂度，d<<n时高效

# K/V压缩示例
original_kv_dim = 1024    # 原始维度
compressed_dim = 256      # 压缩后维度
compression_ratio = 4     # 压缩比例
```

**MLA实现细节**：

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args):
        # 查询/键头维度，使用标准的d_model/n_heads
        self.qk_head_dim = args.d_model // args.n_heads
        
        # 确保头维度是偶数（RoPE要求）
        assert self.qk_head_dim % 2 == 0
        
        # K/V 潜在投影压缩
        self.kv_compress = nn.Linear(args.d_model, args.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(args.kv_lora_rank)
        
        # 从压缩表示恢复 K/V
        self.k_up = nn.Linear(args.kv_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.v_up = nn.Linear(args.kv_lora_rank, self.n_heads * self.v_dim, bias=False)
        
        # 查询投影（用于RoPE）
        self.q_proj = nn.Linear(args.d_model, self.n_heads * self.qk_head_dim, bias=False)
```

### 2. RoPE (Rotary Position Embedding)

**位置编码原理**：

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 1e4):
    # 计算频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 计算位置索引
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs).float()
    
    # 转换为复数形式
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # 重塑为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 应用旋转
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**RoPE 优势**：
- 相对位置编码，适合长序列
- 旋转不变性，保持向量长度
- 外推能力强，可处理训练时未见过的序列长度

### 3. SwiGLU 前馈网络

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        gate = F.silu(self.w1(x))    # 门控分支
        value = self.w3(x)           # 值分支
        hidden = gate * value        # 门控机制
        return self.w2(hidden)       # 输出投影
```

**特点**：
- 使用 SiLU (Swish) 激活函数
- 门控机制提高表达能力
- 比传统 ReLU FFN 效果更好

### 4. RMSNorm

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

**优势**：
- 计算更简单，无需计算均值
- 训练更稳定
- 在大模型中表现更好

---

## 🚀 使用方法

### 创建模型

```python
from src.price_prediction.price_transformer import PriceTransformer
from src.price_prediction.config import PricePredictionConfigs

# 创建配置
config = PricePredictionConfigs.base()

# 创建模型
model = PriceTransformer(config)

# 前向传播
batch_size, seq_len, n_features = 4, 180, 20
financial_data = torch.randn(batch_size, seq_len, n_features)

outputs = model(financial_data, return_features=True, return_dict=True)
print(f"价格预测: {outputs['price_predictions'].shape}")  # [4, 10]
print(f"特征向量: {outputs['strategy_features'].shape}")    # [4, 512]
```

### 配置参数

```python
# 模型配置
config = PricePredictionConfigs.base()
config.d_model = 512          # 模型维度
config.n_layers = 8          # Transformer层数
config.n_heads = 8           # 注意力头数
config.kv_lora_rank = 256    # K/V压缩维度
config.intermediate_size = 2048  # FFN隐藏维度

# 训练配置
config.learning_rate = 1e-4
config.weight_decay = 0.01
config.dropout = 0.1

# RoPE配置
config.rope_theta = 10000.0
config.max_seq_len = 512
```

### 单独使用 TransformerBlock

```python
from src.price_prediction.price_transformer import TransformerBlock

# 创建单个 Transformer 层
transformer_block = TransformerBlock(config, layer_idx=0)

# 准备输入
x = torch.randn(batch_size, seq_len, config.d_model)
freqs_cis = precompute_freqs_cis(
    dim=config.d_model // config.n_heads,
    end=seq_len,
    theta=config.rope_theta
)

# 前向传播
output = transformer_block(x, freqs_cis)
print(f"输出形状: {output.shape}")  # [batch_size, seq_len, d_model]
```

---

## ⚡ 性能优化

### 1. 内存优化

```python
# 使用MLA减少内存占用
traditional_memory = n_heads * seq_len * seq_len * head_dim  # O(n²)
mla_memory = kv_lora_rank * seq_len + n_heads * head_dim     # O(n)

print(f"内存节省: {traditional_memory / mla_memory:.2f}x")

# 示例计算
seq_len = 180
n_heads = 8
head_dim = 64
kv_lora_rank = 256

traditional = n_heads * seq_len * seq_len * head_dim  # 663,552,000
mla = kv_lora_rank * seq_len + n_heads * head_dim     # 46,592

print(f"传统注意力内存: {traditional:,}")
print(f"MLA内存: {mla:,}")
print(f"节省比例: {traditional / mla:.1f}x")
```

### 2. 计算速度

```python
# 压缩比配置
config.kv_lora_rank = 128    # 高压缩
config.kv_lora_rank = 256    # 标准压缩
config.kv_lora_rank = 512    # 低压缩

# FFN配置
config.intermediate_size = 2 * config.d_model  # 轻量级
config.intermediate_size = 4 * config.d_model  # 标准配置
config.intermediate_size = 8 * config.d_model  # 高容量
```

---

## 🔗 相关文件

- `src/price_prediction/price_transformer.py` - 主Transformer实现
- `src/price_prediction/feedforward.py` - 前馈网络实现
- `src/price_prediction/attention.py` - 注意力机制实现
- `src/price_prediction/embedding.py` - 嵌入层实现
- `doc/feedforward.md` - 前馈网络文档

这套Transformer架构能够为金融时序预测提供强大且高效的编码模型基础！

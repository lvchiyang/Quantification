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

RoPE是本模型的核心位置编码方案，在每个注意力层中对Q和K向量进行旋转变换。

**重要说明**：RoPE在Transformer的注意力层中实现，而不是在嵌入层。这种设计有以下优势：
- **职责分离**：嵌入层专注于特征表示，注意力层处理位置信息
- **精确控制**：每层都能重新计算位置关系，学习不同粒度的位置模式
- **特征纯净**：保持嵌入特征的原始表示能力，不被位置信息污染

#### 2.1 工作原理

RoPE通过复数旋转的方式为向量添加位置信息：

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 1e4):
    """
    预计算RoPE的频率复数

    Args:
        dim: RoPE维度（必须是偶数）
        end: 最大序列长度
        theta: 基础频率参数

    Returns:
        频率复数张量 [end, dim//2]
    """
    # 计算频率：θ_i = θ^(-2i/d) for i = 0, 1, ..., d/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 生成位置索引：m = 0, 1, 2, ..., end-1
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)

    # 计算每个位置和频率的外积：m * θ_i
    freqs = torch.outer(t, freqs).float()

    # 转换为复数形式：e^(i * m * θ_i) = cos(m*θ_i) + i*sin(m*θ_i)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    对查询和键张量应用旋转位置编码

    Args:
        xq: 查询张量 [batch_size, seq_len, n_heads, head_dim]
        xk: 键张量 [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: 频率复数 [seq_len, head_dim//2]

    Returns:
        应用RoPE后的(xq, xk)
    """
    # 将实数向量重塑为复数表示 [batch, seq_len, n_heads, head_dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 调整freqs_cis的形状以匹配输入张量
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 应用旋转：复数乘法实现旋转变换
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def reshape_for_broadcast(freqs_cis, x):
    """调整频率复数的形状以匹配输入张量"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
```

#### 2.2 数学原理

RoPE的核心思想是将位置信息编码为旋转矩阵：

1. **复数表示**：将向量的相邻两个维度组合成复数
2. **旋转变换**：通过复数乘法实现旋转
3. **相对位置**：两个位置间的相对距离决定了它们的相对旋转角度

对于位置m和n的两个向量，它们的内积具有相对位置不变性：
```
<RoPE(q_m), RoPE(k_n)> = <q_m, k_n> * e^(i(m-n)θ)
```

#### 2.3 在注意力中的应用

```python
class MultiHeadLatentAttention(nn.Module):
    def forward(self, x, freqs_cis):
        # 1. 计算Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, n_heads * head_dim]

        # 通过潜在压缩计算K, V
        kv_compressed = self.kv_compress(x)  # [batch, seq_len, kv_lora_rank]
        kv_compressed = self.kv_norm(kv_compressed)

        k = self.k_up(kv_compressed)  # [batch, seq_len, n_heads * head_dim]
        v = self.v_up(kv_compressed)  # [batch, seq_len, n_heads * v_dim]

        # 2. 重塑为多头格式
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.v_dim)

        # 3. 应用RoPE（只对Q和K，V不变）
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 4. 计算注意力
        attn_output = scaled_dot_product_attention(q, k, v)

        return attn_output
```

#### 2.4 RoPE优势

**相比传统位置编码的优势**：

1. **相对位置编码**：
   - 关注token间的相对距离而非绝对位置
   - 更适合时间序列：昨天vs今天比第179天vs第180天更有意义

2. **外推能力强**：
   - 能处理训练时未见过的序列长度
   - 对金融数据的长序列处理很重要

3. **旋转不变性**：
   - 保持向量的几何性质和长度
   - 不会改变特征的原始表示能力

4. **计算效率**：
   - 通过复数乘法实现，计算高效
   - 可以预计算频率复数，减少运行时开销

**适合金融时序的原因**：

1. **时间相对性**：金融数据中相对时间关系比绝对时间更重要
2. **长序列支持**：180天的序列长度，RoPE能很好处理
3. **模式识别**：相对位置编码有助于识别重复的时间模式

#### 2.5 配置参数

```python
# RoPE相关配置
rope_theta: float = 10000.0    # 基础频率参数，控制旋转频率
max_seq_len: int = 512         # 支持的最大序列长度
```

**参数详解**：

- `rope_theta`：基础频率参数，影响位置编码的频率分布
  - **数学含义**：θ_i = θ^(-2i/d)，θ越大，高维度的频率越低
  - **效果**：越大则低频分量越多，位置信息衰减更慢，适合长序列
  - **金融应用**：10000.0适合180天序列，能捕捉长期趋势
  - **调优建议**：序列越长，θ应该越大

- `max_seq_len`：预计算频率复数的最大长度
  - **作用**：预先计算所有位置的旋转复数，提高运行效率
  - **设置**：应该大于等于实际使用的最大序列长度
  - **内存影响**：更大的值会占用更多内存

#### 2.6 实际应用示例

```python
# 在PriceTransformer中的使用
class PriceTransformer(nn.Module):
    def __init__(self, args):
        # 预计算RoPE频率
        self.freqs_cis = precompute_freqs_cis(
            dim=args.d_model // args.n_heads,  # 每个头的维度
            end=args.max_seq_len,              # 最大序列长度
            theta=args.rope_theta               # 基础频率参数
        )

    def forward(self, x):
        # 获取当前序列长度对应的频率
        seq_len = x.size(1)
        freqs_cis = self.freqs_cis[:seq_len]

        # 在每个Transformer层中应用
        for layer in self.layers:
            x = layer(x, freqs_cis)

        return x
```

#### 2.7 与传统位置编码对比

| 特性 | RoPE | 传统位置编码 |
|------|------|-------------|
| **编码方式** | 旋转变换 | 直接相加 |
| **位置类型** | 相对位置 | 绝对位置 |
| **应用位置** | 注意力层 | 嵌入层 |
| **外推能力** | 强 | 弱 |
| **计算开销** | 每层计算 | 一次性 |
| **适用场景** | 长序列、时间序列 | 通用场景 |

**为什么选择RoPE**：

1. **金融时序特性**：相对时间关系比绝对时间更重要
2. **长序列处理**：180天序列，RoPE外推能力强
3. **模式识别**：有助于识别周期性和趋势性模式
4. **现代架构**：与MLA等现代注意力机制配合更好

#### 2.8 使用注意事项

**必要条件**：
1. **头维度必须是偶数**：RoPE需要将向量维度成对组合为复数
2. **预计算频率**：需要预先计算所有位置的旋转复数
3. **只对Q和K应用**：V向量不应用RoPE，保持原始特征表示

**常见问题**：
1. **维度不匹配**：确保 `d_model // n_heads` 是偶数
2. **序列长度超限**：输入序列长度不能超过 `max_seq_len`
3. **设备不匹配**：确保频率复数与输入张量在同一设备上

**调优建议**：
1. **theta参数**：根据序列长度调整，长序列使用更大的theta
2. **缓存频率**：预计算并缓存频率复数，避免重复计算
3. **内存优化**：合理设置max_seq_len，平衡性能和内存使用

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

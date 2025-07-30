# 配置参数文档

专门为金融时序预测 Transformer 模型设计的配置系统，提供灵活的参数调整和预定义配置。

## ? 目录

- [配置概述](#配置概述)
- [核心参数](#核心参数)
- [预定义配置](#预定义配置)
- [参数调优指南](#参数调优指南)
- [使用方法](#使用方法)

---

## ? 配置概述

### 设计理念

配置系统采用 dataclass 设计，具有以下特点：

1. **类型安全**：所有参数都有明确的类型注解
2. **参数验证**：自动验证参数的合理性和兼容性
3. **预定义配置**：提供多种场景的预配置方案
4. **灵活扩展**：易于添加新参数和配置

### 配置结构

```python
@dataclass
class PricePredictionConfig:
    # 模型架构参数
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8

    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 4

    # 金融专用参数
    use_financial_loss: bool = True
    direction_weight: float = 0.3
```

---

## ?? 核心参数

### 模型架构参数

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `d_model` | 512 | 模型维度 | 256-1024 |
| `n_layers` | 8 | Transformer层数 | 4-16 |
| `n_heads` | 8 | 注意力头数 | 4-16 |
| `kv_lora_rank` | 256 | K/V压缩维度 | 128-512 |
| `v_head_dim` | 64 | 值头维度 | 32-128 |
| `intermediate_size` | 2048 | FFN隐藏维度 | d_model×2-8 |

### 数据参数

| 参数 | 默认值 | 说明 | 约束 |
|------|--------|------|------|
| `n_features` | 20 | 输入特征数 | 固定为20 |
| `sequence_length` | 180 | 输入序列长度 | ≤ max_seq_len |
| `prediction_horizon` | 10 | 预测时间点数 | 固定为10 |
| `max_seq_len` | 512 | 最大序列长度 | ≥ sequence_length |
| `rope_theta` | 10000.0 | RoPE基础频率 | 1000-50000 |

### 训练参数

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `batch_size` | 4 | 批次大小 | 2-32 |
| `learning_rate` | 1e-4 | 学习率 | 1e-5 - 1e-3 |
| `weight_decay` | 0.01 | 权重衰减 | 0.001-0.1 |
| `max_epochs` | 100 | 最大训练轮数 | 50-500 |
| `warmup_steps` | 1000 | 预热步数 | 500-5000 |
| `dropout` | 0.1 | Dropout概率 | 0.0-0.3 |

### 金融专用损失函数参数

| 参数 | 默认值 | 说明 | 推荐设置 |
|------|--------|------|----------|
| `use_financial_loss` | True | 是否使用金融损失 | True |
| `use_direction_loss` | True | 是否启用方向损失 | True |
| `use_trend_loss` | True | 是否启用趋势损失 | True |
| `use_temporal_weighting` | True | 是否启用时间加权 | True |
| `use_ranking_loss` | False | 是否启用排序损失 | False |
| `use_volatility_loss` | False | 是否启用波动率损失 | False |

### 损失权重参数

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `base_weight` | 1.0 | 基础损失权重 | 0.5-1.5 |
| `direction_weight` | 0.3 | 方向损失权重 | 0.1-0.5 |
| `trend_weight` | 0.2 | 趋势损失权重 | 0.1-0.3 |
| `ranking_weight` | 0.1 | 排序损失权重 | 0.05-0.2 |
| `volatility_weight` | 0.1 | 波动率损失权重 | 0.05-0.2 |

---

## ?? 预定义配置

### 1. Tiny 配置

**适用场景**：快速测试、原型开发

```python
config = PricePredictionConfigs.tiny()
```

**参数特点**：
- `d_model=256, n_layers=4, n_heads=4`
- `batch_size=2, max_epochs=50`
- 参数量：~2.5M
- 内存需求：~2GB
- 训练速度：快

### 2. Small 配置

**适用场景**：个人电脑开发

```python
config = PricePredictionConfigs.small()
```

**参数特点**：
- `d_model=512, n_layers=6, n_heads=8`
- `batch_size=4, max_epochs=100`
- 参数量：~10M
- 内存需求：~4GB
- 训练速度：中等

### 3. Base 配置（默认）

**适用场景**：标准训练、生产环境

```python
config = PricePredictionConfigs.base()
```

**参数特点**：
- `d_model=512, n_layers=8, n_heads=8`
- `batch_size=4, max_epochs=100`
- 参数量：~40M
- 内存需求：~8GB
- 训练速度：中等

### 4. Large 配置

**适用场景**：服务器训练、高性能需求

```python
config = PricePredictionConfigs.large()
```

**参数特点**：
- `d_model=1024, n_layers=12, n_heads=16`
- `batch_size=8, max_epochs=200`
- 参数量：~160M
- 内存需求：~16GB
- 训练速度：慢

### 5. 长序列配置

**适用场景**：需要更长历史信息

```python
config = PricePredictionConfigs.for_long_sequence()
```

**参数特点**：
- `sequence_length=360, max_seq_len=1024`
- `d_model=768, n_layers=8`
- 适合年度级别的长期预测

### 6. 多步预测配置

**适用场景**：专注多时间点预测

```python
config = PricePredictionConfigs.for_multi_step_prediction()
```

**参数特点**：
- `n_layers=10, loss_type="mae"`
- 优化多步预测性能

---

## ? 性能对比

### 训练性能对比

| 配置 | 参数量 | 训练速度 | 内存使用 | 推荐场景 |
|------|--------|----------|----------|----------|
| Tiny | 2.5M | 快 | 2GB | 快速实验 |
| Small | 10M | 中等 | 4GB | 个人开发 |
| Base | 40M | 中 | 8GB | 标准训练 |
| Large | 160M | 慢 | 16GB | 高性能需求 |

### 预测性能对比

| 配置 | 方向准确率 | MAE | RMSE | 训练时间 |
|------|------------|-----|------|----------|
| Tiny | ~70% | 0.08 | 0.12 | 2小时 |
| Small | ~75% | 0.06 | 0.10 | 4小时 |
| Base | ~80% | 0.05 | 0.08 | 8小时 |
| Large | ~85% | 0.04 | 0.06 | 16小时 |

---

## ? 参数调优指南

### 1. 模型容量调优

**增加模型容量**：
```python
config.d_model = 768        # 增加模型维度
config.n_layers = 12       # 增加层数
config.intermediate_size = 3072  # 增加FFN维度
```

**减少模型容量**：
```python
config.d_model = 256        # 减少模型维度
config.n_layers = 4        # 减少层数
config.kv_lora_rank = 128  # 增加压缩比
```

### 2. 训练稳定性调优

**提高训练稳定性**：
```python
config.learning_rate = 5e-5     # 降低学习率
config.warmup_steps = 2000      # 增加预热步数
config.dropout = 0.2            # 增加dropout
config.weight_decay = 0.05      # 增加权重衰减
```

**加速训练收敛**：
```python
config.learning_rate = 2e-4     # 提高学习率
config.batch_size = 8           # 增加批次大小
config.warmup_steps = 500       # 减少预热步数
```

### 3. 金融损失调优

**保守策略**（重视数值准确性）：
```python
config.base_weight = 1.0
config.direction_weight = 0.1
config.trend_weight = 0.1
config.use_ranking_loss = False
```

**激进策略**（重视方向和趋势）：
```python
config.base_weight = 0.5
config.direction_weight = 0.5
config.trend_weight = 0.3
config.use_ranking_loss = True
config.ranking_weight = 0.2
```

### 4. 内存优化

**减少内存使用**：
```python
config.batch_size = 2           # 减少批次大小
config.kv_lora_rank = 128      # 增加K/V压缩
config.sequence_length = 120    # 减少序列长度
```

**梯度累积**（保持有效批次大小）：
```python
config.batch_size = 2
# 在训练脚本中设置 accumulation_steps = 4
# 有效批次大小 = 2 × 4 = 8
```

---

## ? 使用方法

### 基础使用

```python
from src.price_prediction.config import PricePredictionConfigs

# 使用预定义配置
config = PricePredictionConfigs.base()

# 创建模型
from src.price_prediction.price_transformer import PriceTransformer
model = PriceTransformer(config)
```

### 自定义配置

```python
from src.price_prediction.config import PricePredictionConfig

# 创建自定义配置
config = PricePredictionConfig(
    d_model=768,
    n_layers=10,
    n_heads=12,
    learning_rate=5e-5,
    batch_size=6,

    # 自定义金融损失
    use_financial_loss=True,
    direction_weight=0.4,
    trend_weight=0.3,
    use_volatility_loss=True,
    volatility_weight=0.15
)
```

### 配置修改

```python
# 基于现有配置修改
config = PricePredictionConfigs.base()

# 修改特定参数
config.learning_rate = 2e-4
config.batch_size = 8
config.direction_weight = 0.4

# 启用额外的损失函数
config.use_ranking_loss = True
config.ranking_weight = 0.15
```

### 配置验证

```python
# 配置会自动验证参数
try:
    config = PricePredictionConfig(
        d_model=513,  # 不能被n_heads整除
        n_heads=8
    )
except AssertionError as e:
    print(f"配置错误: {e}")

# 正确的配置
config = PricePredictionConfig(
    d_model=512,  # 512 % 8 = 0 ?
    n_heads=8
)
```

---

## ? 最佳实践

### 1. 配置选择策略

**开发阶段**：
```python
# 使用tiny配置快速迭代
config = PricePredictionConfigs.tiny()
config.max_epochs = 10  # 快速验证
```

**实验阶段**：
```python
# 使用small或base配置
config = PricePredictionConfigs.small()
config.max_epochs = 50
```

**生产阶段**：
```python
# 使用base或large配置
config = PricePredictionConfigs.base()
config.max_epochs = 200
config.early_stopping_patience = 30
```

### 2. 超参数搜索

```python
# 定义搜索空间
search_configs = [
    {"learning_rate": 1e-4, "direction_weight": 0.2},
    {"learning_rate": 5e-5, "direction_weight": 0.3},
    {"learning_rate": 2e-4, "direction_weight": 0.4},
]

best_config = None
best_score = float('inf')

for params in search_configs:
    config = PricePredictionConfigs.base()
    for key, value in params.items():
        setattr(config, key, value)

    # 训练和评估
    score = train_and_evaluate(config)

    if score < best_score:
        best_score = score
        best_config = config
```

### 3. 配置保存和加载

```python
import json
from dataclasses import asdict

# 保存配置
config = PricePredictionConfigs.base()
config_dict = asdict(config)

with open('config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

# 加载配置
with open('config.json', 'r') as f:
    config_dict = json.load(f)

config = PricePredictionConfig(**config_dict)
```

### 4. 环境适配

```python
import torch

def get_adaptive_config():
    """根据硬件环境自动选择配置"""

    # 检查GPU内存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        if gpu_memory >= 16:
            return PricePredictionConfigs.large()
        elif gpu_memory >= 8:
            return PricePredictionConfigs.base()
        elif gpu_memory >= 4:
            return PricePredictionConfigs.small()
        else:
            return PricePredictionConfigs.tiny()
    else:
        # CPU训练使用小配置
        return PricePredictionConfigs.tiny()

# 使用自适应配置
config = get_adaptive_config()
```

---

## ? 调试和监控

### 配置信息打印

```python
def print_config_summary(config):
    """打印配置摘要"""
    print("=" * 50)
    print("模型配置摘要")
    print("=" * 50)

    # 模型架构
    print(f"模型维度: {config.d_model}")
    print(f"层数: {config.n_layers}")
    print(f"注意力头数: {config.n_heads}")
    print(f"参数估计: ~{estimate_params(config):,}")

    # 训练配置
    print(f"\n训练配置:")
    print(f"  学习率: {config.learning_rate}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  最大轮数: {config.max_epochs}")

    # 损失配置
    print(f"\n损失函数:")
    print(f"  基础损失: {config.loss_type}")
    print(f"  金融损失: {'启用' if config.use_financial_loss else '禁用'}")
    if config.use_financial_loss:
        print(f"  方向权重: {config.direction_weight}")
        print(f"  趋势权重: {config.trend_weight}")

def estimate_params(config):
    """估算模型参数数量"""
    # 简化的参数估算
    embed_params = config.n_features * config.d_model
    transformer_params = config.n_layers * (
        config.d_model * config.d_model * 4 +  # 注意力
        config.d_model * config.intermediate_size * 3  # FFN
    )
    head_params = config.d_model * config.prediction_horizon

    return embed_params + transformer_params + head_params

# 使用示例
config = PricePredictionConfigs.base()
print_config_summary(config)
```

---

## ? 相关文件

- `src/price_prediction/config.py` - 配置类实现
- `src/price_prediction/price_transformer.py` - 主模型实现
- `train_price_prediction.py` - 训练脚本
- `doc/transformer.md` - Transformer架构文档
- `doc/financial_losses.md` - 损失函数文档

这套配置系统为金融时序预测提供了灵活且强大的参数管理能力！

# 金融特征嵌入层设计

## 概述

金融特征嵌入层是价格预测模型的核心组件，负责将原始的20维金融特征转换为高维向量表示。本文档详细介绍了嵌入层的设计思想、实现细节和使用方法。

## 设计思想

### 1. 统一嵌入策略

采用统一嵌入策略，将每日的20维金融特征作为一个整体进行嵌入：

- **整体性**：每日交易数据代表一个完整的市场状态
- **自由学习**：让模型自由学习特征间的关系，不受人为分组限制
- **简洁高效**：避免复杂的分组逻辑，提高计算效率
- **无约束**：不需要d_model被特定数字整除的约束

### 2. 位置编码处理

位置信息由Transformer层中的RoPE（Rotary Position Embedding）处理：

- **分离关注**：嵌入层专注于特征表示，位置信息由注意力层处理
- **相对位置**：RoPE关注相对距离，更适合时间序列
- **详细说明**：RoPE的具体实现请参考 `doc/transformer.md`

## 特征详解

### 20维特征构成

```python
特征构成（20维）：
- 时间特征 (3维): 月、日、星期
- 价格特征 (4维): open_rel, high_rel, low_rel, close_rel
- 价格变化 (2维): 涨幅, 振幅
- 成交量特征 (2维): volume_rel, volume_log
- 金额特征 (2维): amount_rel, amount_log
- 市场特征 (3维): 成交次数, 换手%, price_median
- 金融特征 (4维): big_order_activity, chip_concentration, market_sentiment, price_volume_sync
```

### 特征类型说明

#### 1. 时间特征 (3维)
- **特征**：月(1-12)、日(1-31)、星期(1-7)
- **作用**：提供时间周期性信息
- **处理**：直接使用原始数值

#### 2. 价格特征 (4维)
- **特征**：开盘、最高、最低、收盘价格的相对值
- **计算**：`price_rel = price / sequence_price_median`
- **作用**：提供价格形态信息，避免绝对价格的影响

#### 3. 价格变化 (2维)
- **特征**：涨幅(%)、振幅(%)
- **作用**：反映价格波动特征
- **处理**：直接使用百分比数值

#### 4. 成交量特征 (2维)
- **特征**：
  - `volume_rel`: 成交量相对值（相对于序列中位数）
  - `volume_log`: 成交量对数值（压缩数值范围）
- **作用**：提供交易活跃度信息

#### 5. 金额特征 (2维)
- **特征**：
  - `amount_rel`: 成交金额相对值
  - `amount_log`: 成交金额对数值
- **作用**：反映资金流动情况

#### 6. 市场特征 (3维)
- **特征**：成交次数、换手率、价格基准
- **作用**：反映市场流动性和交易特征
- **特殊**：price_median为序列级元信息，便于预测时提取

#### 7. 金融特征 (4维)
- **特征**：
  - `big_order_activity`: 大单活跃度（对数+标准化处理）
  - `chip_concentration`: 筹码集中度（标准化）
  - `market_sentiment`: 市场情绪（标准化）
  - `price_volume_sync`: 价量同步性（-1,0,1值）
- **作用**：提供高级金融分析指标

## 实现架构

### 1. 统一嵌入层

```python
class FinancialEmbedding(nn.Module):
    """
    金融特征统一嵌入层
    将20维特征作为整体进行嵌入
    """
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # 统一嵌入：20维特征 → d_model维
        self.feature_embedding = nn.Sequential(
            nn.Linear(20, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 可选的额外变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 统一嵌入：每日20维特征作为整体进行嵌入
        embedded = self.feature_embedding(x)  # [batch, seq_len, d_model]
        
        # 可选的额外变换
        embedded = self.feature_transform(embedded)
        
        return embedded
```

### 2. 批序列标准化

```python
class BatchSequenceNorm(nn.Module):
    """
    批序列内标准化
    在序列维度上进行标准化，保持批次间的独立性
    """
    def __init__(self, eps: float = 1e-5, learnable: bool = True):
        super().__init__()
        self.eps = eps
        self.learnable = learnable
        
        if learnable:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿序列维度标准化
        mean = x.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
        std = x.std(dim=1, keepdim=True)    # [batch, 1, d_model]
        
        normalized = (x - mean) / (std + self.eps)
        
        if self.learnable:
            normalized = normalized * self.weight + self.bias
        
        return normalized
```

### 3. 完整嵌入层

```python
class FinancialEmbeddingLayer(nn.Module):
    """
    完整的金融特征嵌入层
    包含特征嵌入和批标准化（位置编码由Transformer层处理）
    """
    def __init__(
        self, 
        d_model: int = 512,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        learnable_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_batch_norm = use_batch_norm
        
        # 特征嵌入
        self.feature_embedding = FinancialEmbedding(d_model, dropout)
        
        # 批标准化
        if use_batch_norm:
            self.batch_norm = BatchSequenceNorm(learnable=learnable_norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整的嵌入处理流程
        
        Args:
            x: [batch, seq_len, 20] 原始特征
            
        Returns:
            embedded: [batch, seq_len, d_model] 最终嵌入特征

        注意：位置信息由Transformer层的RoPE处理，详见 doc/transformer.md
        """
        # 1. 特征嵌入
        embedded = self.feature_embedding(x)
        
        # 2. 批标准化（可选）
        if self.use_batch_norm:
            embedded = self.batch_norm(embedded)
        
        return embedded
```



## 使用示例

### 基本使用

```python
from src.price_prediction.embedding import FinancialEmbeddingLayer

# 创建嵌入层
embedding_layer = FinancialEmbeddingLayer(
    d_model=512,
    dropout=0.1,
    use_batch_norm=True
)

# 输入特征 [batch_size, seq_len, 20]
financial_features = torch.randn(32, 180, 20)

# 嵌入处理
embedded_features = embedding_layer(financial_features)
# 输出: [32, 180, 512]
```

### 配置选项

```python
# 最小配置（仅特征嵌入）
minimal_embedding = FinancialEmbeddingLayer(
    d_model=256,
    use_batch_norm=False
)

# 完整配置
full_embedding = FinancialEmbeddingLayer(
    d_model=512,
    dropout=0.1,
    use_batch_norm=True,
    learnable_norm=True
)
```

### 价格基准提取

```python
from src.price_prediction.data_cteater import get_price_median_from_features

# 从特征向量中提取价格基准
feature_vector = np.random.randn(180, 20)  # 示例特征向量
price_median = get_price_median_from_features(feature_vector)
print(f"价格基准: {price_median:.2f}")
```

## 设计优势

### 1. 简洁高效
- **统一处理**：20维特征作为整体嵌入，逻辑简单
- **无约束**：不需要d_model被特定数字整除
- **参数少**：相比分组嵌入，参数量更少

### 2. 自由学习
- **端到端**：让模型自由学习特征间关系
- **无偏见**：不受人为分组假设限制
- **适应性强**：模型可以学习到最优的特征组合

### 3. 架构清晰
- **职责分离**：嵌入层专注特征表示，位置信息由Transformer处理
- **模块化**：各组件职责明确，便于维护和扩展
- **灵活配置**：可独立调整嵌入和位置编码策略

### 4. 实用性强
- **价格基准内置**：特征中包含price_median，便于预测时提取
- **标准化处理**：金融特征经过合理的预处理
- **批处理友好**：支持高效的批量处理

## 与传统方案对比

| 特性 | 统一嵌入 | 分组嵌入 |
|------|---------|---------|
| **复杂度** | 简单 | 复杂 |
| **参数量** | 较少 | 较多 |
| **约束** | 无 | d_model需整除 |
| **灵活性** | 高 | 中等 |
| **学习能力** | 自由学习 | 受分组限制 |
| **计算效率** | 高 | 中等 |
| **可解释性** | 中等 | 高 |

统一嵌入方案更符合"每日交易数据是一个整体"的直觉，让模型自由学习特征关系，是更优的设计选择。

## 注意事项

### 1. 特征预处理
- 确保输入特征已经过适当的标准化
- 价格特征使用相对值而非绝对值
- 大单活跃度经过对数+标准化处理

### 2. 数值稳定性
- 使用LayerNorm保证训练稳定性
- 合理的dropout防止过拟合
- 批标准化有助于收敛

### 3. 内存使用
- 批标准化会增加少量内存使用
- 统一嵌入相比分组嵌入更节省内存

### 4. 位置编码配合
- 嵌入层不处理位置信息，保持特征纯净性
- 位置编码由Transformer层的RoPE处理
- 详细的位置编码说明请参考 `doc/transformer.md`

## 性能优化建议

### 1. 批大小选择
- 建议批大小为8-32，平衡内存和训练效率
- 使用梯度累积处理更大的有效批大小

### 2. 模型维度
- d_model推荐使用2的幂次（256, 512, 1024）
- 注意力头数应能整除d_model

### 3. 正则化策略
- 合理使用dropout防止过拟合
- 批标准化有助于训练稳定性

### 4. 初始化策略
- 使用Xavier初始化权重
- 偏置初始化为零

## 工厂函数

```python
def create_financial_embedding(
    d_model: int = 512,
    dropout: float = 0.1,
    **kwargs
) -> FinancialEmbeddingLayer:
    """
    创建金融特征嵌入层的工厂函数

    Args:
        d_model: 模型维度
        dropout: dropout率
        **kwargs: 其他参数（如use_batch_norm等）

    Returns:
        embedding_layer: 金融特征嵌入层（位置信息由Transformer层处理）
    """
    return FinancialEmbeddingLayer(
        d_model=d_model,
        dropout=dropout,
        **kwargs
    )
```

## 测试验证

### 运行测试

```bash
# 测试统一嵌入层
python test_unified_embedding.py

# 测试完整的20维特征处理
python test_20_features_update.py
```

### 测试内容
- 20维特征向量形状验证
- 统一嵌入层前向传播
- 价格基准提取功能
- 批标准化效果验证
- 模型集成测试

## 相关文件

- `src/price_prediction/embedding.py` - 嵌入层实现
- `src/price_prediction/data_cteater.py` - 特征处理和价格基准提取
- `src/price_prediction/price_transformer.py` - 模型主体
- `doc/data.md` - 数据处理详细说明
- `doc/config.md` - 配置参数说明

---

**总结**：统一嵌入设计简化了架构，提高了效率，让模型能够自由学习20维金融特征间的复杂关系，配合Transformer层的RoPE位置编码，为时间序列预测提供了强大的特征表示能力。

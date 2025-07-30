# 金融时序预测专用损失函数

专门为股票价格预测任务设计的多损失函数组合，通过结合多种损失函数来全面优化预测效果。

## 📋 目录

- [设计理念](#设计理念)
- [损失函数组件](#损失函数组件)
- [使用方法](#使用方法)
- [配置参数](#配置参数)
- [实验效果](#实验效果)

---

## 🎯 设计理念

### 股票预测的多重目标

股票价格预测不仅仅是数值回归问题，还包含多个重要目标：

1. **数值准确性**：预测价格要接近真实值
2. **方向正确性**：预测涨跌方向要准确（比数值更重要）
3. **趋势一致性**：预测的价格变化趋势要合理
4. **时间重要性**：近期预测比远期预测更重要
5. **相对关系**：不同时间点之间的相对大小关系
6. **波动模式**：价格波动的模式要符合市场规律

### 单一损失函数的局限性

- **MSE损失**：只关注数值差异，忽略方向和趋势
- **MAE损失**：对异常值不敏感，但同样忽略金融特性
- **方向准确率**：只看方向，不考虑数值大小

---

## 🧩 损失函数组件

### 1. DirectionLoss - 方向损失

**目标**：确保预测涨跌方向的准确性

```python
class DirectionLoss(nn.Module):
    def __init__(self, weight_by_magnitude: bool = True):
        # weight_by_magnitude: 是否根据变化幅度加权
```

**计算方式**：
- 计算相邻时间点的价格变化方向
- 比较预测方向与真实方向
- 可选：根据变化幅度加权（大幅变化的方向错误惩罚更重）

**适用场景**：方向比数值更重要的交易策略

### 2. TrendLoss - 趋势损失

**目标**：确保整体价格变化趋势的一致性

```python
class TrendLoss(nn.Module):
    def __init__(self, order: int = 2):
        # order: 差分阶数，1=速度，2=加速度
```

**计算方式**：
- 一阶差分：价格变化速度的一致性
- 二阶差分：价格变化加速度的一致性（推荐）

**适用场景**：需要捕捉价格变化模式的长期预测

### 3. RankingLoss - 排序损失

**目标**：保持预测值之间的相对大小关系

```python
class RankingLoss(nn.Module):
    def __init__(self, margin: float = 1.0, threshold: float = 0.01):
        # margin: hinge loss的边界
        # threshold: 认为有显著差异的最小阈值
```

**计算方式**：
- 对所有时间点配对比较
- 使用hinge loss惩罚相对关系错误
- 只对显著差异的配对进行约束

**适用场景**：相对排序比绝对数值更重要的场景

### 4. VolatilityLoss - 波动率损失

**目标**：确保预测的价格波动模式与真实波动一致

```python
class VolatilityLoss(nn.Module):
    def __init__(self, window_size: int = 3):
        # window_size: 计算波动率的滑动窗口大小
```

**计算方式**：
- 在滑动窗口内计算标准差（波动率）
- 比较预测波动率与真实波动率
- 确保模型学习到正确的波动模式

**适用场景**：波动率预测、风险管理

### 5. TemporalWeightedLoss - 时间加权损失

**目标**：近期预测比远期预测更重要

```python
class TemporalWeightedLoss(nn.Module):
    def __init__(self, base_loss_fn: nn.Module, decay_factor: float = 0.9):
        # base_loss_fn: 基础损失函数
        # decay_factor: 时间衰减因子
```

**权重设计**：
```python
# 专门为10个时间点设计的权重
weights = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
#         第1天 第2天 第3天 第4天 第5天 第10天 第15天 第20天 第25天 第30天
```

**适用场景**：短期交易策略，近期预测更关键

---

## 🚀 使用方法

### 基础使用

```python
from src.price_prediction.financial_losses import FinancialMultiLoss

# 创建多损失函数
criterion = FinancialMultiLoss(
    base_loss_type='mse',           # 基础损失类型
    use_direction_loss=True,        # 启用方向损失
    use_trend_loss=True,            # 启用趋势损失
    use_temporal_weighting=True,    # 启用时间加权
    use_ranking_loss=False,         # 禁用排序损失
    use_volatility_loss=False,      # 禁用波动率损失
    
    # 损失权重
    base_weight=1.0,
    direction_weight=0.3,
    trend_weight=0.2
)

# 训练中使用
predictions = model(inputs)  # [batch_size, 10]
targets = batch_targets      # [batch_size, 10]

loss_dict = criterion(predictions, targets)
total_loss = loss_dict['loss']  # 用于反向传播

# 监控各组件损失
print(f"总损失: {total_loss:.4f}")
print(f"基础损失: {loss_dict['base_loss']:.4f}")
print(f"方向损失: {loss_dict['direction_loss']:.4f}")
print(f"趋势损失: {loss_dict['trend_loss']:.4f}")
print(f"方向准确率: {loss_dict['direction_accuracy']:.2%}")
```

### 高级配置

```python
# 保守配置：主要关注数值准确性
conservative_loss = FinancialMultiLoss(
    base_loss_type='huber',         # 对异常值鲁棒
    use_direction_loss=True,
    use_trend_loss=False,
    use_temporal_weighting=True,
    base_weight=1.0,
    direction_weight=0.1            # 较小的方向权重
)

# 激进配置：重视方向和趋势
aggressive_loss = FinancialMultiLoss(
    base_loss_type='mae',
    use_direction_loss=True,
    use_trend_loss=True,
    use_ranking_loss=True,
    use_volatility_loss=True,
    base_weight=0.5,                # 降低基础损失权重
    direction_weight=0.4,           # 提高方向权重
    trend_weight=0.3,
    ranking_weight=0.2,
    volatility_weight=0.1
)

# 短期交易配置：重视近期预测
short_term_loss = FinancialMultiLoss(
    base_loss_type='mse',
    use_temporal_weighting=True,    # 强调时间权重
    use_direction_loss=True,
    base_weight=1.0,
    direction_weight=0.5            # 方向非常重要
)
```

---

## ⚙️ 配置参数

### 损失函数开关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_direction_loss` | `True` | 是否使用方向损失 |
| `use_trend_loss` | `True` | 是否使用趋势损失 |
| `use_temporal_weighting` | `True` | 是否使用时间加权 |
| `use_ranking_loss` | `False` | 是否使用排序损失 |
| `use_volatility_loss` | `False` | 是否使用波动率损失 |

### 损失权重

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `base_weight` | `1.0` | `0.5-1.5` | 基础回归损失权重 |
| `direction_weight` | `0.3` | `0.1-0.5` | 方向损失权重 |
| `trend_weight` | `0.2` | `0.1-0.3` | 趋势损失权重 |
| `ranking_weight` | `0.1` | `0.05-0.2` | 排序损失权重 |
| `volatility_weight` | `0.1` | `0.05-0.2` | 波动率损失权重 |

### 调参建议

1. **保守策略**：`base_weight=1.0, direction_weight=0.1-0.2`
2. **平衡策略**：`base_weight=1.0, direction_weight=0.3, trend_weight=0.2`
3. **激进策略**：`base_weight=0.5, direction_weight=0.4-0.5`

---

## 📊 实验效果

### 预期改进

1. **方向准确率提升**：从60-70%提升到75-85%
2. **趋势一致性**：减少预测中的趋势反转错误
3. **近期预测精度**：前5天预测误差降低20-30%
4. **整体稳定性**：减少极端预测值

### 使用建议

1. **开始训练**：使用默认配置
2. **观察指标**：重点关注`direction_accuracy`
3. **调整权重**：根据业务需求调整各损失权重
4. **A/B测试**：对比单一损失和多损失的效果

### 注意事项

- 多损失函数会增加训练复杂度
- 需要更仔细的超参数调优
- 建议先在小数据集上验证效果
- 不同市场环境可能需要不同的权重配置

---

## 🔗 相关文件

- `src/price_prediction/financial_losses.py` - 损失函数实现
- `src/price_prediction/price_transformer.py` - 模型实现
- `train_price_prediction.py` - 训练脚本

这套多损失函数组合专门为金融时序预测设计，能够从多个维度优化模型性能，特别适合股票价格预测任务！

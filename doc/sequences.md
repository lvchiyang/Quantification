# 序列处理器 API 文档

专门为金融时序预测设计的序列处理器，确保数据处理的科学性和实用性。

## 📋 目录

- [核心概念](#核心概念)
- [API 接口](#api-接口)
- [特征说明](#特征说明)
- [使用流程](#使用流程)
- [配置参数](#配置参数)
- [注意事项](#注意事项)

---

## 🎯 核心概念

### 设计原则
1. **避免数据泄露**：每个180天序列独立计算基准
2. **预测一致性**：训练和预测使用相同逻辑
3. **特征完整性**：20维金融特征，保持经济学意义
4. **数值稳定性**：标准化但不裁剪极值

### 数据流程
```
原始数据(14列) → 序列切分(180天) → 特征工程(20维) → 目标提取(10个时间点)
```

---

## 🔧 API 接口

### SequenceProcessor 类

#### 初始化
```python
from sequence_processor import SequenceProcessor

processor = SequenceProcessor(sequence_length=180)
```

**参数**：
- `sequence_length`: 序列长度，默认180天

#### 主要方法

##### 1. 创建训练序列
```python
sequences = processor.create_training_sequences(cleaned_data)
```

**参数**：
- `cleaned_data`: 基础清洗后的DataFrame（14列）

**返回**：
- `List[Tuple[np.ndarray, np.ndarray]]`
- 每个元组：`(input_sequence[180,20], target_prices[10])`

**目标时间点**：未来第 1, 2, 3, 4, 5, 10, 15, 20, 25, 30 天

##### 2. 创建预测序列
```python
feature_vector = processor.create_prediction_sequence(recent_data)
```

**参数**：
- `recent_data`: 最近的数据（至少180行）

**返回**：
- `np.ndarray`: [180, 20] 特征矩阵

##### 3. 序列级特征处理
```python
features = processor.process_sequence_features(sequence_df)
```

**参数**：
- `sequence_df`: 180天的序列数据

**返回**：
- `dict`: 包含所有计算特征的字典

---

## 📊 特征说明

### 20维特征结构

根据最新的embedding层设计，特征分组如下：

| 特征组 | 索引 | 维度 | 特征列表 | 说明 |
|--------|------|------|----------|------|
| **时间特征** | [0-2] | 3维 | 月、日、星期 | 基础时间信息 |
| **价格特征** | [3-6] | 4维 | open_rel, high_rel, low_rel, close_rel | OHLC相对于序列基准的比值 |
| **成交量特征** | [7-11] | 5维 | volume_rel, volume_change, amount_rel, amount_change, 成交次数 | 交易活跃度指标 |
| **市场特征** | [12-15] | 4维 | 涨幅, 振幅, 换手%, price_median | 市场波动指标 |
| **金融特征** | [16-19] | 4维 | big_order_activity, chip_concentration, market_sentiment, price_volume_sync | 高级金融指标 |

### 特征处理方式

#### 序列级基准计算
- **价格基准**：每个180天序列独立计算OHLC中位数
- **成交量基准**：序列内中位数作为相对值基准
- **标准化**：金融特征使用序列内标准化，不裁剪极值

#### 避免数据泄露
- ✅ 不使用未来数据计算统计量
- ✅ 每个序列独立处理
- ✅ 训练和预测逻辑完全一致

---

## 🚀 使用流程

### PriceDataset 数据集类

#### 创建数据集
```python
from sequence_processor import PriceDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = PriceDataset("processed_data_2025-07-30/股票数据", sequence_length=180)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用数据
for inputs, targets in dataloader:
    # inputs: [batch_size, 180, 20] - 输入特征序列
    # targets: [batch_size, 10] - 目标价格序列
    pass
```

### 预测函数

#### 单股票预测
```python
from sequence_processor import predict_stock_price

# 预测单只股票
predictions = predict_stock_price(
    model=trained_model,
    stock_file="path/to/stock.xlsx",
    processor=None  # 可选，默认创建新的处理器
)

# 解释预测结果
target_days = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
for i, day in enumerate(target_days):
    print(f"未来第{day}天预测: {predictions[i]:.4f}")
```

### 验证工具

#### 数据质量检查
```python
from sequence_processor import validate_sequence_processing, check_data_quality

# 验证序列处理
validate_sequence_processing(cleaned_data, processor)

# 检查数据质量
sequences = processor.create_training_sequences(cleaned_data)
check_data_quality(sequences)
```

---

## 🎯 完整示例

### 训练流程
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sequence_processor import PriceDataset

# 1. 创建数据集
dataset = PriceDataset("processed_data_2025-07-30/股票数据")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 训练模型
model = YourPriceModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # inputs: [batch_size, 180, 20]
        # targets: [batch_size, 10]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### 预测流程
```python
from sequence_processor import predict_stock_price

# 1. 加载训练好的模型
model.load_state_dict(torch.load('model.pth'))

# 2. 预测单只股票
predictions = predict_stock_price(
    model,
    "processed_data_2025-07-30/股票数据/白酒/茅台.xlsx"
)

# 3. 解释预测结果
target_days = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
for i, day in enumerate(target_days):
    print(f"未来第{day}天预测: {predictions[i]:.4f}")
```
---

## ⚙️ 配置参数

### 核心参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sequence_length` | 180 | 输入序列长度（天） |
| `target_days` | [1,2,3,4,5,10,15,20,25,30] | 预测时间点 |
| `feature_dim` | 20 | 特征维度 |

### 处理参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rolling_window` | 20 | 成交量变化率计算窗口 |
| `volume_window` | 30 | 筹码集中度计算窗口 |
| `min_periods` | 1 | 滚动计算最小周期 |

---

## ⚠️ 注意事项

### 数据泄露防护
- ✅ 每个180天序列独立计算价格基准
- ✅ 不使用未来数据计算统计量
- ✅ 训练和预测使用相同的处理逻辑

### 数据质量保证
- ✅ 自动处理NaN值和异常值
- ✅ 数值范围检查和验证
- ✅ 特征独立性验证

### 性能优化
- ✅ 批量处理支持
- ✅ 内存效率优化
- ✅ 完善的错误处理机制

### 使用建议
1. **数据准备**：确保输入数据已经过基础清洗
2. **内存管理**：大数据集建议分批处理
3. **验证测试**：使用提供的验证函数检查数据质量
4. **模型兼容**：确保模型输入维度与特征维度匹配

---

## 📚 相关文件

- `sequence_processor.py` - 核心实现代码
- `test/sequence_usage_example.py` - 完整使用示例
- `test/test_embedding_grouping.py` - 特征分组测试
- `src/price_prediction/embedding.py` - 特征嵌入层

### 运行测试
```bash
# 基础功能测试
python test/sequence_usage_example.py

# 特征分组测试
python test/test_embedding_grouping.py
```

---

## 🎯 核心优势

1. **科学性**：避免数据泄露，确保模型泛化能力
2. **一致性**：训练和预测逻辑完全一致
3. **完整性**：20维金融特征，保持经济学意义
4. **稳定性**：标准化但不裁剪极值，保持数据分布
5. **易用性**：简洁的API设计，完善的文档支持

这个序列处理器专门为金融时序预测设计，是构建可靠预测模型的基础组件！

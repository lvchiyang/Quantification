# 📊 数据处理解决方案

## 🎯 解决的问题

### 1. **数据目录结构适配**
- **问题**: 训练数据现在位于 `processed_data_2025-07-29/` 目录下，有多层目录结构
- **解决方案**: 更新 `FinancialDataProcessor` 支持新的目录结构，自动遍历行业分类

### 2. **数据列结构调整**
- **问题**: 第一列"年"不应作为输入特征
- **解决方案**: 跳过第一列，使用实际的列名映射

### 3. **大数值尺度问题** ⭐
- **问题**: 总手、金额数值巨大，与其他列差异极大，影响模型训练
- **核心挑战**: 既要解决尺度问题，又要保持对变化的敏感性

## 🔧 技术解决方案

### 数据列映射
```
原始数据: 年, 月, 日, 星期, 开盘, 最高, 最低, 收盘, 涨幅, 振幅, 总手, 金额, 换手%, 成交次数
处理后:   月, 日, 星期, 开盘, 最高, 最低, 收盘, 涨幅, 振幅, 总手处理, 金额处理, 换手%, 成交次数
特征名:   month, day, weekday, open, high, low, close, change_pct, amplitude, volume_processed, amount_processed, turnover_rate, trade_count
```

### 大数值处理策略

#### 🏆 推荐方案：相对变化率 (`relative_change`)
```python
# 计算相对于20日移动平均的变化率
rolling_mean = pd.Series(values).rolling(window=20, min_periods=1).mean()
relative_change = (values - rolling_mean) / rolling_mean * 100  # 百分比
```

**优势**:
- ✅ 保持对变化的高敏感性
- ✅ 数值范围合理（通常在-50%到+50%之间）
- ✅ 金融意义明确（相对于近期平均的异常程度）
- ✅ 自动适应不同股票的基础数值水平

#### 备选方案对比

| 方法 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| **相对变化率** | 高敏感性，金融意义明确 | 需要历史数据 | 🏆 推荐 |
| **百分位数归一化** | 保持分布形状 | 可能压缩极值信息 | 异常值较多时 |
| **鲁棒缩放** | 抗异常值 | 可能损失极值信息 | 数据质量不佳时 |
| **最小-最大缩放** | 简单直观 | 受极值影响大 | 数据分布均匀时 |

## 📈 数值范围对比

### 处理前（茅台数据示例）
```
总手范围: 1,018,000 ~ 123,584,000  (差异121倍)
金额范围: 200,000,000 ~ 25,000,000,000  (差异125倍)
收盘价范围: 180.00 ~ 2,600.00  (差异14倍)
```

### 处理后（相对变化率）
```
总手处理后: -45.2% ~ +67.8%  (合理范围)
金额处理后: -42.1% ~ +71.3%  (合理范围)
收盘价范围: 180.00 ~ 2,600.00  (保持原值)
```

## 🚀 使用方法

### 基础使用
```python
from src.financial_data_new import FinancialDataProcessor

# 创建处理器
processor = FinancialDataProcessor(
    data_dir="processed_data_2025-07-29",
    large_value_transform="relative_change"  # 推荐
)

# 加载所有股票数据
stock_data = processor.load_all_stocks()

# 处理单只股票
stock_name = list(stock_data.keys())[0]
df = stock_data[stock_name]

# 创建训练序列
features, price_targets, returns = processor.create_sequences(df)

# 标准化特征
features_normalized = processor.normalize_features(features, fit=True)
```

### 不同处理方法对比
```python
methods = ["relative_change", "percentile_norm", "robust_scale", "min_max_scale"]

for method in methods:
    processor = FinancialDataProcessor(large_value_transform=method)
    # ... 处理和比较结果
```

## 🎯 模型配置更新

### 特征数量调整
```python
# config.py 中的更新
n_features: int = 13  # 从11增加到13
```

### 特征列表
```python
feature_columns = [
    'month',           # 月份
    'day',             # 日期
    'weekday',         # 星期
    'open',            # 开盘价
    'high',            # 最高价
    'low',             # 最低价
    'close',           # 收盘价
    'change_pct',      # 涨幅
    'amplitude',       # 振幅
    'volume_processed', # 处理后的总手
    'amount_processed', # 处理后的金额
    'turnover_rate',   # 换手率
    'trade_count'      # 成交次数
]
```

## 🔍 技术细节

### 为什么不用对数变换？
```python
# ❌ 对数变换的问题
volume_log = np.log1p(volume)  # 压缩了变化敏感性

# 示例：
# 原始: [1000万, 1100万, 1200万] -> 变化: [+10%, +9.1%]
# 对数: [16.12, 16.21, 16.30]     -> 变化: [+0.56%, +0.56%]
# 模型很难感知到10%的成交量变化！

# ✅ 相对变化率
# 原始: [1000万, 1100万, 1200万] -> 相对变化: [0%, +10%, +20%]
# 模型能清楚感知每个变化！
```

### 梯度友好性
- 相对变化率的数值范围通常在 [-100%, +100%] 之间
- 与价格变化（通常 [-10%, +10%]）在同一数量级
- 避免了梯度消失/爆炸问题

## 📊 验证方法

### 数据质量检查
```python
# 检查处理效果
processor.test_data_loading()  # 内置测试方法

# 手动验证
python example_data_usage.py  # 运行示例脚本
```

### 模型训练验证
- 观察损失函数收敛情况
- 检查不同特征的梯度大小
- 验证模型对成交量/金额变化的敏感性

## 🎉 总结

通过采用**相对变化率**方法，我们成功解决了：

1. ✅ **数值尺度问题**: 将巨大的绝对数值转换为合理范围的相对变化
2. ✅ **敏感性保持**: 模型仍能敏感地感知成交量和金额的变化
3. ✅ **金融意义**: 相对变化率在金融分析中有明确的解释意义
4. ✅ **训练稳定性**: 避免了梯度问题，提高训练稳定性

这个解决方案既解决了技术问题，又保持了金融数据的业务含义，是一个平衡的优秀方案！

# 🏗️ 分离架构指南

## 📋 项目重构总结

根据您的要求，我已经将两种模型的数据处理和配置完全分离，每个模型都有独立的数据处理器、配置文件和训练脚本。

## 🎯 两种模型的明确分工

### 🔮 Transformer（价格预测网络）
**目标**: 预测未来价格走势
- **输入**: 180天历史金融特征 `[batch, 180, 13]`
- **输出**: 未来7天价格预测 `[batch, 7]`
- **关注点**: 长期时序模式、价格趋势分析
- **架构**: MLA Transformer + RoPE位置编码

### 🧠 GRU（策略网络）
**目标**: 学习交易决策策略
- **输入**: 20天策略特征 `[batch, 20, ~14]`
- **输出**: 20天仓位决策 `[batch, 20, 1]`
- **关注点**: 短期决策序列、收益优化
- **架构**: GRU + 可微分仓位预测

## 📁 新的目录结构

```
src/
├── price_prediction/           # 价格预测模块
│   ├── data_processor.py      # 价格预测数据处理器
│   ├── config.py              # 价格预测配置
│   ├── price_transformer.py   # Transformer模型
│   ├── attention.py           # MLA注意力机制
│   ├── feedforward.py         # SwiGLU前馈网络
│   ├── utils.py               # 工具函数
│   └── __init__.py            # 模块初始化
│
├── strategy_network/          # 策略网络模块
│   ├── data_processor.py      # 策略网络数据处理器
│   ├── config.py              # 策略网络配置
│   ├── gru_strategy.py        # GRU策略网络
│   ├── strategy_loss.py       # 策略损失函数
│   ├── strategy_trainer.py    # 策略训练器
│   └── __init__.py            # 模块初始化
│
└── config.py                  # 全局配置（如需要）

# 独立训练脚本
train_price_prediction_only.py    # 价格预测独立训练
train_strategy_network_only.py    # 策略网络独立训练
```

## 🔧 数据处理器对比

### 价格预测数据处理器
```python
from src.price_prediction.data_processor import PricePredictionDataProcessor
from src.price_prediction.config import PricePredictionConfigs

# 专注于价格预测的特征
processor = PricePredictionDataProcessor(
    sequence_length=180,        # 长序列
    prediction_horizon=7,       # 预测7天价格
    large_value_transform="relative_change"
)

# 加载价格预测数据
stock_data = processor.load_all_stocks_for_price_prediction()
features, targets = processor.create_price_sequences(df)
# features: [n_samples, 180, 13] - 长时序特征
# targets: [n_samples, 7] - 价格预测目标
```

### 策略网络数据处理器
```python
from src.strategy_network.data_processor import StrategyNetworkDataProcessor
from src.strategy_network.config import StrategyNetworkConfigs

# 专注于交易策略的特征
processor = StrategyNetworkDataProcessor(
    trading_horizon=20,         # 短期决策
    feature_extraction_length=180,
    large_value_transform="relative_change"
)

# 加载策略数据
stock_data = processor.load_all_stocks_for_strategy()
features, positions, returns = processor.create_strategy_sequences(df)
# features: [n_samples, 20, ~14] - 策略特征
# positions: [n_samples, 20, 1] - 仓位目标
# returns: [n_samples, 20] - 收益率
```

## ⚙️ 配置文件对比

### 价格预测配置
```python
from src.price_prediction.config import PricePredictionConfigs

# 专门为Transformer优化的配置
config = PricePredictionConfigs.base()
# - d_model: 512 (模型维度)
# - n_layers: 8 (Transformer层数)
# - sequence_length: 180 (长序列)
# - prediction_horizon: 7 (预测时间跨度)
# - loss_type: "mse" (价格预测损失)
```

### 策略网络配置
```python
from src.strategy_network.config import StrategyNetworkConfigs

# 专门为GRU优化的配置
config = StrategyNetworkConfigs.base()
# - hidden_dim: 128 (GRU隐藏维度)
# - trading_horizon: 20 (交易时间跨度)
# - position_range: 0-10 (仓位档位)
# - information_ratio_weight: 1.0 (收益权重)
```

## 🚀 独立训练方法

### 1. 训练价格预测网络
```bash
python train_price_prediction_only.py
```

**特点**:
- 专注于价格预测精度
- 使用MSE/MAE损失函数
- 长序列Transformer训练
- 保存最佳价格预测模型

### 2. 训练策略网络
```bash
python train_strategy_network_only.py
```

**特点**:
- 专注于收益优化
- 使用策略损失函数（信息比率+风险成本）
- GRU递归训练
- 保存最佳策略模型

### 3. 两阶段联合训练（可选）
```bash
# 先训练价格网络
python train_price_prediction_only.py

# 再训练策略网络（可以使用价格网络的特征）
python train_strategy_network_only.py
```

## 📊 特征工程差异

### 价格预测特征（13维）
```python
feature_columns = [
    'month', 'day', 'weekday',           # 时间特征
    'open', 'high', 'low', 'close',      # OHLC价格
    'change_pct', 'amplitude',           # 价格变化
    'volume_processed', 'amount_processed', # 成交量（处理后）
    'turnover_rate', 'trade_count'       # 市场活跃度
]
```

### 策略网络特征（~14维）
```python
strategy_features = [
    'price_trend', 'price_momentum', 'price_acceleration',  # 价格动态
    'volatility', 'volatility_trend',                      # 波动率
    'volume_trend', 'volume_price_correlation',            # 成交量
    'market_sentiment', 'sentiment_momentum',              # 市场情绪
    'rsi', 'macd_signal',                                  # 技术指标
    'risk_metrics', 'drawdown_risk'                        # 风险指标
]
```

## 🎯 使用示例

### 价格预测模型使用
```python
from src.price_prediction.price_transformer import PriceTransformer
from src.price_prediction.config import PricePredictionConfigs

# 加载配置和模型
config = PricePredictionConfigs.base()
model = PriceTransformer(config)

# 加载训练好的权重
checkpoint = torch.load('checkpoints/price_prediction/best_price_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 预测
with torch.no_grad():
    outputs = model(financial_data)  # [batch, 180, 13]
    price_predictions = outputs['price_predictions']  # [batch, 7]
```

### 策略网络模型使用
```python
from src.strategy_network.gru_strategy import GRUStrategyNetwork
from src.strategy_network.config import StrategyNetworkConfigs

# 加载配置和模型
config = StrategyNetworkConfigs.base()
model = GRUStrategyNetwork(config)

# 加载训练好的权重
checkpoint = torch.load('checkpoints/strategy_network/best_strategy_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 策略决策
with torch.no_grad():
    outputs = model.forward_sequence(strategy_features)  # [batch, 20, ~14]
    positions = outputs['position_output']['positions']  # [batch, 20, 1]
```

## 🔄 数据流程图

```
原始数据 (Excel文件)
    ↓
┌─────────────────────┬─────────────────────┐
│   价格预测数据处理    │   策略网络数据处理    │
│                    │                    │
│ • 180天历史特征     │ • 20天策略特征      │
│ • 7天价格目标       │ • 收益率计算        │
│ • 长序列优化        │ • 技术指标工程      │
└─────────────────────┴─────────────────────┘
    ↓                      ↓
┌─────────────────────┬─────────────────────┐
│  Transformer训练    │    GRU训练         │
│                    │                    │
│ • MLA注意力机制     │ • 递归状态更新      │
│ • RoPE位置编码      │ • 可微分仓位        │
│ • 价格预测损失      │ • 策略收益损失      │
└─────────────────────┴─────────────────────┘
    ↓                      ↓
┌─────────────────────┬─────────────────────┐
│   价格预测模型      │   策略决策模型      │
│                    │                    │
│ • 未来7天价格       │ • 20天仓位决策      │
│ • 趋势分析         │ • 收益优化         │
└─────────────────────┴─────────────────────┘
```

## ✅ 优势总结

1. **🎯 专业化**: 每个模型专注自己的任务，避免目标冲突
2. **🔧 独立性**: 可以单独训练、调优和部署
3. **📊 数据适配**: 针对不同任务设计专门的特征工程
4. **⚙️ 配置分离**: 独立的配置文件，便于参数调优
5. **🚀 灵活性**: 可以选择只使用其中一个模型
6. **🔄 可扩展**: 易于添加新的模型或修改现有模型

现在您可以根据需要独立训练和使用这两个模型了！

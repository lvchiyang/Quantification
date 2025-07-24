# 🚀 金融量化交易系统

基于深度学习的两阶段金融量化交易系统，采用 **价格预测网络** + **策略网络** 的解耦架构，专门用于股票价格预测和交易策略学习。

## ✨ 核心特性

### 🏗️ 两阶段解耦架构
- **价格预测网络**: 基于MLA Transformer的价格预测模型
- **策略网络**: 基于GRU的交易策略学习网络
- **完全解耦**: 两个网络独立训练，避免目标冲突
- **专业化优化**: 每个网络专注自己的任务

### 📈 价格预测网络
- **MLA注意力机制**: 多头潜在注意力，压缩K/V降低计算复杂度
- **RoPE位置编码**: 旋转位置编码，更好处理时序关系
- **11维金融特征**: OHLC + 技术指标 + 时间编码
- **7天价格预测**: 预测未来7个时间点的收盘价

### 🧠 策略网络
- **GRU记忆网络**: 20天递归状态更新的策略记忆
- **离散仓位决策**: 0-10档位的可微分仓位预测
- **智能损失函数**: 相对基准收益 + 风险成本 + 机会成本
- **市场自适应**: 自动识别牛市/熊市/震荡市并调整策略

## 🏗️ 项目结构

```
Quantification/
├── src/                          # 核心源码
│   ├── price_prediction/         # 价格预测网络模块
│   │   ├── price_transformer.py  # 价格预测Transformer
│   |   ├── attention.py          # MLA注意力机制（共享）
│   |   ├── feedforward.py        # SwiGLU前馈网络（共享）
│   |   ├── utils.py              # 工具函数（共享）
│   │   └── __init__.py           # 模块初始化
│   ├── strategy_network/         # 策略网络模块
│   │   ├── gru_strategy.py       # GRU策略网络
│   │   ├── strategy_loss.py      # 策略损失函数
│   │   ├── strategy_trainer.py   # 策略训练器
│   │   └── __init__.py           # 模块初始化
│   ├── config.py                 # 模型配置
│   └── financial_data.py         # 金融数据处理
├── train.py                      # 两阶段训练入口
├── train_price_network.py        # 价格网络训练
├── train_strategy_network.py     # 策略网络训练
└── requirements.txt              # 项目依赖
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
```

### 2. 开始训练

```bash
# 完整两阶段训练
python train.py

# 或分阶段训练
python train.py price        # 只训练价格网络
python train.py strategy     # 只训练策略网络

# 或独立训练
python train_price_network.py    # 第一阶段：价格预测
python train_strategy_network.py # 第二阶段：策略学习
```

### 3. 模型配置

```python
from src.config import ModelConfigs

# 创建配置
config = ModelConfigs.tiny()  # 轻量级配置，适合快速实验

# 启用状态化训练
config.enable_stateful_training = True
config.strategy_state_dim = 128
config.state_update_method = 'gru'  # 'gru', 'lstm', 'attention'

# 调整损失权重
config.information_ratio_weight = 1.0
config.opportunity_cost_weight = 0.1
config.risk_adjustment_weight = 0.05
```

## 📊 数据格式

### 输入特征 (11维)
1. **开盘价** (Open)
2. **最高价** (High) 
3. **最低价** (Low)
4. **收盘价** (Close)
5. **涨幅** (Change %)
6. **振幅** (Amplitude %)
7. **总手** (Volume)
8. **金额** (Amount)
9. **换手率** (Turnover %)
10. **成交次数** (Trade Count)
11. **时间编码** (Time Encoding)

### 数据示例
```
时间,开盘,最高,最低,收盘,涨幅,振幅,总手,金额,换手%,成交次数
2009-10-15,16.11,17.51,15.53,17.08,44.99%,16.81%,153586470,2501742900,87.27,2867
```

### 输出预测
- **价格预测**: 未来7个时间点的收盘价
- **仓位决策**: 0-10档位的交易仓位

## 🔄 两阶段训练详解

### 核心思想
采用解耦的两阶段训练方法，避免价格预测和策略学习的目标冲突：

```python
# 第一阶段：价格预测网络训练
price_network = PriceTransformer(config)
price_loss = mse_loss(price_pred, price_target)  # 专注价格预测精度

# 第二阶段：策略网络训练（基于冻结的价格网络）
strategy_network = GRUStrategyNetwork(config)
price_features = price_network.extract_features(data)  # 冻结特征提取
strategy_loss = -relative_return + risk_cost + opportunity_cost  # 专注策略收益
```

### 优势对比

| 特性 | 耦合训练 | 两阶段解耦训练 |
|------|----------|----------------|
| **梯度传播** | 部分阻断 | 完全畅通 |
| **目标冲突** | 严重 | 无冲突 |
| **专业化程度** | 低 | 高 |
| **调优难度** | 困难 | 简单 |
| **训练效率** | 低 | 高 |

### 信息比率损失
解决了传统方法在不同市场环境下评价不公平的问题：

```python
# 自动选择基准
if market_type == 'bull':
    benchmark = buy_and_hold_strategy()    # 与满仓比较
elif market_type == 'bear':
    benchmark = conservative_strategy()    # 与保守策略比较
else:
    benchmark = momentum_strategy()        # 与动量策略比较

# 计算信息比率
information_ratio = excess_return_mean / excess_return_std
loss = -information_ratio + opportunity_cost + risk_penalty
```

## 🎯 使用示例

### 基础预测
```python
from src.price_prediction.price_transformer import PriceTransformer
from src.config import ModelConfigs

# 创建模型
config = ModelConfigs.tiny()
model = PriceTransformer(config)

# 预测
financial_data = torch.randn(1, 180, 11)  # [batch, seq_len, features]
outputs = model(financial_data)

print(f"价格预测: {outputs['price_predictions']}")
```

### 策略预测
```python
from src.strategy_network.gru_strategy import GRUStrategyNetwork
from src.price_prediction.price_transformer import PriceTransformer

# 加载预训练的价格网络
price_network = PriceTransformer(config)
price_network.load_state_dict(torch.load('best_price_network.pth'))

# 创建策略网络
strategy_network = GRUStrategyNetwork(config)

# 提取特征并预测仓位
with torch.no_grad():
    features = price_network.extract_features(financial_data)

positions = strategy_network.forward_sequence(features)
print(f"仓位决策: {positions['position_output']['positions']}")
```

## 📈 性能指标

模型会自动计算多种评估指标：

- **累计收益率**: 策略的总收益表现
- **信息比率**: 超额收益的风险调整指标
- **夏普比率**: 收益风险比
- **最大回撤**: 最大损失幅度
- **机会成本**: 错失收益的量化
- **风险惩罚**: 波动率和回撤的综合

## 🔧 高级配置

### 模型规模
```python
# 不同规模的预设配置
config = ModelConfigs.tiny()     # 轻量级：适合快速实验
config = ModelConfigs.small()    # 小型：适合个人电脑
config = ModelConfigs.base()     # 中型：适合服务器训练
config = ModelConfigs.large()    # 大型：适合高性能计算
```

### 离散化方法
```python
config.position_method = 'gumbel_softmax'    # Gumbel-Softmax (推荐)
config.position_method = 'straight_through'  # 直通估计器
config.position_method = 'concrete'          # Concrete分布
```

### 状态更新方式
```python
config.state_update_method = 'gru'        # GRU (推荐)
config.state_update_method = 'lstm'       # LSTM
config.state_update_method = 'attention'  # 注意力机制
```

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- PyTorch 团队提供的深度学习框架
- Transformer 架构的原始论文作者
- MLA (Multi-Head Latent Attention) 的研究者
- 金融量化社区的宝贵建议

## 📚 更多信息

更多技术细节和高级用法，请参阅 [技术详情文档](TECHNICAL_DETAILS.md)。

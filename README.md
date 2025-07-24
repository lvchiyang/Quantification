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
│   │   ├── attention.py          # MLA注意力机制
│   │   ├── feedforward.py        # SwiGLU前馈网络
│   │   └── utils.py              # 工具函数
│   ├── strategy_network/         # 策略网络模块
│   │   ├── gru_strategy.py       # GRU策略网络
│   │   ├── strategy_loss.py      # 策略损失函数
│   │   └── strategy_trainer.py   # 策略训练器
│   ├── config.py                 # 模型配置
│   └── financial_data.py         # 金融数据处理
├── examples/                     # 示例代码
├── tests/                        # 测试文件
├── train.py                      # 两阶段训练入口
├── train_price_network.py        # 价格网络训练
├── train_strategy_network.py     # 策略网络训练
└── test_stateful_model.py        # 功能测试脚本
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch einops transformers datasets numpy tqdm matplotlib pandas scikit-learn

# 验证安装
python test_stateful_model.py
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
config = ModelConfigs.tiny()

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
from src.transformer import FinancialTransformer
from src.config import ModelConfigs

# 创建模型
config = ModelConfigs.tiny()
model = FinancialTransformer(config)

# 预测
financial_data = torch.randn(1, 180, 11)  # [batch, seq_len, features]
outputs = model(financial_data)

print(f"价格预测: {outputs['price_predictions']}")
print(f"仓位决策: {outputs['position_predictions']}")
```

### 状态化预测
```python
# 20天递归预测
strategy_state = None
positions_over_time = []

for day in range(20):
    outputs = model.forward_single_day(
        financial_data[day], 
        strategy_state=strategy_state
    )
    
    positions_over_time.append(outputs['position_predictions'])
    strategy_state = outputs['strategy_state']  # 更新状态

print(f"20天仓位序列: {positions_over_time}")
```

### 自定义训练
```python
from src.recurrent_trainer import RecurrentStrategyTrainer

# 创建训练器
trainer = RecurrentStrategyTrainer(model, use_information_ratio=True)

# 训练步骤
sliding_window_data = {
    'features': features,          # [batch, 20, 180, 11]
    'price_targets': targets,      # [batch, 20, 7]
    'next_day_returns': returns    # [batch, 20]
}

loss_dict = trainer.train_step(sliding_window_data)
loss_dict['loss_tensor'].backward()
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
config = ModelConfigs.medium()   # 中型：适合服务器训练
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

## 🧪 测试和验证

```bash
# 功能测试
python test_stateful_model.py

# 基础模型测试
python tests/simple_test.py

# 交易策略测试
python tests/test_trading_strategy.py

# 滑动窗口测试
python tests/test_sliding_window.py
```

## 📚 相关文档

- [快速开始指南](QUICKSTART.md) - 5分钟快速上手
- [金融模型详解](README_Financial.md) - 金融特性详细说明
- [项目总结](PROJECT_SUMMARY.md) - 技术实现总结
- [网络架构](网络模型.md) - 模型架构详细说明

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- PyTorch 团队提供的深度学习框架
- Transformer 架构的原始论文作者
- MLA (Multi-Head Latent Attention) 的研究者
- 金融量化社区的宝贵建议

## 🔬 技术细节

### MLA (Multi-Head Latent Attention) 架构
```python
# 传统注意力 vs MLA
Traditional: Q@K^T@V  # O(n²d) 复杂度
MLA: Q@(W_kv@X)^T@(W_kv@X)  # O(nd²) 复杂度，d<<n时更高效

# K/V压缩示例
original_kv_dim = 1024    # 原始维度
compressed_dim = 256      # 压缩后维度
compression_ratio = 4     # 压缩比例
```

### 递归状态更新机制
```python
# 梯度传播路径
final_loss → position_logits[19] → strategy_state[19]
          → position_logits[18] → strategy_state[18]
          → ... → position_logits[0]

# 内存优化
for day in range(20):
    # 保持梯度的部分
    position = model.predict_position(features[day], strategy_state)
    strategy_state = model.update_state(strategy_state, position)

    # 不保持梯度的部分
    with torch.no_grad():
        portfolio_value *= (1 + position * returns[day])
```

### 市场分类算法
```python
def classify_market(returns):
    # 1. 统计特征
    mean_return = torch.mean(returns)
    volatility = torch.std(returns)

    # 2. 技术指标
    ma_short = torch.mean(returns[-5:])
    ma_long = torch.mean(returns[-10:])

    # 3. 投票机制
    votes = [
        simple_classifier(mean_return),
        technical_classifier(ma_short, ma_long),
        adaptive_classifier(returns)
    ]

    return majority_vote(votes)
```

## 🎯 性能基准

### 训练性能
| 配置 | 参数量 | 训练速度 | 内存使用 | 推荐场景 |
|------|--------|----------|----------|----------|
| Tiny | 2.5M | 快 | 2GB | 快速实验 |
| Small | 10M | 中等 | 4GB | 个人开发 |
| Medium | 40M | 慢 | 8GB | 服务器训练 |
| Large | 160M | 很慢 | 16GB | 生产环境 |

### 策略表现 (回测示例)
```
时间段: 2020-2023
基准: 沪深300指数

传统方法:
- 年化收益: 8.5%
- 最大回撤: -15.2%
- 夏普比率: 0.65

状态化方法:
- 年化收益: 12.3%
- 最大回撤: -11.8%
- 夏普比率: 0.89
- 信息比率: 0.45
```

## 🛠️ 故障排除

### 常见问题

**Q: 训练时内存不足？**
```python
# 解决方案：减小批次大小
config.batch_size = 1  # 从默认的4减少到1
config.strategy_state_dim = 64  # 减小状态维度
```

**Q: 梯度消失或爆炸？**
```python
# 解决方案：调整学习率和梯度裁剪
optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # 降低学习率
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
```

**Q: 模型不收敛？**
```python
# 解决方案：检查数据和损失权重
config.information_ratio_weight = 0.5  # 降低信息比率权重
config.opportunity_cost_weight = 0.05   # 降低机会成本权重
```

### 调试技巧
```python
# 1. 检查梯度流
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.6f}")

# 2. 监控状态变化
print(f"状态变化: {torch.mean(torch.abs(new_state - old_state)).item():.6f}")

# 3. 验证市场分类
market_type = model.market_classifier.classify_market(returns)
print(f"市场类型: {market_type}")
```

## 📊 可视化工具

### 训练过程可视化
```python
import matplotlib.pyplot as plt

# 绘制损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(cumulative_returns, label='累计收益')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(information_ratios, label='信息比率')
plt.legend()
plt.show()
```

### 策略分析
```python
# 仓位变化分析
positions = model.predict_positions(test_data)
plt.figure(figsize=(10, 6))
plt.plot(positions, label='仓位变化')
plt.plot(returns, label='市场收益', alpha=0.7)
plt.legend()
plt.title('仓位决策 vs 市场表现')
plt.show()
```

## 🔮 未来规划

### 即将推出的功能
- [ ] **多资产组合**: 支持多只股票的组合优化
- [ ] **强化学习**: 集成PPO/SAC等强化学习算法
- [ ] **实时交易**: 对接实盘交易接口
- [ ] **风险模型**: 更精细的风险控制模块
- [ ] **因子挖掘**: 自动特征工程和因子发现

### 技术改进
- [ ] **模型压缩**: 知识蒸馏和模型剪枝
- [ ] **分布式训练**: 多GPU和多机训练支持
- [ ] **在线学习**: 增量学习和模型更新
- [ ] **解释性**: 注意力可视化和决策解释

## 🌟 社区与支持

### 加入社区
- **GitHub Discussions**: 技术讨论和问题解答
- **Issues**: Bug报告和功能请求
- **Wiki**: 详细文档和教程

### 获得帮助
1. 查看 [FAQ](docs/FAQ.md)
2. 搜索已有的 [Issues](https://github.com/your-repo/issues)
3. 提交新的 Issue 并提供详细信息
4. 参与 [Discussions](https://github.com/your-repo/discussions)

---

**🚀 开始您的量化交易之旅！**

如有问题，请提交 Issue 或查看文档。

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**

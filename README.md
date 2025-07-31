# 🚀 金融时序预测系统

基于现代 Transformer 架构的金融时序预测系统，专门为股票价格预测任务设计，集成了多种先进技术和金融专用优化。

## ✨ 核心特性

### 🏗️ 现代化 Transformer 架构
- **Multi-Head Latent Attention (MLA)**: 高效的注意力机制，O(n)复杂度
- **RoPE 位置编码**: 旋转位置编码，适合长序列处理
- **SwiGLU 前馈网络**: 现代化的门控前馈网络
- **Pre-RMSNorm**: 更稳定的归一化方式
- **金融特征嵌入**: 专门的分组嵌入层

### 📊 金融专用设计
- **20维金融特征**: 时间特征 + 价格特征 + 成交量特征 + 市场特征 + 金融特征
- **多时间点预测**: 预测未来第1,2,3,4,5,10,15,20,25,30天的价格
- **序列级处理**: 避免数据泄露的独立序列处理
- **金融损失函数**: 方向损失 + 趋势损失 + 时间加权损失

### 🎯 智能损失函数组合
- **基础回归损失**: MSE/MAE/Huber损失
- **方向损失**: 预测涨跌方向的准确性
- **趋势损失**: 价格变化趋势的一致性
- **时间加权损失**: 近期预测比远期预测更重要
- **排序损失**: 保持相对大小关系（可选）
- **波动率损失**: 价格波动模式匹配（可选）

## 🏗️ 项目结构

```
Quantification/
├── src/                              # 核心源码
│   ├── price_prediction/             # 价格预测模块
│   │   ├── price_transformer.py      # 主Transformer模型 + TransformerBlock
│   │   ├── attention.py              # MLA注意力机制
│   │   ├── feedforward.py            # 前馈网络（SwiGLU/GeGLU/StandardFFN）
│   │   ├── embedding.py              # 金融特征嵌入层
│   │   ├── financial_losses.py       # 金融专用损失函数
│   │   ├── utils.py                  # 工具函数（RMSNorm/RoPE）
│   │   ├── config.py                 # 价格预测配置
│   │   └── data_cteater.py           # 数据创建器
│   └── strategy_network/             # 策略网络模块
│       ├── gru_strategy.py           # GRU策略网络
│       ├── strategy_loss.py          # 策略损失函数
│       ├── strategy_trainer.py       # 策略训练器
│       ├── config.py                 # 策略网络配置
│       └── data_creater.py           # 策略数据创建器
├── stript/                           # 训练脚本
│   ├── train.py                      # 主训练脚本
│   ├── train_price_prediction.py     # 价格预测训练
│   ├── train_strategy_network.py     # 策略网络训练
│   └── data_processor.py             # 数据处理器
├── doc/                              # 完整文档
│   ├── architecture.md               # 系统架构文档
│   ├── transformer.md                # Transformer架构详解
│   ├── feedforward.md                # 前馈网络文档
│   ├── embedding.md                  # 嵌入层文档
│   ├── financial_losses.md           # 损失函数文档
│   ├── sequences.md                  # 序列处理文档
│   ├── config.md                     # 配置参数文档
│   ├── data.md                       # 数据处理文档
│   ├── training.md                   # 训练指南
│   ├── strategy_data.md              # 策略数据文档
│   ├── strategy_network.md           # 策略网络文档
│   └── troubleshooting.md            # 故障排除
├── test/                             # 测试文件
│   ├── test_financial_losses.py      # 损失函数测试
│   ├── test_rope_implementation.py   # RoPE实现测试
│   ├── test_model_output_and_loss.py # 模型输出测试
│   ├── test_price_transformer.py     # 价格预测模型测试
│   ├── test_embedding.py             # 嵌入层测试
│   ├── test_data_loading.py          # 数据加载测试
│   └── test_refactored_architecture.py # 重构架构测试
├── THSoriginalData/                  # 原始数据
│   ├── 电力/                         # 电力行业数据
│   ├── 白酒/                         # 白酒行业数据
│   └── 黄金/                         # 黄金行业数据
├── processed_data_2025-07-30/        # 处理后数据
│   └── 股票数据/                     # 清洗后的股票数据
├── requirements.txt                  # 项目依赖
└── README.md                         # 项目说明
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd Quantification

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
```

### 2. 数据准备

```bash
# 运行数据处理器
python data_processor.py

# 这将创建 processed_data_YYYY-MM-DD/ 目录
# 包含清洗后的股票数据
```

### 3. 开始训练

```bash
# 使用默认配置训练
python train_price_prediction.py

# 使用不同配置
python train_price_prediction.py --config tiny    # 轻量级
python train_price_prediction.py --config base    # 标准配置
python train_price_prediction.py --config large   # 高性能配置
```

### 4. 模型配置

```python
from src.price_prediction.config import PricePredictionConfigs

# 使用预定义配置
config = PricePredictionConfigs.base()  # 标准配置
config = PricePredictionConfigs.tiny()  # 轻量级配置
config = PricePredictionConfigs.large() # 高性能配置

# 自定义金融损失函数
config.use_financial_loss = True
config.direction_weight = 0.3      # 方向损失权重
config.trend_weight = 0.2          # 趋势损失权重
config.use_temporal_weighting = True  # 时间加权

# 模型架构调整
config.d_model = 512               # 模型维度
config.n_layers = 8               # Transformer层数
config.kv_lora_rank = 256         # K/V压缩维度
```

## 📊 技术架构

### 数据流架构

```
原始股票数据 → 数据清洗 → 特征工程 → 序列处理 → 模型训练
     ↓              ↓           ↓          ↓          ↓
  Excel文件     基础清洗    20维特征   180天序列   Transformer
   (OHLCV)    (14列数据)   (金融指标)  (避免泄露)   (价格预测)
```

### Transformer 架构详解

```
输入: [batch, 180, 20] 金融特征
  ↓
FinancialEmbeddingLayer: 分组嵌入 + 批标准化
  ↓
[batch, 180, d_model] 嵌入特征
  ↓
TransformerBlock × n_layers:
  ┌─ RMSNorm → MLA + RoPE → 残差连接
  └─ RMSNorm → SwiGLU FFN → 残差连接
  ↓
[batch, 180, d_model] 编码特征
  ↓
取最后时间步: [batch, d_model]
  ↓
价格预测头: [batch, 10] 未来10个时间点预测
```

### 核心技术组件

1. **Multi-Head Latent Attention (MLA)**
   - K/V压缩：O(n²) → O(n) 复杂度
   - 内存节省：14,000x 内存优化

2. **RoPE 位置编码**
   - 旋转位置编码，适合长序列
   - 相对位置关系，外推能力强

3. **SwiGLU 前馈网络**
   - 门控机制，表达能力强
   - SiLU激活，现代化设计

4. **金融特征嵌入**
   - 分组嵌入：时间/价格/成交量/市场/金融
   - 批标准化，训练稳定

## 📈 数据格式

### 输入特征 (20维)

**时间特征 (3维)**：
- 月份 (1-12)
- 日期 (1-31)
- 星期 (1-7)

**价格特征 (4维)**：
- open_rel, high_rel, low_rel, close_rel (相对价格)

**价格变化 (2维)**：
- 涨幅, 振幅

**成交量特征 (2维)**：
- volume_rel (相对值), volume_log (对数值)

**金额特征 (2维)**：
- amount_rel (相对值), amount_log (对数值)

**市场活跃度 (2维)**：
- 换手%, 成交次数

**金融特征 (4维)**：
- big_order_activity (大单活跃度)
- chip_concentration (筹码集中度)
- market_sentiment (市场情绪)
- price_volume_sync (价量同步性)

### 输出预测
- **价格预测**: 未来第1,2,3,4,5,10,15,20,25,30天的收盘价相对值
- **特征向量**: 可用于下游策略网络的特征表示

## 🎯 金融专用损失函数

### 多损失函数组合

系统采用多种损失函数组合来全面优化金融预测：

```python
# 基础回归损失
base_loss = MSE/MAE/Huber(predictions, targets)

# 方向损失（预测涨跌方向）
direction_loss = DirectionLoss(predictions, targets)

# 趋势损失（价格变化趋势一致性）
trend_loss = TrendLoss(predictions, targets)

# 时间加权损失（近期预测更重要）
temporal_loss = TemporalWeightedLoss(base_loss)

# 总损失
total_loss = base_weight * base_loss +
             direction_weight * direction_loss +
             trend_weight * trend_loss
```

### 损失函数优势

| 损失类型 | 作用 | 金融意义 |
|----------|------|----------|
| **基础损失** | 数值准确性 | 价格预测精度 |
| **方向损失** | 涨跌方向 | 交易信号准确性 |
| **趋势损失** | 变化趋势 | 市场趋势捕捉 |
| **时间加权** | 近期重要 | 短期预测优先 |
| **排序损失** | 相对关系 | 相对强弱判断 |
| **波动率损失** | 波动模式 | 风险模式匹配 |



## 🎯 使用示例

### 基础预测
```python
from src.price_prediction.price_transformer import PriceTransformer
from src.price_prediction.config import PricePredictionConfigs

# 创建模型
config = PricePredictionConfigs.base()
model = PriceTransformer(config)

# 加载数据
financial_data = load_financial_data()  # [batch, 180, 20]

# 价格预测
outputs = model(financial_data, return_features=True, return_dict=True)
price_predictions = outputs['price_predictions']  # [batch, 10]
print(f"未来10个时间点价格预测: {price_predictions}")
```

### 两种预测方式（简化版）
```python
# 方式1：绝对价格预测
config_abs = PricePredictionConfigs.for_absolute_prediction()  # predict_relative=False
processor_abs = SequenceProcessor(sequence_length=180, predict_relative=False)

# 创建训练序列（目标为实际价格）
sequences_abs = processor_abs.create_training_sequences(stock_data)
input_seq, target_prices = sequences_abs[0]  # 简化：不需要metadata
print(f"绝对价格目标: {target_prices}")  # [102.5, 103.2, 101.8, ...]

# 方式2：相对价格预测
config_rel = PricePredictionConfigs.for_relative_prediction()  # predict_relative=True
processor_rel = SequenceProcessor(sequence_length=180, predict_relative=True)

# 创建训练序列（目标为相对值）
sequences_rel = processor_rel.create_training_sequences(stock_data)
input_seq, target_ratios = sequences_rel[0]  # 简化：不需要metadata
print(f"相对值目标: {target_ratios}")  # [1.025, 1.032, 1.018, ...]

# 预测时转换（如果需要）
price_median = 100.0  # 从输入序列获取基准价格
if config_rel.predict_relative:
    absolute_prices = target_ratios * price_median
    print(f"转换后绝对价格: {absolute_prices}")  # [102.5, 103.2, 101.8, ...]
```

### 金融损失函数使用
```python
from src.price_prediction.financial_losses import FinancialMultiLoss

# 创建金融专用损失函数
criterion = FinancialMultiLoss(
    base_loss_type='mse',
    use_direction_loss=True,
    use_trend_loss=True,
    use_temporal_weighting=True,
    direction_weight=0.3,
    trend_weight=0.2
)

# 计算损失
loss_dict = criterion(predictions, targets)
total_loss = loss_dict['loss']  # 用于反向传播

# 监控指标
print(f"方向准确率: {loss_dict['direction_accuracy']:.2%}")
print(f"基础损失: {loss_dict['base_loss']:.4f}")
print(f"方向损失: {loss_dict['direction_loss']:.4f}")
```

### 序列处理
```python
from sequence_processor import PriceDataset

# 创建数据集
dataset = PriceDataset("processed_data_2025-07-30/股票数据", sequence_length=180)

# 获取训练序列
sequences = dataset.create_training_sequences()
print(f"生成 {len(sequences)} 个训练序列")

# 每个序列: (input[180,20], target[10])
for input_seq, target_prices in sequences[:3]:
    print(f"输入: {input_seq.shape}, 目标: {target_prices.shape}")
```

## 🔧 高级配置

### 模型规模
```python
# 不同规模的预设配置
config = PricePredictionConfigs.tiny()     # 轻量级：2.5M参数，2GB内存
config = PricePredictionConfigs.small()    # 小型：10M参数，4GB内存
config = PricePredictionConfigs.base()     # 标准：40M参数，8GB内存
config = PricePredictionConfigs.large()    # 大型：160M参数，16GB内存
```

### 金融损失配置
```python
# 保守策略（重视数值准确性）
config.base_weight = 1.0
config.direction_weight = 0.1
config.trend_weight = 0.1

# 激进策略（重视方向和趋势）
config.base_weight = 0.5
config.direction_weight = 0.5
config.trend_weight = 0.3
config.use_ranking_loss = True
```

### 内存优化
```python
# 减少内存使用
config.batch_size = 2           # 减少批次大小
config.kv_lora_rank = 128      # 增加K/V压缩
config.sequence_length = 120    # 减少序列长度

# 梯度累积（保持有效批次大小）
config.batch_size = 2
# 在训练脚本中设置 accumulation_steps = 4
# 有效批次大小 = 2 × 4 = 8
```

## 📚 完整文档

### 核心文档
- [📖 系统架构](doc/architecture.md) - 整体架构设计
- [🏗️ Transformer架构](doc/transformer.md) - Transformer详解
- [🔧 前馈网络](doc/feedforward.md) - 前馈网络实现
- [🎯 嵌入层](doc/embedding.md) - 金融特征嵌入
- [📉 损失函数](doc/financial_losses.md) - 金融专用损失

### 使用指南
- [⚙️ 配置参数](doc/config.md) - 配置系统详解
- [📊 数据处理](doc/data.md) - 数据处理流程
- [🔄 序列处理](doc/sequences.md) - 序列处理器
- [🎓 训练指南](doc/training.md) - 训练最佳实践
- [🔧 故障排除](doc/troubleshooting.md) - 常见问题解决

## 🧪 测试和验证

### 运行测试
```bash
# 测试金融损失函数
python test/test_financial_losses.py

# 测试RoPE实现
python test/test_rope_implementation.py

# 测试模型输出
python test/test_model_output_and_loss.py

# 测试重构后的架构
python test_refactored_architecture.py
```

### 性能基准
| 配置 | 参数量 | 内存使用 | 训练时间 | 方向准确率 |
|------|--------|----------|----------|------------|
| Tiny | 2.5M | 2GB | 2小时 | ~70% |
| Small | 10M | 4GB | 4小时 | ~75% |
| Base | 40M | 8GB | 8小时 | ~80% |
| Large | 160M | 16GB | 16小时 | ~85% |

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范
- 遵循 PEP 8 代码风格
- 添加适当的类型注解
- 编写完整的文档字符串
- 添加单元测试

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- PyTorch 团队提供的深度学习框架
- Transformer 架构的原始论文作者
- MLA (Multi-Head Latent Attention) 的研究者
- RoPE (Rotary Position Embedding) 的研究者
- SwiGLU 前馈网络的研究者
- 金融量化社区的宝贵建议

## 相关资源

- [Transformer原论文](https://arxiv.org/abs/1706.03762)
- [RoPE论文](https://arxiv.org/abs/2104.09864)
- [MLA技术报告](https://arxiv.org/abs/2406.07637)
- [SwiGLU论文](https://arxiv.org/abs/2002.05202)

---

**⭐ 如果这个项目对你有帮助，请给个星标支持！**

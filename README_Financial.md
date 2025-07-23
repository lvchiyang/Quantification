# 金融量化 Transformer 模型

基于 MLA (Multi-Head Latent Attention) 的金融时序预测模型，专门用于股票价格预测。

## 🎯 模型特点

### 输入数据格式
模型接受以下11个金融特征：
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

### 输出预测
- 预测未来 **7个时间点** 的收盘价格

### 数据格式示例
```
时间	开盘	最高	最低	收盘	涨幅	振幅	总手	金额	换手%	成交次数
2009-10-15,四	16.11	17.51	15.53	17.08	44.99%	16.81%	153,586,470	2,501,742,900	87.27	2867
```

## 🏗️ 模型架构

### 核心组件
1. **特征嵌入层**: 将11维金融特征映射到模型维度
2. **MLA注意力**: 多头潜在注意力机制，压缩K/V降低计算复杂度
3. **RoPE位置编码**: 旋转位置编码，更好处理时序关系
4. **SwiGLU前馈网络**: 高效的激活函数
5. **价格预测头**: 输出7个时间点的价格预测

### 与传统NLP模型的区别
- ❌ 不使用词嵌入 (Token Embedding)
- ✅ 使用特征嵌入 (Feature Embedding)
- ❌ 不进行文本生成
- ✅ 进行数值回归预测
- ❌ 不使用交叉熵损失
- ✅ 使用均方误差损失 (MSE)

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install torch numpy pandas einops
```

### 2. 训练模型
```bash
python train_financial_model.py
```

### 3. 使用模型进行预测
```python
import torch
from src.transformer import FinancialTransformer
from src.config import ModelConfigs
from src.financial_data import FinancialDataProcessor

# 加载模型
config = ModelConfigs.tiny()
model = FinancialTransformer(config)
model.load_state_dict(torch.load('best_financial_model.pth'))

# 处理数据
processor = FinancialDataProcessor()
features, _ = processor.process_file('your_data.txt')

# 预测
model.eval()
with torch.no_grad():
    predictions = model.predict(features[-1:])  # 使用最新数据预测
    predicted_prices = predictions['predictions']
    print(f"未来7天预测价格: {predicted_prices[0].numpy()}")
```

## 📊 数据预处理

### FinancialDataProcessor 功能
- **数据解析**: 自动解析文本格式的金融数据
- **特征工程**: 添加时间编码等衍生特征
- **数据标准化**: Z-score标准化，提高训练稳定性
- **序列创建**: 创建滑动窗口训练样本
- **反标准化**: 将预测结果转换回原始价格范围

### 数据格式要求
每行数据格式：
```
日期,星期\t开盘\t最高\t最低\t收盘\t涨幅%\t振幅%\t总手\t金额\t换手%\t成交次数
```

## ⚙️ 模型配置

### 预设配置
- **tiny**: 测试用小模型 (256维, 4层)
- **small**: 小型模型 (512维, 8层)  
- **base**: 基础模型 (1024维, 24层)
- **large**: 大型模型 (2048维, 32层)

### 自定义配置
```python
from src.config import ModelArgs

config = ModelArgs(
    d_model=512,           # 模型维度
    n_layers=8,            # Transformer层数
    n_heads=8,             # 注意力头数
    n_features=11,         # 输入特征数
    n_predictions=7,       # 预测时间点数
    max_seq_len=60,        # 最大序列长度
    # ... 其他参数
)
```

## 📈 训练建议

### 超参数调优
- **学习率**: 建议从 2e-4 开始
- **批次大小**: 根据GPU内存调整 (8-32)
- **序列长度**: 30-120个交易日
- **预测跨度**: 1-30个交易日

### 数据要求
- **最少数据量**: 建议至少500个交易日
- **数据质量**: 确保数据连续性，处理缺失值
- **特征选择**: 可根据具体需求调整输入特征

### 训练技巧
- 使用梯度裁剪防止梯度爆炸
- 采用余弦退火学习率调度
- 早停机制防止过拟合
- 定期验证模型性能

## 🔧 模型优化

### 内存优化
- MLA机制显著降低注意力计算复杂度
- 支持梯度检查点减少内存使用
- 可调整序列长度平衡性能和内存

### 推理优化
- 支持批量预测
- 可导出为ONNX格式加速推理
- 支持量化部署

## 📝 注意事项

1. **金融数据特性**: 模型专门针对金融时序数据设计
2. **风险提示**: 预测结果仅供参考，不构成投资建议
3. **数据质量**: 模型性能高度依赖输入数据质量
4. **市场变化**: 需要定期重训练适应市场变化

## 🤝 贡献

欢迎提交Issue和Pull Request来改进模型！

## 📄 许可证

MIT License

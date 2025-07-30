**策略网络数据流程**：
```python
# 第一阶段：训练价格预测网络
price_network = PriceTransformer(config)
price_network.train()  # 使用本文档的数据处理方法

# 第二阶段：策略网络使用价格网络的特征
price_network.eval()  # 冻结价格网络
with torch.no_grad():
    # 提取特征（不是原始数据处理）
    strategy_features = price_network.extract_features(financial_data)

# GRU策略网络
strategy_network = GRUStrategyNetwork(config)
positions = strategy_network(strategy_features, previous_state)
```

### 策略网络的数据来源

```python
# 策略网络不需要原始数据处理！
# 它使用价格网络的输出作为输入

# 第一阶段：价格网络训练（使用本文档的方法）
price_network = PriceTransformer(config)
price_network.train_with_processed_data()  # 使用上述数据处理

# 第二阶段：策略网络使用价格网络特征
price_network.eval()  # 冻结
strategy_features = price_network.extract_features(data)  # 提取特征
strategy_network = GRUStrategyNetwork(config)
positions = strategy_network(strategy_features, prev_state)
```
## ?? 故障排除

### 常见问题

### 梯度传播问题解决

#### 原问题分析
```python
# 原问题：argmax不可导，阻断梯度传播
expected_position = position_logits.argmax()  # ? 梯度阻断
gru_input = cat([features, expected_position])
strategy_state = gru_cell(gru_input, strategy_state)

# 实际梯度路径：
# final_loss → position_logits ?
# position_logits → expected_position ? (argmax阻断)
# expected_position → gru_input ?
# gru_input → strategy_state ?
```

#### 解决方案
```python
# 价格网络：标准监督学习
price_loss = mse_loss(price_pred, price_target)
price_loss.backward()  # 梯度完整传播


```

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
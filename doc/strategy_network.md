# 策略网络：基于收益的强化学习
strategy_loss = -relative_return + risk_cost + opportunity_cost
strategy_loss.backward()  # 只优化策略参数



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

## ? 性能指标

模型会自动计算多种评估指标：

- **累计收益率**: 策略的总收益表现
- **信息比率**: 超额收益的风险调整指标
- **夏普比率**: 收益风险比
- **最大回撤**: 最大损失幅度
- **机会成本**: 错失收益的量化
- **风险惩罚**: 波动率和回撤的综合

### 4. GRU策略网络

#### 递归状态更新机制
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

#### 可微分仓位预测
```python
class GumbelSoftmaxPositionHead(nn.Module):
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.linear(x)  # [batch_size, 11] (0-10仓位)
        
        if self.training:
            # 训练时：Gumbel-Softmax，可微分
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
            y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
            
            if hard:
                # 直通估计器
                y_hard = torch.zeros_like(y)
                y_hard.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
                y = (y_hard - y).detach() + y
                
            # 计算期望仓位
            position_values = torch.arange(11, device=x.device, dtype=torch.float32)
            positions = torch.sum(y * position_values, dim=-1, keepdim=True)
        else:
            # 推理时：直接argmax，离散
            positions = logits.argmax(dim=-1, keepdim=True).float()
            
        return {
            'logits': logits,
            'positions': positions,
            'probabilities': F.softmax(logits, dim=-1)
        }
```

### 2. 策略损失函数
```python
class StrategyLoss(nn.Module):
    def forward(self, position_predictions, next_day_returns):
        total_loss = 0
        
        for b in range(batch_size):
            positions = position_predictions[b, :, 0]  # [seq_len]
            returns = next_day_returns[b, :]           # [seq_len]
            
            # 1. 判断市场状态
            market_type = self.market_classifier.classify_market(returns)
            
            # 2. 计算相对基准收益
            relative_return_loss = self._calculate_relative_return_loss(
                positions, returns, market_type
            )
            
            # 3. 计算风险成本
            risk_cost = self._calculate_risk_cost(positions, returns)
            
            # 4. 计算机会成本
            opportunity_cost = self._calculate_opportunity_cost(
                positions, returns, market_type
            )
            
            # 5. 综合损失
            sample_loss = (
                self.relative_return_weight * relative_return_loss +
                self.risk_cost_weight * risk_cost +
                self.opportunity_cost_weight * opportunity_cost
            )
            
            total_loss += sample_loss
            
        return total_loss / batch_size
```

### 3. 市场分类算法
```python
def classify_market(self, returns):
    # 1. 统计特征
    mean_return = torch.mean(returns)
    volatility = torch.std(returns)

    # 2. 技术指标
    ma_short = torch.mean(returns[-5:])
    ma_long = torch.mean(returns[-10:])

    # 3. 投票机制
    votes = [
        self._simple_classifier(mean_return),
        self._technical_classifier(ma_short, ma_long),
        self._adaptive_classifier(returns)
    ]

    return self._majority_vote(votes)
```
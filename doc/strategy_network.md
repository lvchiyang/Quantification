# �������磺���������ǿ��ѧϰ
strategy_loss = -relative_return + risk_cost + opportunity_cost
strategy_loss.backward()  # ֻ�Ż����Բ���



### ��Ϣ������ʧ
����˴�ͳ�����ڲ�ͬ�г����������۲���ƽ�����⣺

```python
# �Զ�ѡ���׼
if market_type == 'bull':
    benchmark = buy_and_hold_strategy()    # �����ֱȽ�
elif market_type == 'bear':
    benchmark = conservative_strategy()    # �뱣�ز��ԱȽ�
else:
    benchmark = momentum_strategy()        # �붯�����ԱȽ�

# ������Ϣ����
information_ratio = excess_return_mean / excess_return_std
loss = -information_ratio + opportunity_cost + risk_penalty
```

### ����Ԥ��
```python
from src.strategy_network.gru_strategy import GRUStrategyNetwork
from src.price_prediction.price_transformer import PriceTransformer

# ����Ԥѵ���ļ۸�����
price_network = PriceTransformer(config)
price_network.load_state_dict(torch.load('best_price_network.pth'))

# ������������
strategy_network = GRUStrategyNetwork(config)

# ��ȡ������Ԥ���λ
with torch.no_grad():
    features = price_network.extract_features(financial_data)

positions = strategy_network.forward_sequence(features)
print(f"��λ����: {positions['position_output']['positions']}")
```

## ? ����ָ��

ģ�ͻ��Զ������������ָ�꣺

- **�ۼ�������**: ���Ե����������
- **��Ϣ����**: ��������ķ��յ���ָ��
- **���ձ���**: ������ձ�
- **���س�**: �����ʧ����
- **����ɱ�**: ��ʧ���������
- **���ճͷ�**: �����ʺͻس����ۺ�

### 4. GRU��������

#### �ݹ�״̬���»���
```python
# �ݶȴ���·��
final_loss �� position_logits[19] �� strategy_state[19]
          �� position_logits[18] �� strategy_state[18]
          �� ... �� position_logits[0]

# �ڴ��Ż�
for day in range(20):
    # �����ݶȵĲ���
    position = model.predict_position(features[day], strategy_state)
    strategy_state = model.update_state(strategy_state, position)

    # �������ݶȵĲ���
    with torch.no_grad():
        portfolio_value *= (1 + position * returns[day])
```

#### ��΢�ֲ�λԤ��
```python
class GumbelSoftmaxPositionHead(nn.Module):
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.linear(x)  # [batch_size, 11] (0-10��λ)
        
        if self.training:
            # ѵ��ʱ��Gumbel-Softmax����΢��
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
            y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
            
            if hard:
                # ֱͨ������
                y_hard = torch.zeros_like(y)
                y_hard.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
                y = (y_hard - y).detach() + y
                
            # ����������λ
            position_values = torch.arange(11, device=x.device, dtype=torch.float32)
            positions = torch.sum(y * position_values, dim=-1, keepdim=True)
        else:
            # ����ʱ��ֱ��argmax����ɢ
            positions = logits.argmax(dim=-1, keepdim=True).float()
            
        return {
            'logits': logits,
            'positions': positions,
            'probabilities': F.softmax(logits, dim=-1)
        }
```

### 2. ������ʧ����
```python
class StrategyLoss(nn.Module):
    def forward(self, position_predictions, next_day_returns):
        total_loss = 0
        
        for b in range(batch_size):
            positions = position_predictions[b, :, 0]  # [seq_len]
            returns = next_day_returns[b, :]           # [seq_len]
            
            # 1. �ж��г�״̬
            market_type = self.market_classifier.classify_market(returns)
            
            # 2. ������Ի�׼����
            relative_return_loss = self._calculate_relative_return_loss(
                positions, returns, market_type
            )
            
            # 3. ������ճɱ�
            risk_cost = self._calculate_risk_cost(positions, returns)
            
            # 4. �������ɱ�
            opportunity_cost = self._calculate_opportunity_cost(
                positions, returns, market_type
            )
            
            # 5. �ۺ���ʧ
            sample_loss = (
                self.relative_return_weight * relative_return_loss +
                self.risk_cost_weight * risk_cost +
                self.opportunity_cost_weight * opportunity_cost
            )
            
            total_loss += sample_loss
            
        return total_loss / batch_size
```

### 3. �г������㷨
```python
def classify_market(self, returns):
    # 1. ͳ������
    mean_return = torch.mean(returns)
    volatility = torch.std(returns)

    # 2. ����ָ��
    ma_short = torch.mean(returns[-5:])
    ma_long = torch.mean(returns[-10:])

    # 3. ͶƱ����
    votes = [
        self._simple_classifier(mean_return),
        self._technical_classifier(ma_short, ma_long),
        self._adaptive_classifier(returns)
    ]

    return self._majority_vote(votes)
```
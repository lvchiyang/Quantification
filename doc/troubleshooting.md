## ?? �����ų�

### ��������

### �ݶȴ���������

#### ԭ�������
```python
# ԭ���⣺argmax���ɵ�������ݶȴ���
expected_position = position_logits.argmax()  # ? �ݶ����
gru_input = cat([features, expected_position])
strategy_state = gru_cell(gru_input, strategy_state)

# ʵ���ݶ�·����
# final_loss �� position_logits ?
# position_logits �� expected_position ? (argmax���)
# expected_position �� gru_input ?
# gru_input �� strategy_state ?
```

#### �������
```python
# �۸����磺��׼�ලѧϰ
price_loss = mse_loss(price_pred, price_target)
price_loss.backward()  # �ݶ���������


```

**Q: ѵ��ʱ�ڴ治�㣿**
```python
# �����������С���δ�С
config.batch_size = 1  # ��Ĭ�ϵ�4���ٵ�1
config.strategy_state_dim = 64  # ��С״̬ά��
```

**Q: �ݶ���ʧ��ը��**
```python
# �������������ѧϰ�ʺ��ݶȲü�
optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # ����ѧϰ��
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # �ݶȲü�
```

**Q: ģ�Ͳ�������**
```python
# ���������������ݺ���ʧȨ��
config.information_ratio_weight = 0.5  # ������Ϣ����Ȩ��
config.opportunity_cost_weight = 0.05   # ���ͻ���ɱ�Ȩ��
```

### ���Լ���
```python
# 1. ����ݶ���
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.6f}")

# 2. ���״̬�仯
print(f"״̬�仯: {torch.mean(torch.abs(new_state - old_state)).item():.6f}")

# 3. ��֤�г�����
market_type = model.market_classifier.classify_market(returns)
print(f"�г�����: {market_type}")
```
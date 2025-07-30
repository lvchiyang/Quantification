**����������������**��
```python
# ��һ�׶Σ�ѵ���۸�Ԥ������
price_network = PriceTransformer(config)
price_network.train()  # ʹ�ñ��ĵ������ݴ�����

# �ڶ��׶Σ���������ʹ�ü۸����������
price_network.eval()  # ����۸�����
with torch.no_grad():
    # ��ȡ����������ԭʼ���ݴ���
    strategy_features = price_network.extract_features(financial_data)

# GRU��������
strategy_network = GRUStrategyNetwork(config)
positions = strategy_network(strategy_features, previous_state)
```

### ���������������Դ

```python
# �������粻��Ҫԭʼ���ݴ���
# ��ʹ�ü۸�����������Ϊ����

# ��һ�׶Σ��۸�����ѵ����ʹ�ñ��ĵ��ķ�����
price_network = PriceTransformer(config)
price_network.train_with_processed_data()  # ʹ���������ݴ���

# �ڶ��׶Σ���������ʹ�ü۸���������
price_network.eval()  # ����
strategy_features = price_network.extract_features(data)  # ��ȡ����
strategy_network = GRUStrategyNetwork(config)
positions = strategy_network(strategy_features, prev_state)
```
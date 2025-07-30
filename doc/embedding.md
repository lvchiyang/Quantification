# ��������Ƕ��� - ��������˵��

##  ��Ҫ�޸ģ��ɽ����������ɽ���������

���� `doc/data.md` ��Ҫ��**�ɽ�����**�Ѵ��г������������ɽ��������顣

###  �޸ĺ���������飨20ά��

| ������ | ������Χ | ά�� | �����б� | ˵�� |
|--------|----------|------|----------|------|
| **ʱ������** | [0-2] | 3ά | �¡��ա����� | ����ʱ����Ϣ |
| **�۸�����** | [3-6] | 4ά | open_rel, high_rel, low_rel, close_rel | OHLC��Լ۸� |
| **�ɽ�������** | [7-11] | 5ά | volume_rel, volume_change, amount_rel, amount_change, **�ɽ�����** | ���׻�Ծ��ָ�� |
| **�г�����** | [12-15] | 4ά | �Ƿ�, ���, ����%, **price_median** | �г�����ָ�� |
| **��������** | [16-19] | 4ά | big_order_activity, chip_concentration, market_sentiment, price_volume_sync | �߼�����ָ�� |

###  ��Ҫ�仯

#### �仯ǰ���ɰ汾��
```python
# �۸����� (5ά): price_median, open_rel, high_rel, low_rel, close_rel
# �ɽ������� (4ά): volume_rel, volume_change, amount_rel, amount_change  
# �г����� (4ά): �Ƿ�, ���, ����%, �ɽ�����
```

#### �仯���°汾��
```python
# �۸����� (4ά): open_rel, high_rel, low_rel, close_rel
# �ɽ������� (5ά): volume_rel, volume_change, amount_rel, amount_change, �ɽ�����
# �г����� (4ά): �Ƿ�, ���, ����%, price_median
```

###  �޸�ԭ��

1. **�����߼�������**
   - �ɽ�������ɽ���������أ�����ӳ�г����׻�Ծ��
   - �ɽ�����/�ɽ��� = ƽ��ÿ�ʽ�����������Ҫ��volume����ָ��
   - �󵥡�С���Ľ���ģʽ����ͨ���ɽ����ͳɽ������Ĺ�ϵ����

2. **���ݴ���һ����**
   - �ɽ�������Ҫ��׼��������ͬ��Ʊ���������
   - ��volume�����Ĵ���ʽһ��
   - ����ҪBatch�����ڱ�׼��

3. **���������Ż�**
   - price_median��Ϊ�۸��׼�����ʺϷ����г�������
   - �ɽ������������ڰ��������Ľ��׻�Ծ����Ϣ
   - �������ڲ��������ͳһ

###  ����ʵ��

```python
class FinancialEmbedding(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        
        group_dim = d_model // 5
        
        # ʱ������ (3ά)
        self.time_embed = nn.Linear(3, group_dim)
        
        # �۸����� (4ά) - �Ƴ�price_median
        self.price_embed = nn.Linear(4, group_dim)
        
        # �ɽ������� (5ά) - ��ӳɽ�����
        self.volume_embed = nn.Linear(5, group_dim)
        
        # �г����� (4ά) - ���price_median
        self.market_embed = nn.Linear(4, group_dim)
        
        # �������� (4ά)
        self.financial_embed = nn.Linear(4, group_dim)
    
    def forward(self, x):
        # ��������
        time_features = x[..., 0:3]       # [0-2]
        price_features = x[..., 3:7]      # [3-6]
        volume_features = x[..., 7:12]    # [7-11] �����ɽ�����
        market_features = x[..., 12:16]   # [12-15] ����price_median
        financial_features = x[..., 16:20] # [16-19]
        
        # ����Ƕ��
        time_emb = self.time_embed(time_features)
        price_emb = self.price_embed(price_features)
        volume_emb = self.volume_embed(volume_features)
        market_emb = self.market_embed(market_features)
        financial_emb = self.financial_embed(financial_features)
        
        # ƴ��
        return torch.cat([time_emb, price_emb, volume_emb, market_emb, financial_emb], dim=-1)
```

###  ��֤����

�������²�����֤�޸ģ�

```bash
cd src/price_prediction
python test_embedding_grouping.py
```

�������ݣ�
-  ����������ȷ��
-  �ɽ�������volume���еĴ���
-  ά�ȷ�����ȷ��
-  ��ͬ����ģʽ��embeddingЧ��

###  Ԥ��Ч��

1. **���õ�������ʾ**
   - �ɽ�����������������Ľ��׻�Ծ����Ϣ
   - �������ڲ�����Ը�ǿ��ѧϰЧ������

2. **������Ľ�������**
   - ���Ͻ��ڷ������߼�����
   - ����ģ����⽻����Ϊģʽ

3. **���ȶ���ѵ��**
   - ����������ֵ��Χ��ͳһ
   - ���ٲ�ͬ����������ֵ����

###  ����ļ�

- `src/price_prediction/embedding.py` - ��Ҫʵ��
- `src/price_prediction/test_embedding_grouping.py` - ��֤����
- `doc/data.md` - ���ݴ����ĵ���ԭʼ����

---

**ע��**������޸�Ӱ������ʹ�ø�embedding���ģ�ͣ���ȷ����ش���ͬ�����£�

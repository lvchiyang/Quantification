
# �������ݴ�����������

�������м�����Ľ������ݴ���������������й¶��ȷ��ѵ����Ԥ���һ���ԡ�

---

##  ��������

### �������ԭ��

####  ��������й¶
- **����**��ʹ��δ�����ݼ����׼��й¶δ����Ϣ
- **���**��ÿ��180�����ж��������׼
- **����**��ѵ����Ԥ��ʹ����ȫ��ͬ�Ĵ����߼�

####  ��ֵ�߶�ͳһ
- **���ֲ���**��1,018,000 ~ 123,584,000������121����
- **������**��200,000,000 ~ 25,000,000,000������125����
- **�۸����**��ę́1000Ԫ vs ���й�10Ԫ
- **�������**����������Ի�����

###  ��������ܹ�
```mermaid
graph TD
    A[ԭʼExcel�ļ�<br/>12��] --> B[���߻�����ϴ<br/>data_processor.py]
    B --> C[��׼������<br/>14��]
    C --> D[�������д���<br/>price_prediction/data_processor.py]
    D --> E[ѵ������<br/>180��20ά����]

    B --> B1[ʱ���в��]
    B --> B2[��ʽ��׼��]
    B --> B3[���ݹ���]

    D --> D1[���м��۸��׼]
    D --> D2[�����������]
    D --> D3[����ָ������]
```

---

##  ���ݽṹ˵��

### ԭʼ���ݸ�ʽ
**����λ��**��`��Ʊ����/��ҵ����/��Ʊ����.xls`

**ԭʼExcel�ļ��ṹ**��12�У���
```
ʱ��, ����, ���, ���, ����, �Ƿ�, ���, ����, ���, ����%, �ɽ�����, Unnamed: 11
```

**����ʾ��**��ę́ԭʼ���ݣ���
```
2001-08-27,һ, 34.51, 37.78, 32.85, 35.39, 2.55, 14.22, 1,410,347,000, 56.83, 927
```

### ��������ݸ�ʽ
**������ϴ��ṹ**��14�У���
```
��, ��, ��, ����, ����, ���, ���, ����, �Ƿ�, ���, ����, ���, ����%, �ɽ�����
```

**�����ʾ��**��
```
2001, 8, 27, 1, 34.51, 37.78, 32.85, 35.39, 2.55, 14.22, 1410347000, 56.83, 927
```

### ���������ṹ
**ģ����������**��20ά����
```python
# �� src/price_prediction/data_processor.py ��ʵ��
feature_columns = [
    'month', 'day', 'weekday',                    # ʱ������ (3ά)
    'price_median', 'open_rel', 'high_rel',       # �۸��׼ + OHLC���ֵ (5ά)
    'low_rel', 'close_rel',
    'change_pct', 'amplitude',                    # �۸�仯 (2ά)
    'volume_log', 'volume_rel',                   # �ɽ��������� + ���ֵ (2ά)
    'amount_log', 'amount_rel',                   # ������ + ���ֵ (2ά)
    'turnover_rate', 'trade_count',               # �г���Ծ�� (2ά)
    'big_order_activity', 'chip_concentration','market_sentiment', 'price_volume_sync'  # �������� (4ά)
]

# Ԥ��Ŀ�꣺δ�����ʱ����close_rel
prediction_targets = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]  # δ����N��
```

---

##  ���׶δ���ܹ�

---

##  ʹ��ʾ��

### �׶�1�����߻�����ϴ��data_processor.py��
```bash
# ���л���������ϴ
python data_processor.py

# ѡ��ѡ��1���������ݴ���
# ���룺ԭʼExcel�ļ���12�У�
# �������׼�����ݣ�14�У�
```
**Ŀ��**����ԭʼ���ݱ�׼��Ϊͳһ��ʽ����������������

**��������**��
1. **�ļ���ȡ**�����������䲻ͬExcel��ʽ
2. **ʱ����**��`2001-08-27,һ` �� `��, ��, ��, ����`
3. **��ʽ��׼��**��ɾ���ٷֺš������ŷָ���
4. **���ݹ���**�����ڹ��ˡ���Ч��������

**���**����׼���Ļ������ݣ�14�У�

```python
def offline_basic_cleaning(input_file, output_file):
    """���߻���������ϴ"""
    # 1. �ļ���ȡ
    df = read_excel_robust(input_file)

    # 2. ʱ���д���
    df = parse_time_columns(df)

    # 3. ��ʽ��׼��
    df = clean_data_formats(df)

    # 4. ���ݹ���
    df = apply_data_filter(df)

    # 5. �����׼������
    df.to_excel(output_file, index=False)
    return df
```


### �׶�2���������д���src/price_prediction/data_processor.py��

**Ŀ��**���ڴ���ѵ������ʱ�����������̣���������й¶

**��������**��
1. **���м��۸��׼**��ʹ��180�������ڵļ۸���λ��
2. **��Լ۸�����**��OHLC��������л�׼�ı�ֵ
3. **�ɽ�������**�������ڵ���Ա仯��ͳ������
4. **��������**���������������ݵļ���ָ��

**���**��ѵ���������������У�180��20ά��
---

##  ���м�����������

### 1. ���м��۸���������

```python
def process_sequence_price_features(sequence_df):
    """
    ��180�����н��м۸���������
    �ؼ���ʹ�����������ݼ����׼����������й¶
    """
    # ��������������OHLC����λ����Ϊ��׼
    ohlc_data = sequence_df[['����', '���', '���', '����']]
    all_prices = ohlc_data.values.flatten()
    all_prices = all_prices[~pd.isna(all_prices)]
    sequence_price_median = np.median(all_prices)

    # OHLC��������л�׼�ı�ֵ
    open_rel = sequence_df['����'] / sequence_price_median
    high_rel = sequence_df['���'] / sequence_price_median
    low_rel = sequence_df['���'] / sequence_price_median
    close_rel = sequence_df['����'] / sequence_price_median

    return {
        'price_median': sequence_price_median,  # ���м۸��׼
        'open_rel': open_rel.fillna(1.0),
        'high_rel': high_rel.fillna(1.0),
        'low_rel': low_rel.fillna(1.0),
        'close_rel': close_rel.fillna(1.0)
    }
```

### 2. ���м��ɽ���/����

```python
def process_sequence_volume_amount(sequence_df, col):
    """
    ��180�����еĳɽ���/�����д���
    ����1����������λ����׼ + ���ֵ
    ����2����������Ա仯��
    """
    values = pd.to_numeric(sequence_df[col], errors='coerce').fillna(0)

    # ����1����������λ����׼
    sequence_median = values.median()
    if sequence_median == 0:
        sequence_median = 1.0
    relative_values = values / sequence_median

    # ����2����Ա仯�ʣ�20�չ�����ֵ��
    rolling_mean = values.rolling(window=20, min_periods=1).mean()
    rolling_mean = rolling_mean.fillna(method='bfill').fillna(1.0)
    rolling_mean = rolling_mean.replace(0, 1.0)
    relative_change = (values - rolling_mean) / rolling_mean * 100

    return relative_values, relative_change.fillna(0.0)
```

### 3. ���м�������������
---

##  Transformer����Ľ�����������

**����רע�ڼ۸�Ԥ�����磨Transformer���������������**

| ������� | ������Ϣ | ���ͳ��� | ��Transformer�е����� |
|---------|----------|----------|----------------------|
| **���� / �ɽ�����** | ƽ��ÿ�ʳɽ����� �� **�󵥻�Ծ��** | >500��/�ʣ������춯 | **ʱ��ע����**��ʶ���������ʱ�� |
| **������ / ����** | ʵ����ͨ�̱仯 �� **���뼯�ж�** | �߻���+���������������Ե� | **����ģʽ**��180���ڵ��ʽ����� |
| **�ɽ����� / ������** | ɢ�������ȶ� �� **�г�����** | ��������+����������ɢ������ | **��������**��ʶ��ɢ���������� |
| **��� / �Ƿ�** | �۸񲨶�Ч�� �� **�г�����** | ��������Ƿ�����շ���� | **�۸�Ԥ��**��Ԥ��δ��7�첨�� |

>  **Transformer����**��ͨ��180��ĳ����У��ܹ���׽��Щ��ϵ��ʱ���ݱ�ģʽ��
```python
def process_sequence_financial_features(sequence_df):
    """
    ��180�����м����������
    �ؼ��������м�ֵ�ü�������ԭʼ��ֵ�ֲ�
    """
    # ����1���󵥻�Ծ�� = ���� / �ɽ�����
    big_order_activity = sequence_df['����'] / (sequence_df['�ɽ�����'] + 1e-6)

    # ����2�����뼯�ж� = ������ / ��׼������
    volume_mean = sequence_df['����'].rolling(30, min_periods=1).mean()
    volume_normalized = sequence_df['����'] / (volume_mean + 1e-6)
    chip_concentration = sequence_df['����%'] / (volume_normalized + 1e-6)

    # ����3���г����� = �Ƿ� * �������
    market_sentiment = sequence_df['�Ƿ�'] * sequence_df['���'] / 100

    # ����4������ͬ���� = �Ƿ����� * �ɽ����仯����
    price_direction = np.sign(sequence_df['�Ƿ�'])
    volume_change = sequence_df['����'].pct_change().fillna(0)
    volume_direction = np.sign(volume_change)
    price_volume_sync = price_direction * volume_direction

    # ��׼�������ü���ֵ
    return {
        'big_order_activity': standardize_without_clipping(big_order_activity),
        'chip_concentration': standardize_without_clipping(chip_concentration),
        'market_sentiment': standardize_without_clipping(market_sentiment),
        'price_volume_sync': price_volume_sync
    }

def standardize_without_clipping(series):
    """��׼�������ü���ֵ������������ֵ�ֲ�"""
    mean = series.mean()
    std = series.std() + 1e-6
    return (series - mean) / std
```



##  ����ͳһ���
### ��������������ֵ��Χ

```python
# �������18ά�����Ƿ�ͳһ����������
def check_feature_ranges(processed_features):
    """
    ��鴦�����������ֵ��Χ��ȷ��ͳһ����
    """
    feature_names = [
        '��', '��', '����',                           # ʱ������
        'price_median', 'open_rel', 'high_rel',       # �۸�����
        'low_rel', 'close_rel',
        '�Ƿ�', '���',   '����%',                     # �ٷֱ�����
        'volume_log', 'volume_rel',                   # �ɽ�������
        'amount_log', 'amount_rel',                   # �������
        '�ɽ�����',                                     # �г���Ծ��
        'big_order_activity', 'chip_concentration',   # ��������1,2
        'market_sentiment', 'price_volume_sync'       # ��������3,4
    ]

    ranges = {}
    for i, name in enumerate(feature_names):
        col_data = processed_features[:, i]
        ranges[name] = {
            'min': np.min(col_data),
            'max': np.max(col_data),
            'mean': np.mean(col_data),
            'std': np.std(col_data)
        }

    return ranges
```
---

##  ������֤��������֤

### ��֤���д�����

```python
def validate_sequence_processing():
    """��֤���д����Ƿ����������й¶"""
    # ��鲻ͬ���еļ۸��׼
    sequence1 = cleaned_data.iloc[0:180]
    sequence2 = cleaned_data.iloc[100:280]

    features1 = process_sequence_features(sequence1)
    features2 = process_sequence_features(sequence2)

    # ��ͬ����Ӧ���в�ͬ�ļ۸��׼
    print(f"����1�۸��׼: {features1['price_median']}")
    print(f"����2�۸��׼: {features2['price_median']}")

    # ���������Χ����Ӧ�ñ��ü���
    print(f"�󵥻�Ծ�ȷ�Χ: [{features1['big_order_activity'].min():.3f}, {features1['big_order_activity'].max():.3f}]")
```

### �����������

```python
def check_data_quality(sequences):
    """��鴦������ݵ�����"""
    print(f"��������: {len(sequences)}")
    print(f"������״: {sequences[0].shape}")  # Ӧ���� (180, 20)

    # ����Ƿ���NaNֵ
    has_nan = np.isnan(sequences).any()
    print(f"����NaNֵ: {has_nan}")

    # �����ֵ��Χ
    for i, feature_name in enumerate(feature_columns):
        feature_values = sequences[:, :, i].flatten()
        print(f"{feature_name}: [{feature_values.min():.3f}, {feature_values.max():.3f}]")
```

### Ԥ�ڵ���ֵ��Χ��20ά������

| ������ | ������ | Ԥ�ڷ�Χ | ����״̬ |
|--------|--------|----------|----------|
| **ʱ��** | �� | [1, 12] |  ͳһ |
| | �� | [1, 31] |  ͳһ |
| | ���� | [1, 7] |  ͳһ |
| **�۸�** | price_median | [252�չ�����λ��] |  ��Batch��׼�� |
| | OHLC_rel | [0.95, 1.05] |  ͳһ |
| **�۸�仯** | �Ƿ� | [-10, 10] |  ͳһ |
| | ��� | [0, 20] |  ͳһ |
| **�ɽ���** | volume_log | [10, 20] |  ͳһ |
| | volume_rel | [-50, 50] |  ͳһ |
| **���** | amount_log | [15, 25] |  ͳһ |
| | amount_rel | [-50, 50] |  ͳһ |
| **�г�** | ����% | [0, 50] |  ͳһ |
| | �ɽ����� | [100, 10000] |  ���׼�� |
| **����1** | big_order_activity | [-3, 3] |  ͳһ |
| **����2** | chip_concentration | [-3, 3] |  ͳһ |
| **����3** | market_sentiment | [-3, 3] |  ͳһ |
| **����4** | price_volume_sync | [-1, 1] |  ͳһ |

### ��ҪBatch��׼��������

```python
# ��Щ�������п��Ʊ���죬��ҪBatch�����ڱ�׼��
batch_norm_needed = [
    'price_median',    # ��ͬ��Ʊ�۸��׼��ͬ
    '�ɽ�����'         # ��ͬ��Ʊ�ɽ�����������ͬ
]

# ��BatchSequenceNorm�л��Զ�������Щ����
```

##  �����������ֵ

### ����ĺ�������

1. ** ����й¶����**
   - **����**����ͳ����ʹ��δ�����ݼ����׼
   - **���**�����м�����ÿ��180�����ж��������׼
   - **��ֵ**��ȷ��ģ�ͷ�����������������

2. ** Ԥ��һ��������**
   - **����**��ѵ����Ԥ��ʹ�ò�ͬ�Ĵ����߼�
   - **���**��ѵ����Ԥ��ʹ����ȫ��ͬ�����м�����
   - **��ֵ**��ȷ��ģ����ʵ��Ӧ���е��ȶ���

3. ** ��ֵ�߶Ȳ���**
   - **����**����ͬ��Ʊ��۸񡢳ɽ�������޴�
   - **���**����������Ի�����ͳһ����
   - **��ֵ**��֧�ֿ��Ʊ��ģ�����ģ��ͨ����

4. ** �������̸�����**
   - **����**����Ҫ���ֽ��������ͬʱͳһ����
   - **���**�����ü���ֵ������ԭʼ��ֵ�ֲ�
   - **��ֵ**�������������г���Ϣ�����Ԥ�⾫��

### �������µ�

- **���м���׼����**����������й¶�Ĺؼ�����
- **���׶δ���ܹ�**��������ϴ + ������������
- **�޼�ֵ�ü�**������ԭʼ��ֵ�ֲ���������
- **������������**�����ڽ������۵��������
##  ���հ����д�����ʹ��ָ��

###  ���ٿ�ʼ

```python
from sequence_processor import SequenceProcessor, PriceDataset, predict_stock_price

# 1. ����ѵ�����ݼ�
dataset = PriceDataset("processed_data_2025-07-30/��Ʊ����")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. ѵ��ģ��
for inputs, targets in dataloader:
    # inputs: [batch_size, 180, 20]
    # targets: [batch_size, 10] - δ��10��ʱ���
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# 3. Ԥ��
predictions = predict_stock_price(model, "ę́.xlsx")
# predictions: [10] - δ����1,2,3,4,5,10,15,20,25,30���Ԥ��ֵ
```

###  ��������

 **��������й¶**��ÿ��180�����ж��������׼
 **Ԥ��һ����**��ѵ����Ԥ��ʹ����ͬ�߼�
 **20ά����**�������Ľ�����������
 **10��Ԥ���**��δ����1,2,3,4,5,10,15,20,25,30��
 **�޼�ֵ�ü�**������ԭʼ��ֵ�ֲ�

###  ��������

```
ԭʼ����(12��) �� ������ϴ(14��) �� ���д���(180��20) �� ģ��ѵ�� �� Ԥ��(10ά)
```

---

##  �ܽ�

������հ�������ݴ�����ʵ���˴�ԭʼExcel�ļ���ѵ���������ݵĶ˵��˴���

###  ���׶μܹ�
 **�׶�1-���߻�����ϴ**��ʱ���֡���ʽ��׼�������ܹ���
 **�׶�2-�������д���**�����м��������̡���������й¶

###  �ؼ������Ľ�
 **���м���׼**��ÿ��180�����ж�������۸��׼
 **�޼�ֵ�ü�**������ԭʼ��ֵ�ֲ�
 **Ԥ��һ����**��ѵ����Ԥ��ʹ����ȫ��ͬ�Ĵ����߼�
 **����������֤**����ʽͳһ���쳣������Ե�������

###  ����ĺ�������
1. **����й¶** �� ���м�������ʹ��δ����Ϣ
2. **Ԥ�ⲻ����** �� ѵ����Ԥ���߼���ȫһ��
3. **��������** �� ��������Ի�����
4. **��Ϣ��ʧ** �� ˫�ش����������ԣ����ü���ֵ

��������ȱ����˽������ݵľ���ѧ���壬�����������ѧϰģ�͵ļ���Ҫ��**����Ҫ���Ǳ���������й¶��ȷ����ģ�͵ķ�������**��



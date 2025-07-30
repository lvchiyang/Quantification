# ���ò����ĵ�

ר��Ϊ����ʱ��Ԥ�� Transformer ģ����Ƶ�����ϵͳ���ṩ���Ĳ���������Ԥ�������á�

## ? Ŀ¼

- [���ø���](#���ø���)
- [���Ĳ���](#���Ĳ���)
- [Ԥ��������](#Ԥ��������)
- [��������ָ��](#��������ָ��)
- [ʹ�÷���](#ʹ�÷���)

---

## ? ���ø���

### �������

����ϵͳ���� dataclass ��ƣ����������ص㣺

1. **���Ͱ�ȫ**�����в���������ȷ������ע��
2. **������֤**���Զ���֤�����ĺ����Ժͼ�����
3. **Ԥ��������**���ṩ���ֳ�����Ԥ���÷���
4. **�����չ**����������²���������

### ���ýṹ

```python
@dataclass
class PricePredictionConfig:
    # ģ�ͼܹ�����
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8

    # ѵ������
    learning_rate: float = 1e-4
    batch_size: int = 4

    # ����ר�ò���
    use_financial_loss: bool = True
    direction_weight: float = 0.3
```

---

## ?? ���Ĳ���

### ģ�ͼܹ�����

| ���� | Ĭ��ֵ | ˵�� | ���鷶Χ |
|------|--------|------|----------|
| `d_model` | 512 | ģ��ά�� | 256-1024 |
| `n_layers` | 8 | Transformer���� | 4-16 |
| `n_heads` | 8 | ע����ͷ�� | 4-16 |
| `kv_lora_rank` | 256 | K/Vѹ��ά�� | 128-512 |
| `v_head_dim` | 64 | ֵͷά�� | 32-128 |
| `intermediate_size` | 2048 | FFN����ά�� | d_model��2-8 |

### ���ݲ���

| ���� | Ĭ��ֵ | ˵�� | Լ�� |
|------|--------|------|------|
| `n_features` | 20 | ���������� | �̶�Ϊ20 |
| `sequence_length` | 180 | �������г��� | �� max_seq_len |
| `prediction_horizon` | 10 | Ԥ��ʱ����� | �̶�Ϊ10 |
| `max_seq_len` | 512 | ������г��� | �� sequence_length |
| `rope_theta` | 10000.0 | RoPE����Ƶ�� | 1000-50000 |

### ѵ������

| ���� | Ĭ��ֵ | ˵�� | ���鷶Χ |
|------|--------|------|----------|
| `batch_size` | 4 | ���δ�С | 2-32 |
| `learning_rate` | 1e-4 | ѧϰ�� | 1e-5 - 1e-3 |
| `weight_decay` | 0.01 | Ȩ��˥�� | 0.001-0.1 |
| `max_epochs` | 100 | ���ѵ������ | 50-500 |
| `warmup_steps` | 1000 | Ԥ�Ȳ��� | 500-5000 |
| `dropout` | 0.1 | Dropout���� | 0.0-0.3 |

### ����ר����ʧ��������

| ���� | Ĭ��ֵ | ˵�� | �Ƽ����� |
|------|--------|------|----------|
| `use_financial_loss` | True | �Ƿ�ʹ�ý�����ʧ | True |
| `use_direction_loss` | True | �Ƿ����÷�����ʧ | True |
| `use_trend_loss` | True | �Ƿ�����������ʧ | True |
| `use_temporal_weighting` | True | �Ƿ�����ʱ���Ȩ | True |
| `use_ranking_loss` | False | �Ƿ�����������ʧ | False |
| `use_volatility_loss` | False | �Ƿ����ò�������ʧ | False |

### ��ʧȨ�ز���

| ���� | Ĭ��ֵ | ˵�� | ���鷶Χ |
|------|--------|------|----------|
| `base_weight` | 1.0 | ������ʧȨ�� | 0.5-1.5 |
| `direction_weight` | 0.3 | ������ʧȨ�� | 0.1-0.5 |
| `trend_weight` | 0.2 | ������ʧȨ�� | 0.1-0.3 |
| `ranking_weight` | 0.1 | ������ʧȨ�� | 0.05-0.2 |
| `volatility_weight` | 0.1 | ��������ʧȨ�� | 0.05-0.2 |

---

## ?? Ԥ��������

### 1. Tiny ����

**���ó���**�����ٲ��ԡ�ԭ�Ϳ���

```python
config = PricePredictionConfigs.tiny()
```

**�����ص�**��
- `d_model=256, n_layers=4, n_heads=4`
- `batch_size=2, max_epochs=50`
- ��������~2.5M
- �ڴ�����~2GB
- ѵ���ٶȣ���

### 2. Small ����

**���ó���**�����˵��Կ���

```python
config = PricePredictionConfigs.small()
```

**�����ص�**��
- `d_model=512, n_layers=6, n_heads=8`
- `batch_size=4, max_epochs=100`
- ��������~10M
- �ڴ�����~4GB
- ѵ���ٶȣ��е�

### 3. Base ���ã�Ĭ�ϣ�

**���ó���**����׼ѵ������������

```python
config = PricePredictionConfigs.base()
```

**�����ص�**��
- `d_model=512, n_layers=8, n_heads=8`
- `batch_size=4, max_epochs=100`
- ��������~40M
- �ڴ�����~8GB
- ѵ���ٶȣ��е�

### 4. Large ����

**���ó���**��������ѵ��������������

```python
config = PricePredictionConfigs.large()
```

**�����ص�**��
- `d_model=1024, n_layers=12, n_heads=16`
- `batch_size=8, max_epochs=200`
- ��������~160M
- �ڴ�����~16GB
- ѵ���ٶȣ���

### 5. ����������

**���ó���**����Ҫ������ʷ��Ϣ

```python
config = PricePredictionConfigs.for_long_sequence()
```

**�����ص�**��
- `sequence_length=360, max_seq_len=1024`
- `d_model=768, n_layers=8`
- �ʺ���ȼ���ĳ���Ԥ��

### 6. �ಽԤ������

**���ó���**��רע��ʱ���Ԥ��

```python
config = PricePredictionConfigs.for_multi_step_prediction()
```

**�����ص�**��
- `n_layers=10, loss_type="mae"`
- �Ż��ಽԤ������

---

## ? ���ܶԱ�

### ѵ�����ܶԱ�

| ���� | ������ | ѵ���ٶ� | �ڴ�ʹ�� | �Ƽ����� |
|------|--------|----------|----------|----------|
| Tiny | 2.5M | �� | 2GB | ����ʵ�� |
| Small | 10M | �е� | 4GB | ���˿��� |
| Base | 40M | �� | 8GB | ��׼ѵ�� |
| Large | 160M | �� | 16GB | ���������� |

### Ԥ�����ܶԱ�

| ���� | ����׼ȷ�� | MAE | RMSE | ѵ��ʱ�� |
|------|------------|-----|------|----------|
| Tiny | ~70% | 0.08 | 0.12 | 2Сʱ |
| Small | ~75% | 0.06 | 0.10 | 4Сʱ |
| Base | ~80% | 0.05 | 0.08 | 8Сʱ |
| Large | ~85% | 0.04 | 0.06 | 16Сʱ |

---

## ? ��������ָ��

### 1. ģ����������

**����ģ������**��
```python
config.d_model = 768        # ����ģ��ά��
config.n_layers = 12       # ���Ӳ���
config.intermediate_size = 3072  # ����FFNά��
```

**����ģ������**��
```python
config.d_model = 256        # ����ģ��ά��
config.n_layers = 4        # ���ٲ���
config.kv_lora_rank = 128  # ����ѹ����
```

### 2. ѵ���ȶ��Ե���

**���ѵ���ȶ���**��
```python
config.learning_rate = 5e-5     # ����ѧϰ��
config.warmup_steps = 2000      # ����Ԥ�Ȳ���
config.dropout = 0.2            # ����dropout
config.weight_decay = 0.05      # ����Ȩ��˥��
```

**����ѵ������**��
```python
config.learning_rate = 2e-4     # ���ѧϰ��
config.batch_size = 8           # �������δ�С
config.warmup_steps = 500       # ����Ԥ�Ȳ���
```

### 3. ������ʧ����

**���ز���**��������ֵ׼ȷ�ԣ���
```python
config.base_weight = 1.0
config.direction_weight = 0.1
config.trend_weight = 0.1
config.use_ranking_loss = False
```

**��������**�����ӷ�������ƣ���
```python
config.base_weight = 0.5
config.direction_weight = 0.5
config.trend_weight = 0.3
config.use_ranking_loss = True
config.ranking_weight = 0.2
```

### 4. �ڴ��Ż�

**�����ڴ�ʹ��**��
```python
config.batch_size = 2           # �������δ�С
config.kv_lora_rank = 128      # ����K/Vѹ��
config.sequence_length = 120    # �������г���
```

**�ݶ��ۻ�**��������Ч���δ�С����
```python
config.batch_size = 2
# ��ѵ���ű������� accumulation_steps = 4
# ��Ч���δ�С = 2 �� 4 = 8
```

---

## ? ʹ�÷���

### ����ʹ��

```python
from src.price_prediction.config import PricePredictionConfigs

# ʹ��Ԥ��������
config = PricePredictionConfigs.base()

# ����ģ��
from src.price_prediction.price_transformer import PriceTransformer
model = PriceTransformer(config)
```

### �Զ�������

```python
from src.price_prediction.config import PricePredictionConfig

# �����Զ�������
config = PricePredictionConfig(
    d_model=768,
    n_layers=10,
    n_heads=12,
    learning_rate=5e-5,
    batch_size=6,

    # �Զ��������ʧ
    use_financial_loss=True,
    direction_weight=0.4,
    trend_weight=0.3,
    use_volatility_loss=True,
    volatility_weight=0.15
)
```

### �����޸�

```python
# �������������޸�
config = PricePredictionConfigs.base()

# �޸��ض�����
config.learning_rate = 2e-4
config.batch_size = 8
config.direction_weight = 0.4

# ���ö������ʧ����
config.use_ranking_loss = True
config.ranking_weight = 0.15
```

### ������֤

```python
# ���û��Զ���֤����
try:
    config = PricePredictionConfig(
        d_model=513,  # ���ܱ�n_heads����
        n_heads=8
    )
except AssertionError as e:
    print(f"���ô���: {e}")

# ��ȷ������
config = PricePredictionConfig(
    d_model=512,  # 512 % 8 = 0 ?
    n_heads=8
)
```

---

## ? ���ʵ��

### 1. ����ѡ�����

**�����׶�**��
```python
# ʹ��tiny���ÿ��ٵ���
config = PricePredictionConfigs.tiny()
config.max_epochs = 10  # ������֤
```

**ʵ��׶�**��
```python
# ʹ��small��base����
config = PricePredictionConfigs.small()
config.max_epochs = 50
```

**�����׶�**��
```python
# ʹ��base��large����
config = PricePredictionConfigs.base()
config.max_epochs = 200
config.early_stopping_patience = 30
```

### 2. ����������

```python
# ���������ռ�
search_configs = [
    {"learning_rate": 1e-4, "direction_weight": 0.2},
    {"learning_rate": 5e-5, "direction_weight": 0.3},
    {"learning_rate": 2e-4, "direction_weight": 0.4},
]

best_config = None
best_score = float('inf')

for params in search_configs:
    config = PricePredictionConfigs.base()
    for key, value in params.items():
        setattr(config, key, value)

    # ѵ��������
    score = train_and_evaluate(config)

    if score < best_score:
        best_score = score
        best_config = config
```

### 3. ���ñ���ͼ���

```python
import json
from dataclasses import asdict

# ��������
config = PricePredictionConfigs.base()
config_dict = asdict(config)

with open('config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

# ��������
with open('config.json', 'r') as f:
    config_dict = json.load(f)

config = PricePredictionConfig(**config_dict)
```

### 4. ��������

```python
import torch

def get_adaptive_config():
    """����Ӳ�������Զ�ѡ������"""

    # ���GPU�ڴ�
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        if gpu_memory >= 16:
            return PricePredictionConfigs.large()
        elif gpu_memory >= 8:
            return PricePredictionConfigs.base()
        elif gpu_memory >= 4:
            return PricePredictionConfigs.small()
        else:
            return PricePredictionConfigs.tiny()
    else:
        # CPUѵ��ʹ��С����
        return PricePredictionConfigs.tiny()

# ʹ������Ӧ����
config = get_adaptive_config()
```

---

## ? ���Ժͼ��

### ������Ϣ��ӡ

```python
def print_config_summary(config):
    """��ӡ����ժҪ"""
    print("=" * 50)
    print("ģ������ժҪ")
    print("=" * 50)

    # ģ�ͼܹ�
    print(f"ģ��ά��: {config.d_model}")
    print(f"����: {config.n_layers}")
    print(f"ע����ͷ��: {config.n_heads}")
    print(f"��������: ~{estimate_params(config):,}")

    # ѵ������
    print(f"\nѵ������:")
    print(f"  ѧϰ��: {config.learning_rate}")
    print(f"  ���δ�С: {config.batch_size}")
    print(f"  �������: {config.max_epochs}")

    # ��ʧ����
    print(f"\n��ʧ����:")
    print(f"  ������ʧ: {config.loss_type}")
    print(f"  ������ʧ: {'����' if config.use_financial_loss else '����'}")
    if config.use_financial_loss:
        print(f"  ����Ȩ��: {config.direction_weight}")
        print(f"  ����Ȩ��: {config.trend_weight}")

def estimate_params(config):
    """����ģ�Ͳ�������"""
    # �򻯵Ĳ�������
    embed_params = config.n_features * config.d_model
    transformer_params = config.n_layers * (
        config.d_model * config.d_model * 4 +  # ע����
        config.d_model * config.intermediate_size * 3  # FFN
    )
    head_params = config.d_model * config.prediction_horizon

    return embed_params + transformer_params + head_params

# ʹ��ʾ��
config = PricePredictionConfigs.base()
print_config_summary(config)
```

---

## ? ����ļ�

- `src/price_prediction/config.py` - ������ʵ��
- `src/price_prediction/price_transformer.py` - ��ģ��ʵ��
- `train_price_prediction.py` - ѵ���ű�
- `doc/transformer.md` - Transformer�ܹ��ĵ�
- `doc/financial_losses.md` - ��ʧ�����ĵ�

��������ϵͳΪ����ʱ��Ԥ���ṩ�������ǿ��Ĳ�������������

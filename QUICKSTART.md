# ��������ָ��

���ָ�Ͻ�������������� Decoder-only Transformer with MLA ��Ŀ��

## ǰ��Ҫ��

- Python 3.8+
- PyTorch 2.1+
- 8GB+ RAM (�Ƽ�)
- GPU (��ѡ�����Ƽ�����ѵ��)

## 5���ӿ�������

### 1. ��װ����

```bash
pip install torch einops transformers datasets numpy tqdm matplotlib
```

### 2. ��֤��װ

```bash
python simple_test.py
```

������� "All tests passed!" ˵����װ�ɹ���

### 3. �鿴������ʾ

```bash
python examples/demo.py --demo creation
```

�⽫չʾ��ͬ��С��ģ�����á�

## ���Ĺ�������

### �����Ͳ���ģ��

```python
# �����Ҫģ��
from src.config import ModelConfigs
from src.transformer import Transformer
import torch

# ����һ��С��ģ��
args = ModelConfigs.tiny()
model = Transformer(args)

print(f"ģ�Ͳ�������: {sum(p.numel() for p in model.parameters()):,}")

# ����ǰ�򴫲�
input_ids = torch.randint(0, args.vocab_size, (2, 32))
outputs = model(input_ids)
print(f"�����״: {outputs['logits'].shape}")

# ������ʧ����
outputs = model(input_ids, labels=input_ids)
print(f"��ʧֵ: {outputs['loss'].item():.4f}")
```

### �ı�����ʾ��

```python
# ����ģ��Ϊ����ģʽ
model.eval()

# ��������
input_ids = torch.randint(0, args.vocab_size, (1, 10))

# �����ı�
with torch.no_grad():
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        temperature=0.8,
        do_sample=True
    )

print(f"���볤��: {input_ids.shape[1]}")
print(f"���ɳ���: {generated.shape[1]}")
```

## ѵ��ģ��

### ����ѵ������

```bash
# ѵ��һ��΢��ģ�ͣ���������ɣ�
python examples/train.py \
    --model_size tiny \
    --num_epochs 1 \
    --max_samples 1000 \
    --batch_size 8
```

### ��׼ѵ��

```bash
# ѵ��һ��С��ģ��
python examples/train.py \
    --model_size small \
    --num_epochs 3 \
    --max_samples 10000 \
    --batch_size 16
```

ѵ����ɺ�ģ�ͻᱣ���� `./checkpoints/` Ŀ¼�¡�

## �ı�����

### ����ʽ����

```bash
python examples/generate.py \
    --checkpoint ./checkpoints/best_checkpoint.pt \
    --mode interactive
```

Ȼ��������ʾ�ʣ��� "Once upon a time"��

### ��������

```bash
python examples/generate.py \
    --checkpoint ./checkpoints/best_checkpoint.pt \
    --mode batch \
    --prompts "Once upon a time" "The little girl" \
    --output_file results.json
```

## �Զ�������

### �����Զ���ģ������

```python
from src.config import ModelArgs

# �Զ�������
custom_args = ModelArgs(
    d_model=512,
    n_layers=6,
    n_heads=8,
    kv_lora_rank=256,
    qk_rope_head_dim=32,
    qk_nope_head_dim=32,
    v_head_dim=64,
    intermediate_size=1024,
    vocab_size=32000,
    max_seq_len=1024
)

model = Transformer(custom_args)
```

### ʹ�ò�ͬ�� FFN

```python
from src.feedforward import get_ffn

# ʹ�� GeGLU ������ SwiGLU
ffn = get_ffn(args, ffn_type="geglu")

# ʹ�ñ�׼ FFN
ffn = get_ffn(args, ffn_type="standard")
```

## ���Ժ���֤

### ����������֤

```bash
python final_verification.py
```

### ���е�Ԫ����

```bash
python tests/test_model.py
```

### ģ�ͼܹ���֤

```python
from src.model_utils import validate_model_architecture

validation_results = validate_model_architecture(model, args)
print(f"��֤ͨ��: {validation_results['passed']}")
```

## ���ܷ���

### ģ����Ϣ����

```python
from src.model_utils import save_model_info

# ������ϸ��ģ����Ϣ
save_model_info(model, args, "./model_analysis/")
```

������ɣ�
- �����ļ�
- ��֤����
- Ȩ�ط���
- ���ܻ�׼

### �ٶȻ�׼����

```python
from src.model_utils import benchmark_model_speed

results = benchmark_model_speed(model)
print(f"ǰ�򴫲��ٶ�: {results['forward_pass_times']}")
```

## �߼�����

### Ȩ�طֲ����ӻ�

```python
from src.model_utils import plot_weight_distributions

plot_weight_distributions(model, save_path="weights.png")
```

### ע�������ƶԱ�

```python
from src.attention import MLA, MultiHeadAttention

# ���� MLA �ͱ�׼ע����
mla = MLA(args)
std_attn = MultiHeadAttention(args)

# �Ƚϲ�������
mla_params = sum(p.numel() for p in mla.parameters())
std_params = sum(p.numel() for p in std_attn.parameters())

print(f"MLA ����: {mla_params:,}")
print(f"��׼ע��������: {std_params:,}")
print(f"��������: {mla_params/std_params:.2f}")
```

## ��������

### Q: ����ʱ�����ڴ治�����
A: ���Լ�С���δ�С��ʹ�ø�С��ģ�����ã�
```bash
python examples/train.py --model_size tiny --batch_size 4
```

### Q: ѵ���ٶȺ���
A: ȷ����װ�� GPU �汾�� PyTorch������ʹ�ø�С��ģ�ͣ�
```bash
# ��� GPU ������
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: ���ɵ��ı���������
A: ���������ģ���Ϊģ��û�о������ѵ�������ԣ�
1. ����ѵ��������
2. �ӳ�ѵ��ʱ��
3. ����������

### Q: ��α���ͼ���ģ��
A: ѵ���ű����Զ�����ģ�ͣ����ط�����
```python
from src.model_utils import load_model_for_inference

model, args, tokenizer = load_model_for_inference("./checkpoints/best_checkpoint.pt")
```

## ��һ��ѧϰ

1. **�Ķ�Դ��**: �� `src/config.py` ��ʼ���˽�ģ������
2. **�鿴����**: `tests/test_model.py` չʾ�˸������ʹ�÷���
3. **�о�����**: �˽� MLA��RoPE��SwiGLU �ȼ�����ԭ��
4. **ʵ��Ľ�**: �����޸ļܹ�������¹���

## ��ϲ��

�������Ѿ������� Decoder-only Transformer with MLA �Ļ���ʹ�÷��������ʵ�ְ��������µ� Transformer �ܹ��Ľ������������о���ʵ��Ӧ�á�

����������⣬��鿴 `PROJECT_SUMMARY.md` ��ȡ����ϸ����Ϣ���������� `python final_verification.py` ���������Ĺ�����֤��

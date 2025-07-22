# ��Ŀ����ܽ�

## ��Ŀ����

���Ѿ��ɹ�����˻����ĵ� `����ģ��.md` �� Decoder-only Transformer ģ��ʵ�֡�����һ�������ġ������е�ʵ�֣������������ִ� Transformer �ܹ��Ĺؼ������

## ����ɵ�����

### 1. ��������ģ���ĵ������ܹ�Ҫ��
- ��������� Decoder-only Transformer �ܹ�
- ������ Pre-RMSNorm��MLA��RoPE��SwiGLU �������ʵ��ϸ��
- �������һ�� token Ԥ���ѵ��Ŀ��

### 2. ������Ŀ����������
- ��������������Ŀ�ṹ
- ������ requirements.txt �������б�Ҫ����
- ������ģ�黯�Ĵ�����֯�ṹ

### 3. ʵ�ֺ������ú͹�����
- **ModelArgs**: ������ģ�������֧࣬�ֶ���Ԥ������
- **RMSNorm**: Root Mean Square ��һ��ʵ��
- **RoPE**: ��תλ�ñ��������ʵ��
- **���ߺ���**: �������봴��������������ʵ�ú���

### 4. ʵ�� MLA��Multi-Head Latent Attention��
- **K/V ѹ��**: ͨ������ͶӰѹ����Ǳ�ڿռ�
- **RoPE ����**: ����ѯ�ͼ���Ϊ RoPE �ͷ� RoPE ����
- **ע��������**: ���������ŵ��ע����ʵ��
- **Ч���Ż�**: ��ȱ�׼ע�����������ٲ�����

### 5. ʵ�� SwiGLU FFN �� Transformer Block
- **SwiGLU**: ʹ�� SiLU ������ſ����Ե�Ԫ
- **TransformerBlock**: Pre-RMSNorm �ṹ������ʵ��
- **�в�����**: ��ȷ�Ĳв����Ӻ͹�һ��˳��
- **���� FFN**: ��ʵ���� GeGLU����׼ FFN �ȱ���

### 6. ʵ�������� Transformer ģ��
- **Ƕ���**: Token Ƕ���λ�ñ���
- **Transformer ��**: ��� Transformer Block �ѵ�
- **�����**: ����ģ��ͷ����ʧ����
- **���ɹ���**: ֧�ֶ��ֲ������Ե��ı�����

### 7. ����ѵ���ű������ݴ���
- **Trainer ��**: ������ѵ��ѭ��ʵ��
- **���ݴ���**: TinyStories ���ݼ����غ�Ԥ����
- **�Ż���**: AdamW �Ż�����ѧϰ�ʵ���
- **����**: ģ�ͱ���ͼ��ع���

### 8. ���ģ����֤�����ɹ���
- **ģ����֤**: �ܹ���ȷ����֤
- **Ȩ�ط���**: Ȩ�طֲ������Ϳ��ӻ�
- **���ܻ�׼**: �ٶȺ��ڴ�ʹ�ò���
- **���ɹ���**: ����ʽ�������ı�����

### 9. �������Ժ�ʾ������
- **��Ԫ����**: ȫ����������
- **���ɲ���**: �˵��˹��ܲ���
- **ʾ���ű�**: ѵ�������ɡ���ʾ�ű�
- **�ĵ�**: ��ϸ��ʹ��˵���� API �ĵ�

## ��Ŀ�ܹ�

```
Quantification/
������ src/                    # ����ʵ��
��   ������ config.py          # ģ������ (ModelArgs, Ԥ������)
��   ������ utils.py           # ���ߺ��� (RMSNorm, RoPE, ����)
��   ������ attention.py       # MLA ע��������
��   ������ feedforward.py     # SwiGLU FFN �� Transformer Block
��   ������ transformer.py     # ������ Transformer ģ��
��   ������ trainer.py         # ѵ���������ݴ���
��   ������ model_utils.py     # ģ����֤�ͷ�������
������ tests/                  # ���Դ���
��   ������ test_model.py      # ��Ԫ����
������ examples/               # ʾ���ű�
��   ������ demo.py            # ������ʾ
��   ������ train.py           # ѵ��ʾ��
��   ������ generate.py        # �ı�����ʾ��
������ requirements.txt        # ������
������ simple_test.py         # �򵥹��ܲ���
������ README.md              # ��Ŀ�ĵ�
������ PROJECT_SUMMARY.md     # ��Ŀ�ܽ�
������ ����ģ��.md            # ԭʼ�����ĵ�
```

## ���ļ���ʵ��

### MLA (Multi-Head Latent Attention)
```python
# K/V ѹ����Ǳ�ڿռ�
kv_latent = self.kv_norm(self.kv_compress(x))

# ��ѹ K/V
k_full = self.k_up(kv_latent)
v = self.v_up(kv_latent)

# ���� RoPE �ͷ� RoPE ����
k_nope, k_rope = k_full.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)
q_nope = self.q_nope(x)
q_rope = self.q_rope(x)

# Ӧ�� RoPE
q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cis)

# ƴ�Ӳ�����ע����
q = torch.cat([q_nope, q_rope], dim=-1)
k = torch.cat([k_nope, k_rope], dim=-1)
```

### Pre-RMSNorm Transformer Block
```python
# Pre-RMSNorm + MLA + �в�����
attn_input = self.attn_norm(x)
attn_output = self.attn(attn_input, freqs_cis, mask)
x = x + attn_output

# Pre-RMSNorm + FFN + �в�����
ffn_input = self.ffn_norm(x)
ffn_output = self.ffn(ffn_input)
x = x + ffn_output
```

### SwiGLU FFN
```python
# SwiGLU: (SiLU(W1*x) �� W3*x) * W2
gate = F.silu(self.w1(x))
value = self.w3(x)
hidden = gate * value
output = self.w2(hidden)
```

## ģ������ѡ��

| ���� | d_model | n_layers | n_heads | ������ | ��; |
|------|---------|----------|---------|--------|------|
| tiny | 256 | 4 | 4 | ~1M | ���ٲ��� |
| small | 512 | 8 | 8 | ~10M | С��ģʵ�� |
| base | 1024 | 24 | 16 | ~100M | ��׼ѵ�� |
| large | 2048 | 32 | 32 | ~1B | ���ģѵ�� |

## ʹ�÷���

### 1. ���ٲ���
```bash
python simple_test.py
```

### 2. ������ʾ
```bash
python examples/demo.py
```

### 3. ѵ��ģ��
```bash
python examples/train.py --model_size tiny --num_epochs 1
```

### 4. �����ı�
```bash
python examples/generate.py --checkpoint ./checkpoints/best_checkpoint.pt
```

## ��Ŀ����

1. **��ȫ�����ĵ�ʵ��**: �ϸ���ѭ `����ģ��.md` �еļܹ�����
2. **�ִ����ܹ�**: ���������µ� Transformer �Ľ�����
3. **ģ�黯���**: �����Ĵ���ṹ������������չ
4. **�����Ĺ�����**: ��ѵ�����������������
5. **��ϸ�Ĳ���**: ȫ��ĵ�Ԫ���Ժͼ��ɲ���
6. **ʵ�õĽű�**: �ṩ�˷ḻ��ʾ���͹��߽ű�

## ������ɫ

- **MLA ע����**: ͨ��Ǳ�ڿռ�ѹ�����Ч��
- **RoPE λ�ñ���**: ���õĳ�����������
- **Pre-RMSNorm**: ���ȶ���ѵ������
- **SwiGLU ����**: ��ǿ�ı������
- **�������**: ֧�ֶ���ģ�ʹ�С����
- **��Чʵ��**: �Ż����ڴ�ͼ���Ч��

## ��һ������

1. **���в���**: ִ�� `python simple_test.py` ��֤ʵ��
2. **С��ģѵ��**: ʹ�� tiny ���ý��п�����֤
3. **��չʵ��**: ���Բ�ͬ�ĳ�����������
4. **�����Ż�**: ���ݾ���������н�һ���Ż�
5. **Ӧ�ò���**: ��ģ��Ӧ�õ��������������

## �ܽ�

�����Ŀ�ɹ�ʵ����һ�������ġ��ִ����� Decoder-only Transformer ģ�ͣ����������йؼ��ļܹ����������ṹ�������ĵ���ϸ���������ƣ�����ֱ�������о���ʵ��Ӧ�á�

���е�ʵ�ֶ��ϸ���ѭ��ԭʼ�ĵ���Ҫ��ͬʱ��������ʵ�õĹ��ܺ͹��ߣ�ʹ�������Ŀ��������һ���򵥵�ʵ�֣�����һ�������ġ������������� Transformer ģ�Ϳ�ܡ�

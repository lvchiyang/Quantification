"""
������ Decoder-only Transformer ģ��ʵ��
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .config import ModelArgs
from .utils import RMSNorm, precompute_freqs_cis, create_causal_mask, count_parameters
from .feedforward import TransformerBlock


class FinancialTransformer(nn.Module):
    """
    �������� Transformer ģ��

    �ܹ��ص㣺
    - �������ʱ�����ݣ�OHLC + ����ָ�꣩
    - Pre-RMSNorm
    - MLA (Multi-Head Latent Attention)
    - RoPE (Rotary Position Embedding)
    - SwiGLU FFN
    - ��ʱ���۸�Ԥ��
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # ����������������
        self.n_features = 11  # ���̡���ߡ���͡����̡��Ƿ�����������֡�������%���ɽ�������ʱ�����
        self.n_predictions = 7  # Ԥ��7��ʱ���ļ۸�

        # ����Ƕ��㣺����������ӳ�䵽ģ��ά��
        self.feature_embed = nn.Linear(self.n_features, args.d_model)

        # ������һ����
        self.feature_norm = nn.LayerNorm(self.n_features)

        # Transformer ��
        self.layers = nn.ModuleList([
            TransformerBlock(args, layer_idx=i)
            for i in range(args.n_layers)
        ])

        # ���չ�һ����
        self.norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)

        # �۸�Ԥ��ͷ
        self.price_head = nn.Sequential(
            nn.Linear(args.d_model, args.d_model // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model // 2, self.n_predictions)
        )

        # Ԥ���� RoPE Ƶ��
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                args.qk_rope_head_dim,
                args.max_seq_len * 2,  # Ԥ���������еĿռ�
                args.rope_theta
            ),
            persistent=False
        )

        # ��ʼ��Ȩ��
        self.apply(self._init_weights)

        # ��ӡģ����Ϣ
        self._print_model_info()

    def _init_weights(self, module: nn.Module):
        """��ʼ��ģ��Ȩ��"""
        if isinstance(module, nn.Linear):
            # ʹ�ñ�׼��̬�ֲ���ʼ������׼��Ϊ 1/sqrt(fan_in)
            std = 1.0 / math.sqrt(module.in_features)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_model_info(self):
        """��ӡģ����Ϣ"""
        num_params = count_parameters(self)
        print(f"\nTransformer Model Info:")
        print(f"  Total parameters: {num_params:,}")
        print(f"  Model size: ~{num_params * 4 / 1024**2:.1f} MB (fp32)")
        print(f"  Layers: {self.args.n_layers}")
        print(f"  Hidden size: {self.args.d_model}")
        print(f"  Attention heads: {self.args.n_heads}")
        print(f"  Vocab size: {self.args.vocab_size}")
        print(f"  Max sequence length: {self.args.max_seq_len}")

    def forward(
        self,
        financial_data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_prices: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        ǰ�򴫲�

        Args:
            financial_data: �������� [batch_size, seq_len, n_features]
            attention_mask: ע�������� [batch_size, seq_len]
            target_prices: Ŀ��۸� [batch_size, n_predictions]
            output_attentions: �Ƿ����ע����Ȩ��
            output_hidden_states: �Ƿ��������״̬
            return_dict: �Ƿ񷵻��ֵ��ʽ

        Returns:
            ģ�����
        """
        batch_size, seq_len, n_features = financial_data.shape
        device = financial_data.device

        # ��֤������������
        assert n_features == self.n_features, f"����{self.n_features}��������ʵ��{n_features}��"

        # 1. ������һ����Ƕ��
        normalized_data = self.feature_norm(financial_data)  # [batch_size, seq_len, n_features]
        hidden_states = self.feature_embed(normalized_data)  # [batch_size, seq_len, d_model]

        # 2. ��ȡ RoPE Ƶ��
        freqs_cis = self.freqs_cis[:seq_len].to(device)

        # 3. �����������
        causal_mask = None
        if attention_mask is not None:
            # ����ṩ�� attention_mask����Ҫ����������
            # ����򻯴���ֱ��ʹ���������
            pass

        # 4. ͨ�� Transformer ��
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # ����ÿ�������״̬
            
            hidden_states = layer(
                hidden_states,  # ���뵱ǰ�������״̬
                freqs_cis=freqs_cis,
                attn_mask=causal_mask,
                is_causal=True
            )  # ������º������״̬

        # 5. ���չ�һ��
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 6. �۸�Ԥ��ͷ
        # ʹ�����һ��ʱ�䲽������״̬����Ԥ��
        last_hidden = hidden_states[:, -1, :]  # [batch_size, d_model]
        price_predictions = self.price_head(last_hidden)  # [batch_size, n_predictions]

        # 7. ������ʧ
        loss = None
        if target_prices is not None:
            # ʹ�þ��������ʧ���м۸�Ԥ��
            loss_fct = nn.MSELoss()
            loss = loss_fct(price_predictions, target_prices)

        # 8. ���ؽ��
        if return_dict:
            return {
                "predictions": price_predictions,
                "loss": loss,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }
        else:
            outputs = (price_predictions,)
            if loss is not None:
                outputs = (loss,) + outputs
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs
    
    @torch.no_grad()
    def predict(
        self,
        financial_data: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Ԥ��δ���۸�

        Args:
            financial_data: �������� [batch_size, seq_len, n_features]
            return_dict: �Ƿ񷵻��ֵ��ʽ

        Returns:
            �۸�Ԥ���� [batch_size, n_predictions]
        """
        self.eval()

        outputs = self.forward(
            financial_data=financial_data,
            return_dict=return_dict
        )

        return outputs


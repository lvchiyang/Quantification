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


class Transformer(nn.Module):
    """
    Decoder-only Transformer ģ��

    �ܹ��ص㣺
    - Pre-RMSNorm
    - MLA (Multi-Head Latent Attention)
    - RoPE (Rotary Position Embedding)
    - SwiGLU FFN
    - Next Token Prediction
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Token Ƕ��
        self.tok_embed = nn.Embedding(args.vocab_size, args.d_model)

        # Transformer ��
        self.layers = nn.ModuleList([
            TransformerBlock(args, layer_idx=i)
            for i in range(args.n_layers)
        ])

        # ���չ�һ����
        self.norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)

        # ����ģ��ͷ
        if args.tie_word_embeddings:
            # �����������Ƕ��Ȩ��
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

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

    def get_input_embeddings(self) -> nn.Embedding:
        """��ȡ����Ƕ���"""
        return self.tok_embed

    def set_input_embeddings(self, embeddings: nn.Embedding):
        """��������Ƕ���"""
        self.tok_embed = embeddings

    def get_output_embeddings(self) -> Optional[nn.Linear]:
        """��ȡ���Ƕ���"""
        if self.args.tie_word_embeddings:
            return None
        return self.lm_head
    
    def set_output_embeddings(self, embeddings: Optional[nn.Linear]):
        """�������Ƕ���"""
        if not self.args.tie_word_embeddings:
            self.lm_head = embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        ǰ�򴫲�

        Args:
            input_ids: ���� token IDs [batch_size, seq_len]
            attention_mask: ע�������� [batch_size, seq_len]
            labels: ��ǩ [batch_size, seq_len]�����ڼ�����ʧ
            use_cache: �Ƿ�ʹ�û��棨��δʵ�֣�
            output_attentions: �Ƿ����ע����Ȩ��
            output_hidden_states: �Ƿ��������״̬
            return_dict: �Ƿ񷵻��ֵ��ʽ

        Returns:
            ģ�����
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Token Ƕ��
        hidden_states = self.tok_embed(input_ids)  # [batch_size, seq_len, d_model]

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
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer(
                hidden_states,
                freqs_cis=freqs_cis,
                attn_mask=causal_mask,
                is_causal=True
            )

        # 5. ���չ�һ��
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 6. ����ģ��ͷ
        if self.args.tie_word_embeddings:
            # ʹ������Ƕ���ת����Ϊ���Ȩ��
            logits = F.linear(hidden_states, self.tok_embed.weight)
        else:
            logits = self.lm_head(hidden_states)

        # 7. ������ʧ
        loss = None
        if labels is not None:
            # ��λ��ǩ��Ԥ����һ�� token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # ���㽻������ʧ
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # 8. ���ؽ��
        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }
        else:
            outputs = (logits,)
            if loss is not None:
                outputs = (loss,) + outputs
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        �����ı�

        Args:
            input_ids: ���� token IDs [batch_size, seq_len]
            max_new_tokens: ������� token ��
            temperature: �¶Ȳ���
            top_k: Top-K ����
            top_p: Top-P ����
            do_sample: �Ƿ����
            pad_token_id: ��� token ID
            eos_token_id: ���� token ID

        Returns:
            ���ɵ� token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # ����ѭ��
        for _ in range(max_new_tokens):
            # ǰ�򴫲�
            outputs = self.forward(input_ids, return_dict=True)
            logits = outputs["logits"]

            # ��ȡ���һ��λ�õ� logits
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Ӧ���¶�
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-K ����
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)

            # Top-P ����
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # �Ƴ��ۻ����ʳ��� top_p �� token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # ������̰��ѡ��
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # ƴ���� token
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # ����Ƿ��������� token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # ������г�������
            if input_ids.shape[1] >= self.args.max_seq_len:
                break

        return input_ids

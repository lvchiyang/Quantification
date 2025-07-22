"""
Multi-Head Latent Attention (MLA) ʵ��
MLA �ĺ���˼����ͨ������ͶӰѹ�� K/V��ͬʱ���� RoPE �ͷ� RoPE ����
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .config import ModelArgs
from .utils import RMSNorm, apply_rotary_emb, scaled_dot_product_attention


class MLA(nn.Module):
    """
    Multi-Head Latent Attention
    
    �����ص㣺
    1. K/V ͨ������ͶӰѹ����Ǳ�ڿռ�
    2. Q ��Ϊ RoPE ���ֺͷ� RoPE ����
    3. ����ע����ʱ�Ƚ�ѹ K/V����ƴ��
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.qk_nope_dim = args.qk_nope_head_dim  # �� RoPE ��ѯ/��ά��
        self.qk_rope_dim = args.qk_rope_head_dim  # RoPE ��ѯ/��ά��
        self.v_dim = args.v_head_dim              # ֵͷά��
        self.kv_lora_rank = args.kv_lora_rank     # K/V ѹ��ά��
        
        # �ܵĲ�ѯ/��ά��
        self.qk_head_dim = self.qk_nope_dim + self.qk_rope_dim
        
        # K/V ѹ����Ǳ�ڿռ�
        self.kv_compress = nn.Linear(args.d_model, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=args.layer_norm_eps)
        
        # ��Ǳ�ڿռ��ѹ K/V
        self.k_up = nn.Linear(
            self.kv_lora_rank, 
            self.n_heads * self.qk_head_dim, 
            bias=False
        )
        self.v_up = nn.Linear(
            self.kv_lora_rank, 
            self.n_heads * self.v_dim, 
            bias=False
        )
        
        # ��ѯͶӰ����Ϊ RoPE �ͷ� RoPE ���֣�
        self.q_nope = nn.Linear(
            args.d_model, 
            self.n_heads * self.qk_nope_dim, 
            bias=False
        )
        self.q_rope = nn.Linear(
            args.d_model, 
            self.n_heads * self.qk_rope_dim, 
            bias=False
        )
        
        # ���ͶӰ
        self.out_proj = nn.Linear(
            self.n_heads * self.v_dim, 
            args.d_model, 
            bias=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        ǰ�򴫲�
        
        Args:
            x: �������� [batch_size, seq_len, d_model]
            freqs_cis: RoPE ����Ƶ�� [seq_len, qk_rope_dim//2]
            attn_mask: ע��������
            is_causal: �Ƿ�ʹ���������
            
        Returns:
            ������� [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. ѹ�� K/V ��Ǳ�ڿռ�
        kv_latent = self.kv_compress(x)  # [batch_size, seq_len, kv_lora_rank]
        kv_latent = self.kv_norm(kv_latent)
        
        # 2. ��Ǳ�ڿռ��ѹ K/V
        # ��ѹ K
        k_full = self.k_up(kv_latent)  # [batch_size, seq_len, n_heads * qk_head_dim]
        k_full = k_full.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)
        
        # ���� K �� RoPE �ͷ� RoPE ����
        k_nope, k_rope = k_full.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        
        # ��ѹ V
        v = self.v_up(kv_latent)  # [batch_size, seq_len, n_heads * v_dim]
        v = v.view(batch_size, seq_len, self.n_heads, self.v_dim)
        
        # 3. �����ѯ
        q_nope = self.q_nope(x)  # [batch_size, seq_len, n_heads * qk_nope_dim]
        q_nope = q_nope.view(batch_size, seq_len, self.n_heads, self.qk_nope_dim)
        
        q_rope = self.q_rope(x)  # [batch_size, seq_len, n_heads * qk_rope_dim]
        q_rope = q_rope.view(batch_size, seq_len, self.n_heads, self.qk_rope_dim)
        
        # 4. Ӧ�� RoPE �� RoPE ����
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cis)
        
        # 5. ƴ�Ӳ�ѯ�ͼ��� RoPE ��� RoPE ����
        q = torch.cat([q_nope, q_rope], dim=-1)  # [batch_size, seq_len, n_heads, qk_head_dim]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [batch_size, seq_len, n_heads, qk_head_dim]
        
        # 6. ����ע����
        attn_output = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.args.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # 7. ���ܲ�ͶӰ���
        # [batch_size, seq_len, n_heads, v_dim] -> [batch_size, seq_len, n_heads * v_dim]
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        
        # ���ͶӰ
        output = self.out_proj(attn_output)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    ��׼��ͷע���������ڶԱȣ�
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.head_dim = args.d_model // args.n_heads
        
        assert args.d_model % args.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        self.k_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        self.v_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        self.out_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """��׼��ͷע����ǰ�򴫲�"""
        batch_size, seq_len, _ = x.shape
        
        # ͶӰ Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Ӧ�� RoPE�������Ҫ��
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # ����ע����
        attn_output = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.args.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # ���ܲ�ͶӰ���
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        output = self.out_proj(attn_output)
        
        return output

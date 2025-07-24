"""
Multi-Head Latent Attention (MLA) 实现
MLA 的核心思想是通过潜在投影压缩 K/V，同时支持 RoPE 和非 RoPE 模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from ..config import ModelArgs
from .utils import RMSNorm, apply_rotary_emb, scaled_dot_product_attention


class MLA(nn.Module):
    """
    Multi-Head Latent Attention
    
    ???????
    1. K/V ???????????????????
    2. Q ??? RoPE ?????? RoPE ????
    3. ?????????????? K/V???????
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.qk_nope_dim = args.qk_nope_head_dim  # ?? RoPE ???/?????
        self.qk_rope_dim = args.qk_rope_head_dim  # RoPE ???/?????
        self.v_dim = args.v_head_dim              # ?????
        self.kv_lora_rank = args.kv_lora_rank     # K/V ??????
        
        # ?????/?????
        self.qk_head_dim = self.qk_nope_dim + self.qk_rope_dim
        
        # K/V ??????????
        self.kv_compress = nn.Linear(args.d_model, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=args.layer_norm_eps)
        
        # ????????? K/V
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
        
        # ?????????? RoPE ??? RoPE ?????
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
        
        # ?????
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
        ????
        
        Args:
            x: ???????? [batch_size, seq_len, d_model]
            freqs_cis: RoPE ??????? [seq_len, qk_rope_dim//2]
            attn_mask: ?????????
            is_causal: ?????????????
            
        Returns:
            ??????? [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. ??? K/V ???????
        kv_latent = self.kv_compress(x)  # [batch_size, seq_len, kv_lora_rank]
        kv_latent = self.kv_norm(kv_latent)
        
        # 2. ????????? K/V
        # ??? K
        k_full = self.k_up(kv_latent)  # [batch_size, seq_len, n_heads * qk_head_dim]
        k_full = k_full.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)
        
        # ???? K ?? RoPE ??? RoPE ????
        k_nope, k_rope = k_full.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        
        # ??? V
        v = self.v_up(kv_latent)  # [batch_size, seq_len, n_heads * v_dim]
        v = v.view(batch_size, seq_len, self.n_heads, self.v_dim)
        
        # 3. ??????
        q_nope = self.q_nope(x)  # [batch_size, seq_len, n_heads * qk_nope_dim]
        q_nope = q_nope.view(batch_size, seq_len, self.n_heads, self.qk_nope_dim)
        
        q_rope = self.q_rope(x)  # [batch_size, seq_len, n_heads * qk_rope_dim]
        q_rope = q_rope.view(batch_size, seq_len, self.n_heads, self.qk_rope_dim)
        
        # 4. ??? RoPE ?? RoPE ????
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cis)
        
        # 5. ?????????? RoPE ??? RoPE ????
        q = torch.cat([q_nope, q_rope], dim=-1)  # [batch_size, seq_len, n_heads, qk_head_dim]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [batch_size, seq_len, n_heads, qk_head_dim]
        
        # 6. ?????????
        attn_output = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.args.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # 7. ??????????
        # [batch_size, seq_len, n_heads, v_dim] -> [batch_size, seq_len, n_heads * v_dim]
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        
        # ?????
        output = self.out_proj(attn_output)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    ????????????????????
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
        """???????????????"""
        batch_size, seq_len, _ = x.shape
        
        # ?? Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # ??? RoPE??????????
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # ?????????
        attn_output = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.args.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # ??????????
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        output = self.out_proj(attn_output)
        
        return output

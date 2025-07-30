"""
Multi-Head Latent Attention (MLA) 实现
MLA 的核心思想是通过潜在投影压缩 K/V，同时对所有查询/键应用完整的 RoPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .config import PricePredictionConfig as ModelArgs
from .utils import RMSNorm, apply_rotary_emb, scaled_dot_product_attention


class MLA(nn.Module):
    """
    Multi-Head Latent Attention

    核心特性：
    1. K/V 通过潜在投影进行压缩，减少计算量
    2. 对所有查询/键应用完整的 RoPE 位置编码
    3. 支持因果注意力和 K/V 缓存机制
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.qk_head_dim = args.d_model // args.n_heads  # 查询/键头维度（必须是偶数以支持RoPE）
        self.v_dim = args.v_head_dim                      # 值维度
        self.kv_lora_rank = args.kv_lora_rank             # K/V 压缩维度

        # 确保头维度是偶数（RoPE要求）
        assert self.qk_head_dim % 2 == 0, "qk_head_dim must be even for RoPE"

        # K/V 潜在投影压缩
        self.kv_compress = nn.Linear(args.d_model, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=args.layer_norm_eps)

        # 从压缩表示恢复 K/V
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

        # 查询投影（完整RoPE）
        self.q_proj = nn.Linear(
            args.d_model,
            self.n_heads * self.qk_head_dim,
            bias=False
        )

        # 输出投影
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
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            freqs_cis: RoPE 频率复数 [seq_len, qk_head_dim//2]
            attn_mask: 注意力掩码
            is_causal: 是否使用因果掩码

        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 计算 K/V 潜在表示
        kv_latent = self.kv_compress(x)  # [batch_size, seq_len, kv_lora_rank]
        kv_latent = self.kv_norm(kv_latent)

        # 2. 从潜在表示恢复 K
        k = self.k_up(kv_latent)  # [batch_size, seq_len, n_heads * qk_head_dim]
        k = k.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)

        # 恢复 V
        v = self.v_up(kv_latent)  # [batch_size, seq_len, n_heads * v_dim]
        v = v.view(batch_size, seq_len, self.n_heads, self.v_dim)

        # 3. 计算查询
        q = self.q_proj(x)  # [batch_size, seq_len, n_heads * qk_head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_head_dim)

        # 4. 对所有查询/键应用完整的 RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 5. 计算注意力
        attn_output = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.args.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        # 6. 重塑输出张量
        # [batch_size, seq_len, n_heads, v_dim] -> [batch_size, seq_len, n_heads * v_dim]
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")

        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    标准的多头注意力机制实现
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
        """标准多头注意力前向传播"""
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 应用 RoPE（如果提供）
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # 计算注意力
        attn_output = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.args.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        # 重塑并投影输出
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        output = self.out_proj(attn_output)
        
        return output

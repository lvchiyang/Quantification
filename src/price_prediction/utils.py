"""
工具函数和基础组件
包含 RMSNorm、RoPE 等基础实现
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    ??? LayerNorm??RMSNorm ???????????????????
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ???????? [..., dim]
        Returns:
            ????????????
        """
        # ???? RMS
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


def precompute_freqs_cis(dim: int, end: int, theta: float = 1e4) -> torch.Tensor:
    """
    ????? RoPE ????????

    Args:
        dim: RoPE ?????????????
        end: ??????г???
        theta: ??????????

    Returns:
        ??????????? [end, dim//2]
    """
    # ???????
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # ????λ??????
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)

    # ??????????λ?ú????????
    freqs = torch.outer(t, freqs).float()

    # ??????????? e^(i*theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    ???? freqs_cis ?????????? x ???й?

    Args:
        freqs_cis: ??????? [seq_len, head_dim//2]
        x: ???????? [batch_size, seq_len, n_heads, head_dim]

    Returns:
        ?????????? freqs_cis [1, seq_len, 1, ..., 1, head_dim//2]
    """
    ndim = x.ndim
    assert ndim >= 3, "????????????????3?"
    assert freqs_cis.shape[0] == x.shape[1], f"???г???????: freqs_cis={freqs_cis.shape[0]}, x={x.shape[1]}"
    assert freqs_cis.shape[1] == x.shape[-1] // 2, f"????????: freqs_cis???{x.shape[-1]//2}, ???{freqs_cis.shape[1]}"

    # ?????????????: [1, seq_len, 1, ..., 1, head_dim//2]
    shape = [1] * ndim
    shape[1] = x.shape[1]  # ???г??????
    shape[-1] = freqs_cis.shape[1]  # head_dim//2 ???
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ??????λ???????????????

    Args:
        xq: ??????? [batch_size, seq_len, n_heads, head_dim]
        xk: ?????? [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: ??????? [seq_len, head_dim//2]

    Returns:
        ??? RoPE ??? (xq, xk)
    """
    # ??????????????????????
    # [batch_size, seq_len, n_heads, head_dim] -> [batch_size, seq_len, n_heads, head_dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # ???? freqs_cis ???????
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # ???????????????
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # ???????????????
    return xq_out.type_as(xq), xk_out.type_as(xk)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    ????????????????

    Args:
        seq_len: ???г???
        device: ?豸

    Returns:
        ??????? [seq_len, seq_len]????????? -inf
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor:
    """
    ???????????

    Args:
        query: [batch_size, seq_len, n_heads, head_dim]
        key: [batch_size, seq_len, n_heads, head_dim]
        value: [batch_size, seq_len, n_heads, head_dim]
        attn_mask: ?????????
        dropout_p: dropout ????
        is_causal: ?????????????

    Returns:
        ???????? [batch_size, seq_len, n_heads, head_dim]
    """
    # ?????????????
    # [batch_size, n_heads, seq_len, seq_len]
    scores = torch.einsum("bshd,bthd->bhst", query, key) / math.sqrt(query.shape[-1])

    # ???????
    if is_causal:
        seq_len = query.shape[1]
        causal_mask = create_causal_mask(seq_len, query.device)
        scores = scores + causal_mask

    if attn_mask is not None:
        scores = scores + attn_mask

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Dropout
    if dropout_p > 0.0:
        attn_weights = torch.dropout(attn_weights, dropout_p, train=True)

    # ???????????
    # [batch_size, seq_len, n_heads, head_dim]
    output = torch.einsum("bhst,bthd->bshd", attn_weights, value)

    return output


def count_parameters(model: nn.Module) -> int:
    """??????????????"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """????????豸"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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
    相比 LayerNorm，RMSNorm 不减去均值，只进行缩放
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [..., dim]
        Returns:
            归一化后的张量
        """
        # 计算 RMS
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


def precompute_freqs_cis(dim: int, end: int, theta: float = 1e4) -> torch.Tensor:
    """
    预计算 RoPE 的复数频率

    Args:
        dim: RoPE 维度，必须是偶数
        end: 最大序列长度
        theta: 基础频率参数

    Returns:
        复数频率张量 [end, dim//2]
    """
    # 计算频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 生成位置索引
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)

    # 外积得到所有位置和频率的组合
    freqs = torch.outer(t, freqs).float()

    # 转换为复数形式 e^(i*theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    调整 freqs_cis 的形状以便与 x 进行广播

    Args:
        freqs_cis: 复数频率 [seq_len, head_dim//2]
        x: 输入张量 [batch_size, seq_len, n_heads, head_dim]

    Returns:
        调整形状后的 freqs_cis [1, seq_len, 1, ..., 1, head_dim//2]
    """
    ndim = x.ndim
    assert ndim >= 3, "输入张量维度至少为3维"
    assert freqs_cis.shape[0] == x.shape[1], f"序列长度不匹配: freqs_cis={freqs_cis.shape[0]}, x={x.shape[1]}"
    assert freqs_cis.shape[1] == x.shape[-1] // 2, f"头维度不匹配: freqs_cis需要{x.shape[-1]//2}, 实际{freqs_cis.shape[1]}"

    # 构建广播友好的形状: [1, seq_len, 1, ..., 1, head_dim//2]
    shape = [1] * ndim
    shape[1] = x.shape[1]  # 序列长度维度
    shape[-1] = freqs_cis.shape[1]  # head_dim//2 维度
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码到查询和键张量

    Args:
        xq: 查询张量 [batch_size, seq_len, n_heads, head_dim]
        xk: 键张量 [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: 复数频率 [seq_len, head_dim//2]

    Returns:
        应用 RoPE 后的 (xq, xk)
    """
    # 将实数张量重塑为复数张量
    # [batch_size, seq_len, n_heads, head_dim] -> [batch_size, seq_len, n_heads, head_dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 调整 freqs_cis 形状以便广播
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 应用旋转：复数乘法
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # 转换回原始数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    创建因果注意力掩码

    Args:
        seq_len: 序列长度
        device: 设备

    Returns:
        因果掩码 [seq_len, seq_len]，上三角为 -inf
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
    缩放点积注意力

    Args:
        query: [batch_size, seq_len, n_heads, head_dim]
        key: [batch_size, seq_len, n_heads, head_dim]
        value: [batch_size, seq_len, n_heads, head_dim]
        attn_mask: 注意力掩码
        dropout_p: dropout 概率
        is_causal: 是否使用因果掩码

    Returns:
        注意力输出 [batch_size, seq_len, n_heads, head_dim]
    """
    # 计算注意力分数
    # [batch_size, n_heads, seq_len, seq_len]
    scores = torch.einsum("bshd,bthd->bhst", query, key) / math.sqrt(query.shape[-1])

    # 应用掩码
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

    # 应用注意力权重
    # [batch_size, seq_len, n_heads, head_dim]
    output = torch.einsum("bhst,bthd->bshd", attn_weights, value)

    return output


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

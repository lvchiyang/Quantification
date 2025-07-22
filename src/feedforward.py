"""
ǰ������ʵ��
���� SwiGLU �ͱ�׼ FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelArgs


class SwiGLU(nn.Module):
    """
    SwiGLU ǰ������
    
    SwiGLU(x) = (Swish(W1 * x) �� W3 * x) * W2
    ���� Swish(x) = x * sigmoid(x) = SiLU(x)
    �� ��ʾ��Ԫ�س˷�
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # �м�ά��ͨ���� d_model �� 8/3 ��������ʹ�������е� intermediate_size
        hidden_dim = args.intermediate_size
        
        # �������Բ�
        self.w1 = nn.Linear(args.d_model, hidden_dim, bias=False)  # �ſ�ͶӰ
        self.w2 = nn.Linear(hidden_dim, args.d_model, bias=False)  # ���ͶӰ
        self.w3 = nn.Linear(args.d_model, hidden_dim, bias=False)  # ֵͶӰ
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ǰ�򴫲�
        
        Args:
            x: �������� [batch_size, seq_len, d_model]
            
        Returns:
            ������� [batch_size, seq_len, d_model]
        """
        # �����ſ�ֵ����ֵͨ
        gate = self.w1(x)  # [batch_size, seq_len, hidden_dim]
        value = self.w3(x)  # [batch_size, seq_len, hidden_dim]
        
        # Ӧ�� SiLU ��������ſ�ֵ
        gate = F.silu(gate)
        
        # ��Ԫ�س˷�
        hidden = gate * value
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # ���ͶӰ
        output = self.w2(hidden)
        
        return output


class GeGLU(nn.Module):
    """
    GeGLU ǰ�����磨ʹ�� GELU �������
    
    GeGLU(x) = (GELU(W1 * x) �� W3 * x) * W2
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        hidden_dim = args.intermediate_size
        
        self.w1 = nn.Linear(args.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.d_model, bias=False)
        self.w3 = nn.Linear(args.d_model, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ǰ�򴫲�"""
        gate = self.w1(x)
        value = self.w3(x)
        
        # ʹ�� GELU �����
        gate = F.gelu(gate)
        
        hidden = gate * value
        hidden = self.dropout(hidden)
        
        output = self.w2(hidden)
        
        return output


class StandardFFN(nn.Module):
    """
    ��׼ǰ������
    
    FFN(x) = W2 * ReLU(W1 * x + b1) + b2
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        hidden_dim = args.intermediate_size
        
        self.linear1 = nn.Linear(args.d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, args.d_model)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ǰ�򴫲�"""
        hidden = F.relu(self.linear1(x))
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)
        
        return output


class MoEFFN(nn.Module):
    """
    �򵥵�ר�һ��ǰ�����磨Mixture of Experts��
    """
    
    def __init__(self, args: ModelArgs, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.args = args
        self.num_experts = num_experts
        self.top_k = top_k
        
        # �ſ�����
        self.gate = nn.Linear(args.d_model, num_experts, bias=False)
        
        # ר������
        self.experts = nn.ModuleList([
            SwiGLU(args) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ǰ�򴫲�
        
        Args:
            x: �������� [batch_size, seq_len, d_model]
            
        Returns:
            ������� [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # �����ſ�Ȩ��
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # ѡ�� top-k ר��
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        
        # ��һ�� top-k Ȩ��
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # ��ʼ�����
        output = torch.zeros_like(x)
        
        # ��ÿ��ѡ�е�ר�Ҽ������
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, :, i]  # [batch_size, seq_len]
            expert_weights = top_k_weights[:, :, i:i+1]  # [batch_size, seq_len, 1]
            
            # Ϊÿ��ר�Ҽ������
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    if expert_input.numel() > 0:
                        expert_output = self.experts[expert_id](expert_input)
                        output[mask] += expert_weights[mask] * expert_output
        
        return output


def get_ffn(args: ModelArgs, ffn_type: str = "swiglu") -> nn.Module:
    """
    �������ͻ�ȡǰ������
    
    Args:
        args: ģ������
        ffn_type: FFN ���� ("swiglu", "geglu", "standard", "moe")
        
    Returns:
        ǰ������ģ��
    """
    if ffn_type.lower() == "swiglu":
        return SwiGLU(args)
    elif ffn_type.lower() == "geglu":
        return GeGLU(args)
    elif ffn_type.lower() == "standard":
        return StandardFFN(args)
    elif ffn_type.lower() == "moe":
        return MoEFFN(args)
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")


# Ϊ�������ݣ�����ԭ�е�����
class FeedForward(SwiGLU):
    """SwiGLU �ı���������������"""
    pass


class TransformerBlock(nn.Module):
    """
    Transformer Block with Pre-RMSNorm

    �ṹ��
    x -> RMSNorm -> MLA -> Add -> RMSNorm -> FFN -> Add
    """

    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        # ����������Ҫ��ģ��
        from .attention import MLA
        from .utils import RMSNorm

        # ע������
        self.attn_norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)
        self.attn = MLA(args)

        # ǰ�������
        self.ffn_norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)
        self.ffn = SwiGLU(args)

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
            freqs_cis: RoPE ����Ƶ��
            attn_mask: ע��������
            is_causal: �Ƿ�ʹ���������

        Returns:
            ������� [batch_size, seq_len, d_model]
        """
        # Pre-RMSNorm + MLA + �в�����
        attn_input = self.attn_norm(x)
        attn_output = self.attn(attn_input, freqs_cis, attn_mask, is_causal)
        x = x + attn_output

        # Pre-RMSNorm + FFN + �в�����
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output

        return x

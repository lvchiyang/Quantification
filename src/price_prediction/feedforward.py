"""
前馈网络实现

专门实现各种前馈网络变体，包括：
- SwiGLU：使用 SiLU 激活的门控线性单元
- GeGLU：使用 GELU 激活的门控线性单元
- StandardFFN：标准的两层前馈网络
- MoEFFN：专家混合前馈网络

这些前馈网络可以在 Transformer 层中使用，提供不同的非线性变换能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PricePredictionConfig as ModelArgs


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数实现

    SwiGLU(x) = (Swish(W1 * x) ⊙ W3 * x) * W2
    其中 Swish(x) = x * sigmoid(x) = SiLU(x)
    ⊙ 表示逐元素相乘
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # 通常隐藏维度是 d_model 的 8/3 倍，但这里直接使用配置的 intermediate_size
        hidden_dim = args.intermediate_size

        # 三个线性层
        self.w1 = nn.Linear(args.d_model, hidden_dim, bias=False)  # 门控投影
        self.w2 = nn.Linear(hidden_dim, args.d_model, bias=False)  # 输出投影
        self.w3 = nn.Linear(args.d_model, hidden_dim, bias=False)  # 值投影
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 计算门控和值投影
        gate = self.w1(x)  # [batch_size, seq_len, hidden_dim]
        value = self.w3(x)  # [batch_size, seq_len, hidden_dim]

        # 对门控应用 SiLU 激活函数
        gate = F.silu(gate)

        # 逐元素相乘
        hidden = gate * value

        # Dropout
        hidden = self.dropout(hidden)

        # 输出投影
        output = self.w2(hidden)
        
        return output


class GeGLU(nn.Module):
    """
    GeGLU 激活函数，使用 GELU 作为门控

    GeGLU(x) = (GELU(W1 * x) ⊙ W3 * x) * W2
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
        """前向传播"""
        gate = self.w1(x)
        value = self.w3(x)

        # 对门控应用 GELU 激活
        gate = F.gelu(gate)
        
        hidden = gate * value
        hidden = self.dropout(hidden)
        
        output = self.w2(hidden)
        
        return output


class StandardFFN(nn.Module):
    """
    标准前馈网络

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
        """前向传播"""
        hidden = F.relu(self.linear1(x))
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)
        
        return output


class MoEFFN(nn.Module):
    """
    专家混合前馈网络（Mixture of Experts）
    """

    def __init__(self, args: ModelArgs, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.args = args
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络
        self.gate = nn.Linear(args.d_model, num_experts, bias=False)

        # 专家网络
        self.experts = nn.ModuleList([
            SwiGLU(args) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # 计算门控权重
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)

        # 选择 top-k 专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)

        # 归一化 top-k 权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 初始化输出
        output = torch.zeros_like(x)

        # 对每个选中的专家进行计算
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, :, i]  # [batch_size, seq_len]
            expert_weights = top_k_weights[:, :, i:i+1]  # [batch_size, seq_len, 1]

            # 为每个专家处理对应的输入
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
    根据类型创建前馈网络

    Args:
        args: 模型配置
        ffn_type: FFN 类型 ("swiglu", "geglu", "standard", "moe")

    Returns:
        前馈网络模块
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


# 为了向后兼容，保留原有的类名
class FeedForward(SwiGLU):
    """SwiGLU 的别名，用于向后兼容"""
    pass




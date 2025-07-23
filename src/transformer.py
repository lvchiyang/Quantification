"""
完整的 Decoder-only Transformer 模型实现
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
    金融量化 Transformer 模型

    架构特点：
    - 处理金融时序数据（OHLC + 技术指标）
    - Pre-RMSNorm
    - MLA (Multi-Head Latent Attention)
    - RoPE (Rotary Position Embedding)
    - SwiGLU FFN
    - 多时间点价格预测
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # 金融数据特征数量
        self.n_features = 11  # 开盘、最高、最低、收盘、涨幅、振幅、总手、金额、换手%、成交次数、时间编码
        self.n_predictions = 7  # 预测7个时间点的价格

        # 输入嵌入层：将金融特征映射到模型维度
        self.feature_embed = nn.Linear(self.n_features, args.d_model)

        # 特征归一化层
        self.feature_norm = nn.LayerNorm(self.n_features)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(args, layer_idx=i)
            for i in range(args.n_layers)
        ])

        # 最终归一化层
        self.norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)

        # 价格预测头
        self.price_head = nn.Sequential(
            nn.Linear(args.d_model, args.d_model // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model // 2, self.n_predictions)
        )

        # 预计算 RoPE 频率
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                args.qk_rope_head_dim,
                args.max_seq_len * 2,  # 预留更长序列的空间
                args.rope_theta
            ),
            persistent=False
        )

        # 初始化权重
        self.apply(self._init_weights)

        # 打印模型信息
        self._print_model_info()

    def _init_weights(self, module: nn.Module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            # 使用标准正态分布初始化，标准差为 1/sqrt(fan_in)
            std = 1.0 / math.sqrt(module.in_features)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_model_info(self):
        """打印模型信息"""
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
        前向传播

        Args:
            financial_data: 金融数据 [batch_size, seq_len, n_features]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            target_prices: 目标价格 [batch_size, n_predictions]
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典格式

        Returns:
            模型输出
        """
        batch_size, seq_len, n_features = financial_data.shape
        device = financial_data.device

        # 验证输入特征数量
        assert n_features == self.n_features, f"期望{self.n_features}个特征，实际{n_features}个"

        # 1. 特征归一化和嵌入
        normalized_data = self.feature_norm(financial_data)  # [batch_size, seq_len, n_features]
        hidden_states = self.feature_embed(normalized_data)  # [batch_size, seq_len, d_model]

        # 2. 获取 RoPE 频率
        freqs_cis = self.freqs_cis[:seq_len].to(device)

        # 3. 创建因果掩码
        causal_mask = None
        if attention_mask is not None:
            # 如果提供了 attention_mask，需要结合因果掩码
            # 这里简化处理，直接使用因果掩码
            pass

        # 4. 通过 Transformer 层
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # 保存每层的隐藏状态
            
            hidden_states = layer(
                hidden_states,  # 输入当前层的隐藏状态
                freqs_cis=freqs_cis,
                attn_mask=causal_mask,
                is_causal=True
            )  # 输出更新后的隐藏状态

        # 5. 最终归一化
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 6. 价格预测头
        # 使用最后一个时间步的隐藏状态进行预测
        last_hidden = hidden_states[:, -1, :]  # [batch_size, d_model]
        price_predictions = self.price_head(last_hidden)  # [batch_size, n_predictions]

        # 7. 计算损失
        loss = None
        if target_prices is not None:
            # 使用均方误差损失进行价格预测
            loss_fct = nn.MSELoss()
            loss = loss_fct(price_predictions, target_prices)

        # 8. 返回结果
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
        预测未来价格

        Args:
            financial_data: 金融数据 [batch_size, seq_len, n_features]
            return_dict: 是否返回字典格式

        Returns:
            价格预测结果 [batch_size, n_predictions]
        """
        self.eval()

        outputs = self.forward(
            financial_data=financial_data,
            return_dict=return_dict
        )

        return outputs


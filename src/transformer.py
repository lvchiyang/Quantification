"""
金融量化 Decoder-only Transformer 模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Gumbel-Softmax采样，支持离散输出的可微分近似

    Args:
        logits: 输入logits [batch_size, num_classes]
        temperature: 温度参数，越小越接近one-hot
        hard: 是否使用硬采样（前向传播时离散，反向传播时连续）

    Returns:
        采样结果 [batch_size, num_classes]
    """
    # 添加Gumbel噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + gumbel_noise) / temperature
    y_soft = F.softmax(y, dim=-1)

    if hard:
        # 硬采样：前向传播时使用one-hot，反向传播时使用soft
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        # 使用straight-through estimator
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


def straight_through_round(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through Estimator for rounding
    前向传播时四舍五入，反向传播时保持梯度
    """
    return (torch.round(x) - x).detach() + x

from .config import ModelArgs
from .utils import RMSNorm, precompute_freqs_cis, create_causal_mask, count_parameters
from .feedforward import TransformerBlock
from .discrete_position_methods import create_position_head, DiscretePositionLoss
from .market_classifier import create_market_classifier
from .information_ratio_loss import create_information_ratio_loss, MultiObjectiveTradingLoss


class PositionReturnCalculator:
    """
    仓位收益计算器

    计算基于仓位和次日涨跌幅的收益率
    """

    def __init__(self, args):
        self.args = args

    def compute_loss(self, position_predictions: torch.Tensor, next_day_returns: torch.Tensor) -> torch.Tensor:
        """
        计算仓位策略损失

        Args:
            position_predictions: 预测的仓位 [batch_size, 1]
            next_day_returns: 次日涨跌幅 [batch_size]

        Returns:
            负收益率（作为损失）
        """
        # 将仓位从 [batch_size, 1] 压缩为 [batch_size]
        positions = position_predictions.squeeze(-1)

        # 将仓位标准化到 [0, 1] 范围
        normalized_positions = positions / 10.0

        # 计算收益：标准化仓位 * 次日涨跌幅
        returns = normalized_positions * next_day_returns

        # 添加风险调整项（鼓励适度仓位，避免过度激进）
        risk_penalty = torch.mean(torch.abs(normalized_positions - 0.5)) * 0.01

        # 返回负平均收益作为损失（最大化收益）+ 风险惩罚
        return -torch.mean(returns) + risk_penalty

    def compute_loss_with_logits(
        self,
        position_logits: torch.Tensor,
        next_day_returns: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        直接从logits计算损失，避免中间的离散化步骤

        Args:
            position_logits: 仓位logits [batch_size, 11]
            next_day_returns: 次日涨跌幅 [batch_size]
            temperature: Gumbel-Softmax温度

        Returns:
            负收益率（作为损失）
        """
        # 使用软采样计算期望收益
        position_probs = F.softmax(position_logits / temperature, dim=-1)

        # 仓位值 [0, 1, 2, ..., 10]
        position_values = torch.arange(11, dtype=torch.float32, device=position_logits.device)

        # 计算期望仓位
        expected_positions = torch.sum(position_probs * position_values, dim=-1)

        # 标准化到 [0, 1]
        normalized_positions = expected_positions / 10.0

        # 计算期望收益
        expected_returns = normalized_positions * next_day_returns

        return -torch.mean(expected_returns)


class FinancialTransformer(nn.Module):
    """
    金融量化 Transformer 模型

    核心特点
    - 处理金融时序数据（OHLC + 技术指标）
    - Pre-RMSNorm
    - MLA (Multi-Head Latent Attention)
    - RoPE (Rotary Position Embedding)
    - SwiGLU FFN
    - 多时点价格预测和交易策略学习
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # 金融数据参数
        self.n_features = args.n_features  # 输入特征数（开盘、最高、最低、收盘、涨幅、振幅、总手、金额、换手率、成交次数、时间编码）
        self.n_predictions = args.n_predictions  # 预测7个时间点的价格

        # 特征嵌入层：将金融特征映射到模型维度
        self.feature_embed = nn.Linear(self.n_features, args.d_model)

        # 特征归一化层
        self.feature_norm = nn.LayerNorm(self.n_features)

        # Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(args, layer_idx=i)
            for i in range(args.n_layers)
        ])

        # 最终归一化层
        self.norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)

        # 价格预测
        self.price_head = nn.Sequential(
            nn.Linear(args.d_model, args.d_model // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model // 2, self.n_predictions)
        )

        # 策略状态管理（递归状态更新）
        if args.enable_trading_strategy and args.enable_stateful_training:
            self.strategy_state_dim = args.strategy_state_dim

            # 策略状态更新器
            if args.state_update_method == 'gru':
                self.strategy_memory = nn.GRUCell(
                    input_size=args.d_model + 1,  # 隐藏状态 + 仓位值
                    hidden_size=self.strategy_state_dim
                )
            elif args.state_update_method == 'lstm':
                self.strategy_memory = nn.LSTMCell(
                    input_size=args.d_model + 1,
                    hidden_size=self.strategy_state_dim
                )
            else:  # attention
                self.strategy_memory = nn.MultiheadAttention(
                    embed_dim=self.strategy_state_dim,
                    num_heads=8,
                    dropout=args.state_dropout
                )

            # 策略状态初始化参数
            self.strategy_state_init = nn.Parameter(torch.randn(self.strategy_state_dim))

            # 状态dropout
            self.state_dropout = nn.Dropout(args.state_dropout)
        else:
            self.strategy_state_dim = 0
            self.strategy_memory = None
            self.strategy_state_init = None

        # 仓位策略预测头（输出0-10的整数仓位）
        if args.enable_trading_strategy:
            # 选择离散化方法
            position_method = getattr(args, 'position_method', 'gumbel_softmax')

            # 输入维度：基础特征 + 策略状态（如果启用）
            position_input_dim = args.d_model + self.strategy_state_dim

            self.position_head = create_position_head(
                method=position_method,
                d_model=position_input_dim,
                num_positions=11,  # 0-10
                max_position=10,
                dropout=args.dropout
            )
            self.position_loss_fn = DiscretePositionLoss(max_position=10)
        else:
            self.position_head = None
            self.position_loss_fn = None

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

        # 创建市场分类器和信息比率损失函数
        if args.enable_trading_strategy:
            self.market_classifier = create_market_classifier(args)
            self.information_ratio_loss_fn = create_information_ratio_loss(args, self.market_classifier)

            # 多目标损失函数
            self.multi_objective_loss_fn = MultiObjectiveTradingLoss(
                market_classifier=self.market_classifier,
                price_loss_weight=args.price_loss_weight,
                information_ratio_weight=args.information_ratio_weight,
                opportunity_cost_weight=args.opportunity_cost_weight,
                risk_adjustment_weight=args.risk_adjustment_weight
            )

            # 保留原有的仓位计算器作为备用
            self.position_calculator = PositionReturnCalculator(args)
        else:
            self.market_classifier = None
            self.information_ratio_loss_fn = None
            self.multi_objective_loss_fn = None
            self.position_calculator = None

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
        next_day_returns: Optional[torch.Tensor] = None,
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
            next_day_returns: 次日涨跌幅 [batch_size] 用于计算仓位收益
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典形式

        Returns:
            模型输出
        """
        batch_size, seq_len, n_features = financial_data.shape
        device = financial_data.device

        # 验证输入特征数量
        assert n_features == self.n_features, f"期望{self.n_features}个特征，实际得到{n_features}个"

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

        # 7. 仓位策略预测头
        position_output = None
        if self.position_head is not None:
            position_output = self.position_head(last_hidden)
            position_predictions = position_output['positions']  # [batch_size, 1]

        # 8. 计算损失
        loss = None
        price_loss = None
        position_loss = None

        if target_prices is not None:
            # 价格预测损失
            price_loss_fct = nn.MSELoss()
            price_loss = price_loss_fct(price_predictions, target_prices)
            loss = self.args.price_loss_weight * price_loss

        if position_output is not None and next_day_returns is not None and self.position_loss_fn is not None:
            # 仓位策略损失（负收益率）
            position_loss = self.position_loss_fn(position_output, next_day_returns, method='expected_return')

            if loss is not None:
                loss = loss + self.args.trading_loss_weight * position_loss
            else:
                loss = self.args.trading_loss_weight * position_loss

        # 9. 返回结果
        if return_dict:
            result = {
                "price_predictions": price_predictions,
                "loss": loss,
                "price_loss": price_loss,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }
            if position_output is not None:
                result["position_predictions"] = position_output['positions']
                result["position_output"] = position_output  # 包含完整的输出信息
                result["position_loss"] = position_loss
            return result
        else:
            outputs = (price_predictions,)
            if position_predictions is not None:
                outputs = outputs + (position_predictions,)
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
        预测未来价格和交易策略

        Args:
            financial_data: 金融数据 [batch_size, seq_len, n_features]
            return_dict: 是否返回字典形式

        Returns:
            价格预测和交易策略预测
        """
        self.eval()

        outputs = self.forward(
            financial_data=financial_data,
            return_dict=return_dict
        )

        return outputs

    def forward_single_day(
        self,
        financial_data: torch.Tensor,
        strategy_state: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        单日预测：给定历史数据和策略状态，预测价格和仓位

        Args:
            financial_data: [batch_size, seq_len, n_features] 单日的历史数据
            strategy_state: [batch_size, strategy_state_dim] 当前策略状态
            return_dict: 是否返回字典形式

        Returns:
            包含价格预测、仓位预测和新策略状态的字典
        """
        batch_size, seq_len, n_features = financial_data.shape
        device = financial_data.device

        # 验证输入特征数量
        assert n_features == self.n_features, f"期望{self.n_features}个特征，实际得到{n_features}个"

        # 1. 特征归一化和嵌入
        normalized_data = self.feature_norm(financial_data)
        hidden_states = self.feature_embed(normalized_data)

        # 2. 获取 RoPE 频率
        freqs_cis = self.freqs_cis[:seq_len].to(device)

        # 3. 通过 Transformer 层
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                freqs_cis=freqs_cis,
                attn_mask=None,
                is_causal=True
            )

        # 4. 最终归一化
        hidden_states = self.norm(hidden_states)

        # 5. 提取最后时间步的特征
        last_hidden = hidden_states[:, -1, :]  # [batch_size, d_model]

        # 6. 价格预测
        price_predictions = self.price_head(last_hidden)

        # 7. 仓位预测（结合策略状态）
        position_output = None
        new_strategy_state = None

        if self.position_head is not None:
            # 初始化策略状态（如果未提供）
            if strategy_state is None and self.strategy_state_init is not None:
                strategy_state = self.strategy_state_init.unsqueeze(0).expand(
                    batch_size, -1
                ).contiguous()

            # 结合历史特征和策略状态
            if strategy_state is not None:
                combined_features = torch.cat([last_hidden, strategy_state], dim=-1)
            else:
                combined_features = last_hidden

            # 仓位预测
            position_output = self.position_head(combined_features)
            position_predictions = position_output['positions']

            # 更新策略状态
            if self.strategy_memory is not None and strategy_state is not None:
                # 计算期望仓位用于状态更新
                expected_position = position_predictions.squeeze(-1)  # [batch_size]

                # 状态更新输入：历史特征 + 仓位
                state_input = torch.cat([last_hidden, expected_position.unsqueeze(-1)], dim=-1)

                # 根据更新方法选择
                if isinstance(self.strategy_memory, nn.GRUCell):
                    new_strategy_state = self.strategy_memory(state_input, strategy_state)
                elif isinstance(self.strategy_memory, nn.LSTMCell):
                    # LSTM需要处理cell state，这里简化为只用hidden state
                    new_strategy_state, _ = self.strategy_memory(state_input, (strategy_state, strategy_state))
                else:  # attention
                    # 注意力机制更新（简化实现）
                    new_strategy_state = strategy_state

                # 应用dropout
                if self.training:
                    new_strategy_state = self.state_dropout(new_strategy_state)
            else:
                new_strategy_state = strategy_state

        if return_dict:
            result = {
                "price_predictions": price_predictions,
                "strategy_state": new_strategy_state
            }
            if position_output is not None:
                result["position_predictions"] = position_output['positions']
                result["position_output"] = position_output
            return result
        else:
            outputs = (price_predictions,)
            if position_output is not None:
                outputs = outputs + (position_output['positions'],)
            if new_strategy_state is not None:
                outputs = outputs + (new_strategy_state,)
            return outputs


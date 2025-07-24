"""
GRU策略网络
基于价格预测网络的特征，学习交易策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# 导入离散化方法
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 离散仓位预测方法
class GumbelSoftmaxPositionHead(nn.Module):
    """
    Gumbel-Softmax仓位预测头
    训练时连续可微，推理时离散
    """

    def __init__(self, d_model: int, num_positions: int = 11, dropout: float = 0.0):
        super().__init__()
        self.num_positions = num_positions

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_positions)
        )

        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states: torch.Tensor, hard: bool = None) -> dict:
        logits = self.head(hidden_states)

        if hard is None:
            hard = not self.training

        # Gumbel-Softmax采样
        probs = self.gumbel_softmax(logits, self.temperature, hard=hard)

        # 计算期望仓位值
        position_values = torch.arange(self.num_positions, dtype=torch.float32, device=logits.device)
        positions = torch.sum(probs * position_values, dim=-1, keepdim=True)

        # 离散仓位（用于展示）
        discrete_positions = torch.argmax(probs, dim=-1, keepdim=True).float()

        return {
            'logits': logits,
            'probs': probs,
            'positions': positions,
            'discrete_positions': discrete_positions
        }

    def gumbel_softmax(self, logits: torch.Tensor, temperature: float, hard: bool = False) -> torch.Tensor:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        y = (logits + gumbel_noise) / temperature
        y_soft = F.softmax(y, dim=-1)

        if hard:
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
            y = y_hard - y_soft.detach() + y_soft

        return y_soft if not hard else y


def create_position_head(method: str, d_model: int, **kwargs) -> nn.Module:
    """创建仓位预测头"""
    if method == 'gumbel_softmax':
        return GumbelSoftmaxPositionHead(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown position head method: {method}")


class GRUStrategyNetwork(nn.Module):
    """
    GRU策略网络
    输入：价格预测网络的特征
    输出：交易仓位决策
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feature_dim = args.d_model  # 来自价格网络的特征维度
        self.strategy_state_dim = args.strategy_state_dim
        self.position_range = args.position_range_max + 1  # 0-10，共11个仓位
        
        # GRU策略记忆网络
        if args.state_update_method == 'gru':
            self.strategy_memory = nn.GRUCell(
                input_size=self.feature_dim + 1,  # 特征 + 上次仓位
                hidden_size=self.strategy_state_dim
            )
        elif args.state_update_method == 'lstm':
            self.strategy_memory = nn.LSTMCell(
                input_size=self.feature_dim + 1,
                hidden_size=self.strategy_state_dim
            )
        else:
            raise ValueError(f"不支持的状态更新方法: {args.state_update_method}")
        
        # 仓位决策头
        self.position_head = create_position_head(
            method=args.position_method,
            d_model=self.feature_dim + self.strategy_state_dim,
            num_positions=self.position_range,
            max_position=args.position_range_max,
            dropout=args.dropout
        )
        
        # 策略状态初始化
        self.strategy_state_init = nn.Parameter(torch.randn(self.strategy_state_dim))
        
        # Dropout
        self.state_dropout = nn.Dropout(args.state_dropout)
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GRUCell, nn.LSTMCell)):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def init_strategy_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        初始化策略状态
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            初始策略状态 [batch_size, strategy_state_dim]
        """
        return self.strategy_state_init.unsqueeze(0).expand(
            batch_size, -1
        ).contiguous().to(device)
    
    def forward_single_step(
        self,
        price_features: torch.Tensor,
        strategy_state: torch.Tensor,
        last_position: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        单步策略决策
        
        Args:
            price_features: [batch_size, feature_dim] 价格网络提取的特征
            strategy_state: [batch_size, strategy_state_dim] 当前策略状态
            last_position: [batch_size] 上次的仓位（0-10）
            
        Returns:
            包含仓位决策和新状态的字典
        """
        batch_size = price_features.shape[0]
        device = price_features.device
        
        # 如果没有提供上次仓位，默认为0（空仓）
        if last_position is None:
            last_position = torch.zeros(batch_size, device=device)
        
        # 1. 结合特征和策略状态进行仓位决策
        combined_input = torch.cat([price_features, strategy_state], dim=-1)
        position_output = self.position_head(combined_input)
        
        # 2. 更新策略状态
        # 使用可微分的期望仓位而不是argmax
        if 'positions' in position_output:
            expected_position = position_output['positions'].squeeze(-1)  # [batch_size]
        else:
            # 如果是logits，计算期望值
            position_probs = F.softmax(position_output['logits'], dim=-1)
            position_values = torch.arange(
                self.position_range, device=device, dtype=torch.float32
            )
            expected_position = torch.sum(position_probs * position_values, dim=-1)
        
        # 状态更新输入：特征 + 期望仓位
        state_input = torch.cat([price_features, expected_position.unsqueeze(-1)], dim=-1)
        
        if isinstance(self.strategy_memory, nn.GRUCell):
            new_strategy_state = self.strategy_memory(state_input, strategy_state)
        elif isinstance(self.strategy_memory, nn.LSTMCell):
            # LSTM需要处理cell state，这里简化为只用hidden state
            new_strategy_state, _ = self.strategy_memory(
                state_input, (strategy_state, strategy_state)
            )
        
        # 应用dropout
        if self.training:
            new_strategy_state = self.state_dropout(new_strategy_state)
        
        return {
            'position_output': position_output,
            'strategy_state': new_strategy_state,
            'expected_position': expected_position
        }
    
    def forward_sequence(
        self,
        price_features_sequence: torch.Tensor,
        initial_strategy_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        序列策略决策（20天）
        
        Args:
            price_features_sequence: [batch_size, seq_len, feature_dim] 特征序列
            initial_strategy_state: [batch_size, strategy_state_dim] 初始状态
            
        Returns:
            包含所有决策的字典
        """
        batch_size, seq_len, feature_dim = price_features_sequence.shape
        device = price_features_sequence.device
        
        # 初始化策略状态
        if initial_strategy_state is None:
            strategy_state = self.init_strategy_state(batch_size, device)
        else:
            strategy_state = initial_strategy_state
        
        # 存储所有输出
        all_position_outputs = []
        all_strategy_states = []
        all_expected_positions = []
        
        last_position = None
        
        # 逐步处理序列
        for step in range(seq_len):
            step_features = price_features_sequence[:, step, :]  # [batch_size, feature_dim]
            
            # 单步决策
            step_output = self.forward_single_step(
                step_features, strategy_state, last_position
            )
            
            # 保存输出
            all_position_outputs.append(step_output['position_output'])
            all_strategy_states.append(step_output['strategy_state'])
            all_expected_positions.append(step_output['expected_position'])
            
            # 更新状态和仓位
            strategy_state = step_output['strategy_state']
            last_position = step_output['expected_position']
        
        return {
            'all_position_outputs': all_position_outputs,  # List[Dict]
            'all_strategy_states': all_strategy_states,    # List[Tensor]
            'all_expected_positions': torch.stack(all_expected_positions, dim=1),  # [batch, seq_len]
            'final_strategy_state': strategy_state
        }
    
    def get_position_predictions(self, position_outputs: List[Dict]) -> torch.Tensor:
        """
        从position_outputs中提取仓位预测
        
        Args:
            position_outputs: 仓位输出列表
            
        Returns:
            仓位预测张量 [batch_size, seq_len, 1]
        """
        positions = []
        for output in position_outputs:
            if 'positions' in output:
                positions.append(output['positions'])
            else:
                # 从logits计算期望仓位
                logits = output['logits']
                probs = F.softmax(logits, dim=-1)
                position_values = torch.arange(
                    self.position_range, device=logits.device, dtype=torch.float32
                )
                expected_pos = torch.sum(probs * position_values, dim=-1, keepdim=True)
                positions.append(expected_pos)
        
        return torch.stack(positions, dim=1)  # [batch_size, seq_len, 1]


def create_gru_strategy_network(args) -> GRUStrategyNetwork:
    """
    创建GRU策略网络
    
    Args:
        args: 模型配置
        
    Returns:
        GRU策略网络实例
    """
    return GRUStrategyNetwork(args)

"""
离散仓位预测的多种方法
支持输出整数仓位同时保持梯度可计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GumbelSoftmaxPositionHead(nn.Module):
    """
    方法1: Gumbel-Softmax仓位预测头
    
    优点：
    - 训练时连续可微，推理时离散
    - 理论基础扎实
    - 温度参数可调节离散程度
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
        """
        Args:
            hidden_states: [batch_size, d_model]
            hard: 是否使用硬采样，None时根据training模式自动选择
            
        Returns:
            dict包含logits, probs, positions, discrete_positions
        """
        logits = self.head(hidden_states)  # [batch_size, num_positions]
        
        if hard is None:
            hard = not self.training
        
        # Gumbel-Softmax采样
        probs = self.gumbel_softmax(logits, self.temperature, hard=hard)
        
        # 计算期望仓位值
        position_values = torch.arange(self.num_positions, dtype=torch.float32, device=logits.device)
        positions = torch.sum(probs * position_values, dim=-1, keepdim=True)  # [batch_size, 1]
        
        # 离散仓位（用于展示）
        discrete_positions = torch.argmax(probs, dim=-1, keepdim=True).float()  # [batch_size, 1]
        
        return {
            'logits': logits,
            'probs': probs,
            'positions': positions,
            'discrete_positions': discrete_positions
        }
    
    def gumbel_softmax(self, logits: torch.Tensor, temperature: float, hard: bool = False) -> torch.Tensor:
        """Gumbel-Softmax采样"""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        y = (logits + gumbel_noise) / temperature
        y_soft = F.softmax(y, dim=-1)
        
        if hard:
            y_hard = torch.zeros_like(y_soft)
            y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
            y = y_hard - y_soft.detach() + y_soft
        
        return y_soft if not hard else y


class StraightThroughPositionHead(nn.Module):
    """
    方法2: Straight-Through Estimator仓位预测头
    
    优点：
    - 实现简单
    - 前向传播直接输出整数
    - 反向传播梯度直通
    """
    
    def __init__(self, d_model: int, max_position: int = 10, dropout: float = 0.0):
        super().__init__()
        self.max_position = max_position
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # 输出[0,1]
        )
    
    def forward(self, hidden_states: torch.Tensor) -> dict:
        """
        Args:
            hidden_states: [batch_size, d_model]
            
        Returns:
            dict包含continuous_positions, discrete_positions
        """
        continuous = self.head(hidden_states)  # [batch_size, 1], range [0,1]
        continuous_scaled = continuous * self.max_position  # [batch_size, 1], range [0,10]
        
        # Straight-through estimator: 前向取整，反向保持梯度
        discrete = self.straight_through_round(continuous_scaled)
        
        return {
            'continuous_positions': continuous_scaled,
            'discrete_positions': discrete,
            'positions': discrete  # 主要输出
        }
    
    def straight_through_round(self, x: torch.Tensor) -> torch.Tensor:
        """Straight-Through Estimator for rounding"""
        return (torch.round(x) - x).detach() + x


class ConcretePositionHead(nn.Module):
    """
    方法3: Concrete Distribution (连续松弛)
    
    优点：
    - 无需Gumbel噪声
    - 温度参数控制离散程度
    - 训练稳定
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
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, hidden_states: torch.Tensor) -> dict:
        """
        Args:
            hidden_states: [batch_size, d_model]
            
        Returns:
            dict包含logits, probs, positions
        """
        logits = self.head(hidden_states)  # [batch_size, num_positions]
        
        # Concrete distribution
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # 计算期望仓位值
        position_values = torch.arange(self.num_positions, dtype=torch.float32, device=logits.device)
        positions = torch.sum(probs * position_values, dim=-1, keepdim=True)  # [batch_size, 1]
        
        return {
            'logits': logits,
            'probs': probs,
            'positions': positions
        }


class DiscretePositionLoss(nn.Module):
    """
    离散仓位的损失函数
    支持多种仓位预测方法
    """
    
    def __init__(self, max_position: int = 10):
        super().__init__()
        self.max_position = max_position
    
    def forward(
        self, 
        position_output: dict, 
        next_day_returns: torch.Tensor,
        method: str = 'expected_return'
    ) -> torch.Tensor:
        """
        计算仓位策略损失
        
        Args:
            position_output: 仓位预测输出字典
            next_day_returns: 次日涨跌幅 [batch_size]
            method: 损失计算方法
            
        Returns:
            负收益率损失
        """
        if method == 'expected_return':
            return self._expected_return_loss(position_output, next_day_returns)
        elif method == 'discrete_return':
            return self._discrete_return_loss(position_output, next_day_returns)
        elif method == 'distribution_return':
            return self._distribution_return_loss(position_output, next_day_returns)
        else:
            raise ValueError(f"Unknown loss method: {method}")
    
    def _expected_return_loss(self, position_output: dict, next_day_returns: torch.Tensor) -> torch.Tensor:
        """基于期望仓位计算损失"""
        positions = position_output['positions'].squeeze(-1)  # [batch_size]
        normalized_positions = positions / self.max_position  # 标准化到[0,1]
        returns = normalized_positions * next_day_returns
        return -torch.mean(returns)
    
    def _discrete_return_loss(self, position_output: dict, next_day_returns: torch.Tensor) -> torch.Tensor:
        """基于离散仓位计算损失"""
        if 'discrete_positions' in position_output:
            positions = position_output['discrete_positions'].squeeze(-1)
        else:
            positions = position_output['positions'].squeeze(-1)
        
        normalized_positions = positions / self.max_position
        returns = normalized_positions * next_day_returns
        return -torch.mean(returns)
    
    def _distribution_return_loss(self, position_output: dict, next_day_returns: torch.Tensor) -> torch.Tensor:
        """基于概率分布计算期望损失"""
        if 'probs' not in position_output:
            return self._expected_return_loss(position_output, next_day_returns)
        
        probs = position_output['probs']  # [batch_size, num_positions]
        position_values = torch.arange(probs.shape[-1], dtype=torch.float32, device=probs.device)
        
        # 计算每个可能仓位的收益
        normalized_positions = position_values / self.max_position  # [num_positions]
        
        # 计算期望收益
        expected_returns = torch.sum(
            probs * normalized_positions.unsqueeze(0) * next_day_returns.unsqueeze(-1), 
            dim=-1
        )  # [batch_size]
        
        return -torch.mean(expected_returns)


# 工厂函数
def create_position_head(method: str, d_model: int, **kwargs) -> nn.Module:
    """
    创建仓位预测头
    
    Args:
        method: 方法名称 ('gumbel_softmax', 'straight_through', 'concrete')
        d_model: 隐藏维度
        **kwargs: 其他参数
        
    Returns:
        仓位预测头模块
    """
    if method == 'gumbel_softmax':
        return GumbelSoftmaxPositionHead(d_model, **kwargs)
    elif method == 'straight_through':
        return StraightThroughPositionHead(d_model, **kwargs)
    elif method == 'concrete':
        return ConcretePositionHead(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown position head method: {method}")

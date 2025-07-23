"""
交易策略和收益率计算模块
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional


class TradingSimulator:
    """
    交易模拟器
    
    模拟基于预测价格和交易策略的交易过程，计算收益率
    """
    
    def __init__(
        self,
        trading_range_min: int = -10,
        trading_range_max: int = 10,
        max_position: int = 10,
        initial_cash: float = 10000.0
    ):
        """
        初始化交易模拟器
        
        Args:
            trading_range_min: 最小交易量（负数表示卖出）
            trading_range_max: 最大交易量（正数表示买入）
            max_position: 最大持仓量（正负各10份）
            initial_cash: 初始现金
        """
        self.trading_range_min = trading_range_min
        self.trading_range_max = trading_range_max
        self.max_position = max_position
        self.initial_cash = initial_cash
    
    def simulate_trading(
        self,
        trading_actions: torch.Tensor,
        prices: torch.Tensor,
        return_details: bool = False
    ) -> torch.Tensor:
        """
        模拟交易过程
        
        Args:
            trading_actions: 交易动作 [batch_size, n_trading_days]
            prices: 价格序列 [batch_size, n_trading_days]
            return_details: 是否返回详细信息
            
        Returns:
            收益率 [batch_size] 或 (收益率, 详细信息)
        """
        batch_size, n_days = trading_actions.shape
        device = trading_actions.device
        
        # 存储每个样本的收益率
        returns = []
        details = [] if return_details else None
        
        for i in range(batch_size):
            actions = trading_actions[i]  # [n_trading_days]
            price_seq = prices[i]  # [n_trading_days]
            
            # 模拟单个样本的交易
            sample_return, sample_detail = self._simulate_single_sample(
                actions, price_seq, return_details
            )
            
            returns.append(sample_return)
            if return_details:
                details.append(sample_detail)
        
        returns_tensor = torch.tensor(returns, device=device, dtype=torch.float32)
        
        if return_details:
            return returns_tensor, details
        else:
            return returns_tensor
    
    def _simulate_single_sample(
        self,
        actions: torch.Tensor,
        prices: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[float, Optional[dict]]:
        """
        模拟单个样本的交易过程
        
        Args:
            actions: 交易动作 [n_trading_days]
            prices: 价格序列 [n_trading_days]
            return_details: 是否返回详细信息
            
        Returns:
            (总收益率, 详细信息)
        """
        n_days = len(actions)
        
        # 初始化状态
        cash = self.initial_cash
        position = 0  # 当前持仓（正数表示多头，负数表示空头）
        total_value_history = []
        position_history = []
        action_history = []
        
        for day in range(n_days):
            # 获取当前天的交易动作和价格
            raw_action = actions[day].item()
            current_price = prices[day].item()
            
            # 将动作四舍五入到整数并限制在范围内
            action = int(round(raw_action))
            action = max(self.trading_range_min, min(self.trading_range_max, action))
            
            # 执行交易
            if action > 0:  # 买入
                # 计算实际可买入数量（考虑持仓限制）
                max_buy = self.max_position - position
                actual_buy = min(action, max_buy)
                actual_buy = max(0, actual_buy)  # 确保非负
                
                # 检查现金是否足够
                required_cash = actual_buy * current_price
                if cash >= required_cash:
                    cash -= required_cash
                    position += actual_buy
                else:
                    # 现金不足，买入能买的最大数量
                    affordable_shares = int(cash / current_price)
                    affordable_shares = min(affordable_shares, actual_buy)
                    if affordable_shares > 0:
                        cash -= affordable_shares * current_price
                        position += affordable_shares
                        
            elif action < 0:  # 卖出
                # 计算实际可卖出数量（考虑持仓限制）
                max_sell = position + self.max_position  # 可以做空到-max_position
                actual_sell = min(-action, max_sell)
                actual_sell = max(0, actual_sell)  # 确保非负
                
                # 执行卖出
                cash += actual_sell * current_price
                position -= actual_sell
            
            # 记录历史
            total_value = cash + position * current_price
            total_value_history.append(total_value)
            position_history.append(position)
            action_history.append(action)
        
        # 计算最终收益率
        final_value = total_value_history[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # 准备详细信息
        detail = None
        if return_details:
            detail = {
                'initial_cash': self.initial_cash,
                'final_value': final_value,
                'total_return': total_return,
                'position_history': position_history,
                'action_history': action_history,
                'total_value_history': total_value_history,
                'final_position': position,
                'final_cash': cash
            }
        
        return total_return, detail
    
    def compute_sharpe_ratio(
        self,
        returns: torch.Tensor,
        risk_free_rate: float = 0.02
    ) -> torch.Tensor:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列 [batch_size]
            risk_free_rate: 无风险利率
            
        Returns:
            夏普比率 [batch_size]
        """
        excess_returns = returns - risk_free_rate
        return excess_returns / (torch.std(returns) + 1e-8)


class TradingLoss(nn.Module):
    """
    交易策略损失函数
    
    结合收益率最大化和风险控制
    """
    
    def __init__(
        self,
        simulator: TradingSimulator,
        return_weight: float = 1.0,
        risk_weight: float = 0.1,
        action_regularization: float = 0.01
    ):
        """
        初始化交易损失函数
        
        Args:
            simulator: 交易模拟器
            return_weight: 收益率权重
            risk_weight: 风险权重
            action_regularization: 动作正则化权重
        """
        super().__init__()
        self.simulator = simulator
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.action_regularization = action_regularization
    
    def forward(
        self,
        trading_predictions: torch.Tensor,
        prices: torch.Tensor
    ) -> torch.Tensor:
        """
        计算交易损失
        
        Args:
            trading_predictions: 预测的交易动作 [batch_size, n_trading_days]
            prices: 价格序列 [batch_size, n_trading_days]
            
        Returns:
            交易损失
        """
        # 模拟交易获得收益率
        returns = self.simulator.simulate_trading(trading_predictions, prices)
        
        # 收益率损失（负收益率，因为我们要最大化收益）
        return_loss = -torch.mean(returns)
        
        # 风险损失（收益率方差）
        risk_loss = torch.var(returns)
        
        # 动作正则化（减少过度交易）
        action_reg = torch.mean(torch.abs(trading_predictions))
        
        # 总损失
        total_loss = (
            self.return_weight * return_loss +
            self.risk_weight * risk_loss +
            self.action_regularization * action_reg
        )
        
        return total_loss


def create_trading_simulator(args) -> TradingSimulator:
    """
    根据模型配置创建交易模拟器
    
    Args:
        args: 模型配置
        
    Returns:
        交易模拟器实例
    """
    return TradingSimulator(
        trading_range_min=args.trading_range_min,
        trading_range_max=args.trading_range_max,
        max_position=args.trading_range_max,  # 使用最大交易量作为最大持仓
        initial_cash=10000.0
    )


def create_trading_loss(args, simulator: TradingSimulator) -> TradingLoss:
    """
    根据模型配置创建交易损失函数
    
    Args:
        args: 模型配置
        simulator: 交易模拟器
        
    Returns:
        交易损失函数实例
    """
    return TradingLoss(
        simulator=simulator,
        return_weight=1.0,
        risk_weight=0.1,
        action_regularization=0.01
    )

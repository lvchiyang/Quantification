"""
策略网络损失函数
包含：相对基准收益、风险成本、机会成本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import numpy as np

# 导入市场分类器
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 市场分类器
class ComprehensiveMarketClassifier:
    """综合市场分类器"""

    def __init__(self, bull_threshold: float = 0.008, bear_threshold: float = -0.008):
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def classify_market(self, returns: torch.Tensor) -> str:
        """
        分类市场状态

        Args:
            returns: [seq_len] 收益率序列

        Returns:
            市场类型: 'bull', 'bear', 'sideways'
        """
        mean_return = torch.mean(returns)

        if mean_return > self.bull_threshold:
            return 'bull'
        elif mean_return < self.bear_threshold:
            return 'bear'
        else:
            return 'sideways'

    def get_optimal_benchmark(self, market_type: str) -> Dict[str, Any]:
        """获取最优基准策略"""
        if market_type == 'bull':
            return {'position': 0.8, 'description': '牛市高仓位策略'}
        elif market_type == 'bear':
            return {'position': 0.2, 'description': '熊市低仓位策略'}
        else:
            return {'position': 0.5, 'description': '震荡市中性策略'}


def create_market_classifier(args) -> ComprehensiveMarketClassifier:
    """创建市场分类器"""
    return ComprehensiveMarketClassifier(
        bull_threshold=getattr(args, 'bull_threshold', 0.008),
        bear_threshold=getattr(args, 'bear_threshold', -0.008)
    )


class StrategyLoss(nn.Module):
    """
    策略网络专用损失函数
    组成：相对基准收益 + 风险成本 + 机会成本
    """
    
    def __init__(
        self,
        market_classifier: ComprehensiveMarketClassifier,
        relative_return_weight: float = 1.0,
        risk_cost_weight: float = 0.2,
        opportunity_cost_weight: float = 0.1
    ):
        super().__init__()
        self.market_classifier = market_classifier
        self.relative_return_weight = relative_return_weight
        self.risk_cost_weight = risk_cost_weight
        self.opportunity_cost_weight = opportunity_cost_weight
    
    def forward(
        self,
        position_predictions: torch.Tensor,  # [batch_size, seq_len, 1]
        next_day_returns: torch.Tensor       # [batch_size, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        计算策略损失
        
        Args:
            position_predictions: 仓位预测 [batch_size, seq_len, 1]
            next_day_returns: 次日收益率 [batch_size, seq_len]
            
        Returns:
            损失字典
        """
        batch_size = position_predictions.shape[0]
        
        total_loss = 0.0
        total_relative_return = 0.0
        total_risk_cost = 0.0
        total_opportunity_cost = 0.0
        
        # 逐样本计算损失
        for b in range(batch_size):
            positions = position_predictions[b, :, 0]  # [seq_len]
            returns = next_day_returns[b, :]           # [seq_len]
            
            # 1. 判断市场状态
            market_type = self.market_classifier.classify_market(returns)
            
            # 2. 计算相对基准收益
            relative_return_loss = self._calculate_relative_return_loss(
                positions, returns, market_type
            )
            
            # 3. 计算风险成本
            risk_cost = self._calculate_risk_cost(positions, returns)
            
            # 4. 计算机会成本
            opportunity_cost = self._calculate_opportunity_cost(
                positions, returns, market_type
            )
            
            # 5. 综合损失
            sample_loss = (
                self.relative_return_weight * relative_return_loss +
                self.risk_cost_weight * risk_cost +
                self.opportunity_cost_weight * opportunity_cost
            )
            
            total_loss += sample_loss
            total_relative_return += relative_return_loss
            total_risk_cost += risk_cost
            total_opportunity_cost += opportunity_cost
        
        # 平均化
        avg_loss = total_loss / batch_size
        avg_relative_return = total_relative_return / batch_size
        avg_risk_cost = total_risk_cost / batch_size
        avg_opportunity_cost = total_opportunity_cost / batch_size
        
        return {
            'total_loss': avg_loss,
            'relative_return_loss': avg_relative_return,
            'risk_cost': avg_risk_cost,
            'opportunity_cost': avg_opportunity_cost
        }
    
    def _calculate_relative_return_loss(
        self,
        positions: torch.Tensor,
        returns: torch.Tensor,
        market_type: str
    ) -> torch.Tensor:
        """
        计算相对基准收益损失
        
        Args:
            positions: [seq_len] 仓位序列
            returns: [seq_len] 收益率序列
            market_type: 市场类型
            
        Returns:
            相对收益损失
        """
        # 标准化仓位到[0,1]
        normalized_positions = positions / 10.0
        
        # 计算策略收益
        strategy_returns = normalized_positions * returns
        strategy_cumulative_return = torch.sum(strategy_returns)
        
        # 获取基准策略
        benchmark_config = self.market_classifier.get_optimal_benchmark(market_type)
        benchmark_position = benchmark_config['position']
        
        # 计算基准收益
        benchmark_returns = benchmark_position * returns
        benchmark_cumulative_return = torch.sum(benchmark_returns)
        
        # 相对收益（策略收益 - 基准收益）
        relative_return = strategy_cumulative_return - benchmark_cumulative_return
        
        # 损失 = 负相对收益（最大化相对收益）
        return -relative_return
    
    def _calculate_risk_cost(
        self,
        positions: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        计算风险成本
        
        Args:
            positions: [seq_len] 仓位序列
            returns: [seq_len] 收益率序列
            
        Returns:
            风险成本
        """
        normalized_positions = positions / 10.0
        strategy_returns = normalized_positions * returns
        
        # 1. 收益波动率成本
        volatility_cost = torch.std(strategy_returns)
        
        # 2. 最大回撤成本
        cumulative_returns = torch.cumsum(strategy_returns, dim=0)
        running_max = torch.cummax(cumulative_returns, dim=0)[0]
        drawdowns = running_max - cumulative_returns
        max_drawdown_cost = torch.max(drawdowns)
        
        # 3. 仓位波动成本（避免频繁调仓）
        if len(positions) > 1:
            position_volatility_cost = torch.std(torch.diff(normalized_positions))
        else:
            position_volatility_cost = torch.tensor(0.0, device=positions.device)
        
        # 综合风险成本
        total_risk_cost = (
            0.4 * volatility_cost +
            0.4 * max_drawdown_cost +
            0.2 * position_volatility_cost
        )
        
        return total_risk_cost
    
    def _calculate_opportunity_cost(
        self,
        positions: torch.Tensor,
        returns: torch.Tensor,
        market_type: str
    ) -> torch.Tensor:
        """
        计算机会成本
        
        Args:
            positions: [seq_len] 仓位序列
            returns: [seq_len] 收益率序列
            market_type: 市场类型
            
        Returns:
            机会成本
        """
        normalized_positions = positions / 10.0
        
        if market_type == 'bull':
            # 牛市：低仓位的机会成本
            upward_days = returns > 0
            if torch.any(upward_days):
                missed_opportunities = (1.0 - normalized_positions)[upward_days] * returns[upward_days]
                opportunity_cost = torch.mean(missed_opportunities)
            else:
                opportunity_cost = torch.tensor(0.0, device=positions.device)
                
        elif market_type == 'bear':
            # 熊市：高仓位的风险成本
            downward_days = returns < 0
            if torch.any(downward_days):
                excessive_risks = normalized_positions[downward_days] * torch.abs(returns[downward_days])
                opportunity_cost = torch.mean(excessive_risks)
            else:
                opportunity_cost = torch.tensor(0.0, device=positions.device)
                
        else:  # sideways
            # 震荡市：过度交易的成本
            if len(positions) > 1:
                position_changes = torch.abs(torch.diff(normalized_positions))
                opportunity_cost = torch.mean(position_changes) * 0.1  # 交易成本
            else:
                opportunity_cost = torch.tensor(0.0, device=positions.device)
        
        return opportunity_cost


class StrategyEvaluator:
    """
    策略评估器
    用于计算各种策略性能指标
    """
    
    def __init__(self):
        pass
    
    def evaluate_strategy(
        self,
        position_predictions: torch.Tensor,
        next_day_returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        评估策略性能
        
        Args:
            position_predictions: [batch_size, seq_len, 1] 仓位预测
            next_day_returns: [batch_size, seq_len] 次日收益
            
        Returns:
            性能指标字典
        """
        batch_size = position_predictions.shape[0]
        
        # 计算每个样本的指标
        cumulative_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        volatilities = []
        
        with torch.no_grad():
            for b in range(batch_size):
                positions = position_predictions[b, :, 0] / 10.0  # 标准化
                returns = next_day_returns[b, :]
                
                # 计算每日收益
                daily_returns = positions * returns
                
                # 累积收益
                cumulative_return = torch.sum(daily_returns).item()
                cumulative_returns.append(cumulative_return)
                
                # 波动率
                volatility = torch.std(daily_returns).item()
                volatilities.append(volatility)
                
                # 夏普比率
                if volatility > 0:
                    sharpe_ratio = torch.mean(daily_returns).item() / volatility * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
                sharpe_ratios.append(sharpe_ratio)
                
                # 最大回撤
                portfolio_values = [1.0]
                for daily_return in daily_returns:
                    portfolio_values.append(portfolio_values[-1] * (1.0 + daily_return.item()))
                
                peak = portfolio_values[0]
                max_drawdown = 0.0
                for value in portfolio_values[1:]:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                max_drawdowns.append(max_drawdown)
        
        return {
            'mean_cumulative_return': np.mean(cumulative_returns),
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'mean_volatility': np.mean(volatilities),
            'return_std': np.std(cumulative_returns)
        }


def create_strategy_loss(args, market_classifier: ComprehensiveMarketClassifier) -> StrategyLoss:
    """
    创建策略损失函数

    Args:
        args: 模型配置
        market_classifier: 市场分类器

    Returns:
        策略损失函数实例
    """
    return StrategyLoss(
        market_classifier=market_classifier,
        relative_return_weight=getattr(args, 'relative_return_weight', 1.0),
        risk_cost_weight=getattr(args, 'risk_cost_weight', 0.2),
        opportunity_cost_weight=getattr(args, 'opportunity_cost_weight', 0.1)
    )

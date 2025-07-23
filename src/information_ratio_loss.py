"""
信息比率损失函数
基于市场分类的自适应基准比较 + 机会成本计算
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .market_classifier import ComprehensiveMarketClassifier


class InformationRatioLoss(nn.Module):
    """
    信息比率 + 机会成本损失函数
    
    根据市场环境选择合适的基准，计算信息比率和机会成本
    """
    
    def __init__(
        self, 
        market_classifier: ComprehensiveMarketClassifier,
        information_ratio_weight: float = 1.0,
        opportunity_cost_weight: float = 0.1,
        risk_adjustment_weight: float = 0.05
    ):
        super().__init__()
        self.market_classifier = market_classifier
        self.information_ratio_weight = information_ratio_weight
        self.opportunity_cost_weight = opportunity_cost_weight
        self.risk_adjustment_weight = risk_adjustment_weight
    
    def forward(
        self, 
        position_predictions: torch.Tensor,  # [batch_size, 20, 1] 或 [batch_size, 1]
        next_day_returns: torch.Tensor       # [batch_size, 20] 或 [batch_size]
    ) -> Dict[str, torch.Tensor]:
        """
        计算信息比率损失
        
        Args:
            position_predictions: 仓位预测 [batch_size, seq_len, 1]
            next_day_returns: 次日收益率 [batch_size, seq_len]
            
        Returns:
            损失字典
        """
        batch_size = position_predictions.shape[0]
        
        # 处理维度
        if position_predictions.dim() == 2:  # [batch_size, 1]
            position_predictions = position_predictions.unsqueeze(1)  # [batch_size, 1, 1]
        if next_day_returns.dim() == 1:  # [batch_size]
            next_day_returns = next_day_returns.unsqueeze(1)  # [batch_size, 1]
        
        seq_len = position_predictions.shape[1]
        
        total_loss = 0.0
        total_information_ratio = 0.0
        total_opportunity_cost = 0.0
        total_risk_penalty = 0.0
        
        for b in range(batch_size):
            # 获取单个样本的数据
            positions = position_predictions[b, :, 0]  # [seq_len]
            returns = next_day_returns[b, :]           # [seq_len]
            
            # 1. 判断市场状态
            market_type = self.market_classifier.classify_market(returns)
            
            # 2. 获取对应的基准策略
            benchmark_config = self.market_classifier.get_optimal_benchmark(market_type)
            benchmark_returns = self.market_classifier.calculate_benchmark_returns(
                returns, benchmark_config
            )
            
            # 3. 计算策略收益
            normalized_positions = positions / 10.0  # 标准化到[0,1]
            strategy_returns = normalized_positions * returns
            
            # 4. 计算信息比率
            information_ratio = self._calculate_information_ratio(
                strategy_returns, benchmark_returns
            )
            
            # 5. 计算机会成本
            opportunity_cost = self._calculate_opportunity_cost(
                positions, returns, market_type
            )
            
            # 6. 计算风险惩罚
            risk_penalty = self._calculate_risk_penalty(strategy_returns)
            
            # 7. 综合损失
            sample_loss = (
                -self.information_ratio_weight * information_ratio +
                self.opportunity_cost_weight * opportunity_cost +
                self.risk_adjustment_weight * risk_penalty
            )
            
            total_loss += sample_loss
            total_information_ratio += information_ratio
            total_opportunity_cost += opportunity_cost
            total_risk_penalty += risk_penalty
        
        # 平均化
        avg_loss = total_loss / batch_size
        avg_information_ratio = total_information_ratio / batch_size
        avg_opportunity_cost = total_opportunity_cost / batch_size
        avg_risk_penalty = total_risk_penalty / batch_size
        
        return {
            'total_loss': avg_loss,
            'information_ratio': avg_information_ratio,
            'opportunity_cost': avg_opportunity_cost,
            'risk_penalty': avg_risk_penalty
        }
    
    def _calculate_information_ratio(
        self, 
        strategy_returns: torch.Tensor, 
        benchmark_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        计算信息比率
        
        Args:
            strategy_returns: 策略收益序列
            benchmark_returns: 基准收益序列
            
        Returns:
            信息比率
        """
        # 超额收益
        excess_returns = strategy_returns - benchmark_returns
        
        # 信息比率 = 超额收益均值 / 超额收益标准差
        mean_excess = torch.mean(excess_returns)
        std_excess = torch.std(excess_returns) + 1e-8  # 避免除零
        
        information_ratio = mean_excess / std_excess
        
        return information_ratio
    
    def _calculate_opportunity_cost(
        self, 
        positions: torch.Tensor, 
        returns: torch.Tensor,
        market_type: str
    ) -> torch.Tensor:
        """
        计算机会成本
        
        Args:
            positions: 仓位序列 [seq_len]
            returns: 收益率序列 [seq_len]
            market_type: 市场类型
            
        Returns:
            机会成本
        """
        normalized_positions = positions / 10.0  # 标准化到[0,1]
        
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
    
    def _calculate_risk_penalty(self, strategy_returns: torch.Tensor) -> torch.Tensor:
        """
        计算风险惩罚
        
        Args:
            strategy_returns: 策略收益序列
            
        Returns:
            风险惩罚
        """
        # 收益波动率作为风险指标
        volatility = torch.std(strategy_returns)
        
        # 最大回撤
        cumulative_returns = torch.cumsum(strategy_returns, dim=0)
        running_max = torch.cummax(cumulative_returns, dim=0)[0]
        drawdowns = running_max - cumulative_returns
        max_drawdown = torch.max(drawdowns)
        
        # 综合风险惩罚
        risk_penalty = 0.5 * volatility + 0.5 * max_drawdown
        
        return risk_penalty


class MultiObjectiveTradingLoss(nn.Module):
    """
    多目标交易损失函数
    结合信息比率损失和其他目标
    """
    
    def __init__(
        self,
        market_classifier: ComprehensiveMarketClassifier,
        price_loss_weight: float = 1.0,
        information_ratio_weight: float = 1.0,
        opportunity_cost_weight: float = 0.1,
        risk_adjustment_weight: float = 0.05
    ):
        super().__init__()
        self.price_loss_weight = price_loss_weight
        self.information_ratio_loss = InformationRatioLoss(
            market_classifier=market_classifier,
            information_ratio_weight=information_ratio_weight,
            opportunity_cost_weight=opportunity_cost_weight,
            risk_adjustment_weight=risk_adjustment_weight
        )
    
    def forward(
        self,
        price_predictions: torch.Tensor,
        price_targets: torch.Tensor,
        position_predictions: torch.Tensor,
        next_day_returns: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算多目标损失
        
        Args:
            price_predictions: 价格预测
            price_targets: 价格目标
            position_predictions: 仓位预测
            next_day_returns: 次日收益率
            
        Returns:
            损失字典
        """
        # 1. 价格预测损失
        price_loss = nn.MSELoss()(price_predictions, price_targets)
        
        # 2. 信息比率损失
        ir_loss_dict = self.information_ratio_loss(position_predictions, next_day_returns)
        
        # 3. 总损失
        total_loss = (
            self.price_loss_weight * price_loss +
            ir_loss_dict['total_loss']
        )
        
        return {
            'total_loss': total_loss,
            'price_loss': price_loss,
            'information_ratio_loss': ir_loss_dict['total_loss'],
            'information_ratio': ir_loss_dict['information_ratio'],
            'opportunity_cost': ir_loss_dict['opportunity_cost'],
            'risk_penalty': ir_loss_dict['risk_penalty']
        }


def create_information_ratio_loss(args, market_classifier: ComprehensiveMarketClassifier) -> InformationRatioLoss:
    """
    根据配置创建信息比率损失函数
    
    Args:
        args: 模型配置
        market_classifier: 市场分类器
        
    Returns:
        信息比率损失函数实例
    """
    return InformationRatioLoss(
        market_classifier=market_classifier,
        information_ratio_weight=args.information_ratio_weight,
        opportunity_cost_weight=args.opportunity_cost_weight,
        risk_adjustment_weight=args.risk_adjustment_weight
    )

"""
市场状态分类器
综合多种方法判断牛市、熊市、震荡市
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any


class SimpleMarketClassifier:
    """基于统计特征的简单市场分类器"""
    
    def __init__(self, bull_threshold: float = 0.008, bear_threshold: float = -0.008):
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
    
    def classify(self, returns: torch.Tensor) -> str:
        """
        基于平均收益率分类
        
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


class TechnicalMarketClassifier:
    """基于技术指标的市场分类器"""
    
    def __init__(self):
        pass
    
    def classify(self, returns: torch.Tensor) -> str:
        """
        基于技术指标分类
        
        Args:
            returns: [seq_len] 收益率序列
            
        Returns:
            市场类型
        """
        # 1. 移动平均趋势
        ma_signal = self._moving_average_signal(returns)
        
        # 2. RSI指标
        rsi_signal = self._rsi_signal(returns)
        
        # 3. 动量指标
        momentum_signal = self._momentum_signal(returns)
        
        # 4. 综合判断
        return self._combine_signals(ma_signal, rsi_signal, momentum_signal)
    
    def _moving_average_signal(self, returns: torch.Tensor) -> str:
        """移动平均信号"""
        if len(returns) < 10:
            return 'neutral'
        
        short_ma = torch.mean(returns[-5:])   # 短期均线
        long_ma = torch.mean(returns[-10:])   # 长期均线
        
        if short_ma > long_ma * 1.2:
            return 'bullish'
        elif short_ma < long_ma * 0.8:
            return 'bearish'
        else:
            return 'neutral'
    
    def _rsi_signal(self, returns: torch.Tensor) -> str:
        """RSI信号"""
        gains = torch.clamp(returns, min=0)
        losses = torch.clamp(-returns, min=0)
        
        avg_gain = torch.mean(gains)
        avg_loss = torch.mean(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        if rsi > 70:
            return 'overbought'
        elif rsi < 30:
            return 'oversold'
        else:
            return 'neutral'
    
    def _momentum_signal(self, returns: torch.Tensor) -> str:
        """动量信号"""
        if len(returns) < 10:
            return 'neutral'
        
        recent_momentum = torch.sum(returns[-5:])
        early_momentum = torch.sum(returns[-10:-5])
        
        if recent_momentum > early_momentum * 1.2:
            return 'accelerating'
        elif recent_momentum < early_momentum * 0.8:
            return 'decelerating'
        else:
            return 'stable'
    
    def _combine_signals(self, ma_signal: str, rsi_signal: str, momentum_signal: str) -> str:
        """综合多个信号"""
        bull_score = 0
        bear_score = 0
        
        # MA信号评分
        if ma_signal == 'bullish':
            bull_score += 2
        elif ma_signal == 'bearish':
            bear_score += 2
        
        # RSI信号评分
        if rsi_signal == 'oversold':
            bull_score += 1
        elif rsi_signal == 'overbought':
            bear_score += 1
        
        # 动量信号评分
        if momentum_signal == 'accelerating':
            bull_score += 1
        elif momentum_signal == 'decelerating':
            bear_score += 1
        
        # 最终判断
        if bull_score > bear_score + 1:
            return 'bull'
        elif bear_score > bull_score + 1:
            return 'bear'
        else:
            return 'sideways'


class AdaptiveMarketClassifier:
    """自适应阈值的市场分类器"""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.historical_returns = []
        self.historical_volatilities = []
    
    def classify(self, returns: torch.Tensor) -> str:
        """
        基于历史数据自适应分类
        
        Args:
            returns: [seq_len] 收益率序列
            
        Returns:
            市场类型
        """
        # 更新历史数据
        current_return = torch.mean(returns).item()
        current_volatility = torch.std(returns).item()
        
        self.historical_returns.append(current_return)
        self.historical_volatilities.append(current_volatility)
        
        # 保持历史长度
        if len(self.historical_returns) > self.history_length:
            self.historical_returns = self.historical_returns[-self.history_length:]
            self.historical_volatilities = self.historical_volatilities[-self.history_length:]
        
        # 计算动态阈值
        if len(self.historical_returns) >= 10:
            mean_historical = np.mean(self.historical_returns)
            std_historical = np.std(self.historical_returns)
            
            bull_threshold = mean_historical + 0.5 * std_historical
            bear_threshold = mean_historical - 0.5 * std_historical
        else:
            bull_threshold = 0.01
            bear_threshold = -0.01
        
        # 分类
        if current_return > bull_threshold:
            return 'bull'
        elif current_return < bear_threshold:
            return 'bear'
        else:
            return 'sideways'


class ComprehensiveMarketClassifier:
    """综合市场分类器"""
    
    def __init__(self, bull_threshold: float = 0.008, bear_threshold: float = -0.008):
        self.simple_classifier = SimpleMarketClassifier(bull_threshold, bear_threshold)
        self.technical_classifier = TechnicalMarketClassifier()
        self.adaptive_classifier = AdaptiveMarketClassifier()
    
    def classify_market(self, returns: torch.Tensor) -> str:
        """
        综合多种方法的市场分类
        
        Args:
            returns: [seq_len] 收益率序列
            
        Returns:
            市场类型: 'bull', 'bear', 'sideways'
        """
        # 获取各种分类结果
        simple_result = self.simple_classifier.classify(returns)
        technical_result = self.technical_classifier.classify(returns)
        adaptive_result = self.adaptive_classifier.classify(returns)
        
        # 投票机制
        votes = [simple_result, technical_result, adaptive_result]
        
        # 统计投票结果
        bull_votes = votes.count('bull')
        bear_votes = votes.count('bear')
        sideways_votes = votes.count('sideways')
        
        # 返回多数票结果
        if bull_votes > bear_votes and bull_votes > sideways_votes:
            return 'bull'
        elif bear_votes > bull_votes and bear_votes > sideways_votes:
            return 'bear'
        else:
            return 'sideways'
    
    def get_optimal_benchmark(self, market_type: str) -> Dict[str, Any]:
        """
        根据市场类型获取最优基准策略
        
        Args:
            market_type: 市场类型
            
        Returns:
            基准策略配置
        """
        if market_type == 'bull':
            return {
                'name': 'buy_and_hold',
                'position': 1.0,  # 满仓
                'description': '牛市买入持有策略'
            }
        elif market_type == 'bear':
            return {
                'name': 'conservative',
                'position': 0.3,  # 保守仓位
                'description': '熊市保守策略'
            }
        else:  # sideways
            return {
                'name': 'balanced',
                'position': 0.5,  # 平衡仓位
                'description': '震荡市平衡策略'
            }
    
    def calculate_benchmark_returns(
        self, 
        returns: torch.Tensor, 
        benchmark_config: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算基准策略的收益序列
        
        Args:
            returns: [seq_len] 市场收益率
            benchmark_config: 基准策略配置
            
        Returns:
            基准策略的收益序列
        """
        benchmark_position = benchmark_config['position']
        return returns * benchmark_position


def create_market_classifier(args) -> ComprehensiveMarketClassifier:
    """
    根据配置创建市场分类器
    
    Args:
        args: 模型配置
        
    Returns:
        市场分类器实例
    """
    return ComprehensiveMarketClassifier(
        bull_threshold=args.bull_threshold,
        bear_threshold=args.bear_threshold
    )

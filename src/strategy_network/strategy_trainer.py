"""
策略网络训练器
专门用于训练GRU策略网络
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np

from .gru_strategy import GRUStrategyNetwork
from .strategy_loss import StrategyLoss, StrategyEvaluator, create_market_classifier


class StrategyTrainer:
    """
    策略网络训练器
    
    训练GRU策略网络，使用预训练的价格网络特征
    """
    
    def __init__(
        self,
        strategy_network: GRUStrategyNetwork,
        strategy_loss: StrategyLoss
    ):
        """
        初始化训练器
        
        Args:
            strategy_network: GRU策略网络
            strategy_loss: 策略损失函数
        """
        self.strategy_network = strategy_network
        self.strategy_loss = strategy_loss
        self.evaluator = StrategyEvaluator()
        
        # 检查网络是否支持序列训练
        if not hasattr(strategy_network, 'forward_sequence'):
            raise ValueError("策略网络必须支持 forward_sequence 方法")
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        执行一个训练步骤
        
        Args:
            batch_data: 批次数据
                - price_features: [batch_size, seq_len, feature_dim] 价格网络特征
                - next_day_returns: [batch_size, seq_len] 次日收益率
                
        Returns:
            损失和指标字典
        """
        price_features = batch_data['price_features']      # [batch_size, seq_len, feature_dim]
        next_day_returns = batch_data['next_day_returns']  # [batch_size, seq_len]
        
        batch_size, seq_len, feature_dim = price_features.shape
        device = price_features.device
        
        # 1. 策略网络前向传播（20天序列）
        strategy_outputs = self.strategy_network.forward_sequence(price_features)
        
        # 2. 提取仓位预测
        position_predictions = self.strategy_network.get_position_predictions(
            strategy_outputs['all_position_outputs']
        )  # [batch_size, seq_len, 1]
        
        # 3. 计算策略损失
        loss_dict = self.strategy_loss(position_predictions, next_day_returns)
        
        # 4. 计算评估指标
        eval_metrics = self.evaluator.evaluate_strategy(position_predictions, next_day_returns)
        
        # 5. 合并结果
        result = {
            'loss_tensor': loss_dict['total_loss'],  # 用于反向传播
            'total_loss': loss_dict['total_loss'].item() if torch.is_tensor(loss_dict['total_loss']) else loss_dict['total_loss'],
            'relative_return_loss': loss_dict['relative_return_loss'].item() if torch.is_tensor(loss_dict['relative_return_loss']) else loss_dict['relative_return_loss'],
            'risk_cost': loss_dict['risk_cost'].item() if torch.is_tensor(loss_dict['risk_cost']) else loss_dict['risk_cost'],
            'opportunity_cost': loss_dict['opportunity_cost'].item() if torch.is_tensor(loss_dict['opportunity_cost']) else loss_dict['opportunity_cost'],
            **eval_metrics
        }
        
        return result
    
    def validate_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证步骤
        
        Args:
            batch_data: 验证数据
            
        Returns:
            验证指标
        """
        self.strategy_network.eval()
        
        with torch.no_grad():
            result = self.train_step(batch_data)
        
        # 移除loss_tensor（验证时不需要）
        if 'loss_tensor' in result:
            del result['loss_tensor']
        
        return result
    
    def predict_sequence(
        self,
        price_features_sequence: torch.Tensor,
        return_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        预测序列仓位
        
        Args:
            price_features_sequence: [batch_size, seq_len, feature_dim] 特征序列
            return_states: 是否返回策略状态
            
        Returns:
            预测结果字典
        """
        self.strategy_network.eval()
        
        with torch.no_grad():
            strategy_outputs = self.strategy_network.forward_sequence(price_features_sequence)
            position_predictions = self.strategy_network.get_position_predictions(
                strategy_outputs['all_position_outputs']
            )
            
            result = {
                'position_predictions': position_predictions,
                'expected_positions': strategy_outputs['all_expected_positions']
            }
            
            if return_states:
                result['strategy_states'] = strategy_outputs['all_strategy_states']
                result['final_strategy_state'] = strategy_outputs['final_strategy_state']
            
            return result


def create_strategy_batches(
    price_features_list: torch.Tensor,    # [n_sequences, seq_len, feature_dim]
    next_day_returns: torch.Tensor,       # [n_sequences, seq_len]
    batch_size: int = 4
) -> List[Dict[str, torch.Tensor]]:
    """
    创建策略训练批次数据
    
    Args:
        price_features_list: 价格特征序列
        next_day_returns: 次日收益序列
        batch_size: 批次大小
        
    Returns:
        批次数据列表
    """
    n_sequences = price_features_list.shape[0]
    batches = []
    
    for i in range(0, n_sequences, batch_size):
        end_idx = min(i + batch_size, n_sequences)
        
        batch_data = {
            'price_features': price_features_list[i:end_idx],
            'next_day_returns': next_day_returns[i:end_idx]
        }
        
        batches.append(batch_data)
    
    return batches


class StrategyTrainingPipeline:
    """
    策略训练流水线
    整合价格网络特征提取和策略网络训练
    """
    
    def __init__(
        self,
        price_network,
        strategy_network: GRUStrategyNetwork,
        strategy_loss: StrategyLoss
    ):
        """
        初始化训练流水线
        
        Args:
            price_network: 预训练的价格网络
            strategy_network: GRU策略网络
            strategy_loss: 策略损失函数
        """
        self.price_network = price_network
        self.strategy_trainer = StrategyTrainer(strategy_network, strategy_loss)
        
        # 冻结价格网络参数
        for param in self.price_network.parameters():
            param.requires_grad = False
        self.price_network.eval()
    
    def extract_features_batch(
        self,
        financial_data_batch: torch.Tensor  # [batch_size, seq_len, 180, 11]
    ) -> torch.Tensor:
        """
        批量提取价格网络特征
        
        Args:
            financial_data_batch: [batch_size, seq_len, 180, 11] 金融数据批次
            
        Returns:
            price_features: [batch_size, seq_len, feature_dim] 提取的特征
        """
        batch_size, seq_len = financial_data_batch.shape[:2]
        
        # 重塑为 [batch_size * seq_len, 180, 11]
        reshaped_data = financial_data_batch.view(-1, *financial_data_batch.shape[2:])
        
        # 批量提取特征
        with torch.no_grad():
            features = self.price_network.extract_features(reshaped_data)  # [batch_size * seq_len, feature_dim]
        
        # 重塑回 [batch_size, seq_len, feature_dim]
        feature_dim = features.shape[-1]
        price_features = features.view(batch_size, seq_len, feature_dim)
        
        return price_features
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        完整的训练步骤：特征提取 + 策略训练
        
        Args:
            batch_data: 包含金融数据和收益的批次
                - financial_data: [batch_size, seq_len, 180, 11]
                - next_day_returns: [batch_size, seq_len]
                
        Returns:
            训练结果
        """
        # 1. 提取价格网络特征
        price_features = self.extract_features_batch(batch_data['financial_data'])
        
        # 2. 准备策略训练数据
        strategy_batch = {
            'price_features': price_features,
            'next_day_returns': batch_data['next_day_returns']
        }
        
        # 3. 策略网络训练
        return self.strategy_trainer.train_step(strategy_batch)
    
    def validate_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证步骤
        """
        # 1. 提取价格网络特征
        price_features = self.extract_features_batch(batch_data['financial_data'])
        
        # 2. 准备策略验证数据
        strategy_batch = {
            'price_features': price_features,
            'next_day_returns': batch_data['next_day_returns']
        }
        
        # 3. 策略网络验证
        return self.strategy_trainer.validate_step(strategy_batch)

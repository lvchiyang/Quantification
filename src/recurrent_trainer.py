"""
递归策略训练器
实现20天递归状态更新的训练方式，避免内存爆炸
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np


class RecurrentStrategyTrainer:
    """
    递归策略训练器
    
    实现20天滑动窗口的递归状态更新训练：
    - 每天进行一次前向传播
    - 策略状态递归更新
    - 最终基于累积信息计算损失
    """
    
    def __init__(self, model, use_information_ratio: bool = True):
        """
        初始化训练器
        
        Args:
            model: FinancialTransformer模型
            use_information_ratio: 是否使用信息比率损失
        """
        self.model = model
        self.use_information_ratio = use_information_ratio
        
        # 检查模型是否支持递归训练
        if not hasattr(model, 'forward_single_day'):
            raise ValueError("模型必须支持 forward_single_day 方法")
        
        if use_information_ratio and not hasattr(model, 'multi_objective_loss_fn'):
            raise ValueError("模型必须包含 multi_objective_loss_fn 用于信息比率损失")
    
    def train_step(self, sliding_window_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        执行一个递归训练步骤
        
        Args:
            sliding_window_data: 滑动窗口数据
                - features: [batch_size, 20, 180, 11] 20次的180天历史数据
                - price_targets: [batch_size, 20, 7] 20次的7天价格目标
                - next_day_returns: [batch_size, 20] 20天的次日收益
                
        Returns:
            损失和指标字典
        """
        features = sliding_window_data['features']          # [batch_size, 20, 180, 11]
        price_targets = sliding_window_data['price_targets'] # [batch_size, 20, 7]
        next_day_returns = sliding_window_data['next_day_returns'] # [batch_size, 20]
        
        batch_size, n_slides = features.shape[:2]
        device = features.device
        
        # 初始化策略状态
        if hasattr(self.model, 'strategy_state_init') and self.model.strategy_state_init is not None:
            strategy_state = self.model.strategy_state_init.unsqueeze(0).expand(
                batch_size, -1
            ).contiguous()
        else:
            strategy_state = None
        
        # 存储20次预测结果
        all_price_predictions = []
        all_position_predictions = []
        all_strategy_states = []
        
        # 累积价格损失（不保存梯度）
        total_price_loss = 0.0
        
        # 逐个进行20次预测
        for slide in range(n_slides):
            slide_features = features[:, slide, :, :]  # [batch_size, 180, 11]
            slide_price_targets = price_targets[:, slide, :]  # [batch_size, 7]
            
            # 单日预测（保持梯度）
            outputs = self.model.forward_single_day(
                financial_data=slide_features,
                strategy_state=strategy_state,
                return_dict=True
            )
            
            # 提取预测结果
            price_pred = outputs['price_predictions']
            position_pred = outputs.get('position_predictions', None)
            new_strategy_state = outputs.get('strategy_state', None)
            
            # 价格损失（立即计算，不累积梯度）
            with torch.no_grad():
                day_price_loss = nn.MSELoss()(price_pred, slide_price_targets)
                total_price_loss += day_price_loss.item()
            
            # 保存预测结果（保持梯度）
            all_price_predictions.append(price_pred)
            if position_pred is not None:
                all_position_predictions.append(position_pred)
            if new_strategy_state is not None:
                all_strategy_states.append(new_strategy_state)
            
            # 更新策略状态
            strategy_state = new_strategy_state
        
        # 堆叠预测结果
        stacked_price_predictions = torch.stack(all_price_predictions, dim=1)  # [batch_size, 20, 7]
        
        if all_position_predictions:
            stacked_position_predictions = torch.stack(all_position_predictions, dim=1)  # [batch_size, 20, 1]
        else:
            stacked_position_predictions = None
        
        # 计算最终损失
        if self.use_information_ratio and stacked_position_predictions is not None:
            # 使用信息比率损失
            loss_dict = self.model.multi_objective_loss_fn(
                price_predictions=stacked_price_predictions,
                price_targets=price_targets,
                position_predictions=stacked_position_predictions,
                next_day_returns=next_day_returns
            )
            
            # 添加状态正则化
            if all_strategy_states:
                final_state = all_strategy_states[-1]
                state_reg = self.model.args.state_regularization_weight * torch.mean(final_state ** 2)
                loss_dict['total_loss'] = loss_dict['total_loss'] + state_reg
                loss_dict['state_regularization'] = state_reg.item()
            
        else:
            # 使用简单的价格预测损失
            price_loss = nn.MSELoss()(stacked_price_predictions, price_targets)
            loss_dict = {
                'total_loss': price_loss,
                'price_loss': price_loss.item(),
                'information_ratio_loss': 0.0,
                'information_ratio': 0.0,
                'opportunity_cost': 0.0,
                'risk_penalty': 0.0
            }
        
        # 计算额外的评估指标
        evaluation_metrics = self._calculate_evaluation_metrics(
            stacked_position_predictions, next_day_returns
        )
        
        # 合并结果
        result = {
            'loss_tensor': loss_dict['total_loss'],  # 用于反向传播
            'total_loss': loss_dict['total_loss'].item() if torch.is_tensor(loss_dict['total_loss']) else loss_dict['total_loss'],
            'price_loss': loss_dict.get('price_loss', total_price_loss / n_slides),
            'information_ratio_loss': loss_dict.get('information_ratio_loss', 0.0),
            'information_ratio': loss_dict.get('information_ratio', 0.0),
            'opportunity_cost': loss_dict.get('opportunity_cost', 0.0),
            'risk_penalty': loss_dict.get('risk_penalty', 0.0),
            'state_regularization': loss_dict.get('state_regularization', 0.0),
            **evaluation_metrics
        }
        
        return result
    
    def _calculate_evaluation_metrics(
        self, 
        position_predictions: torch.Tensor, 
        next_day_returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            position_predictions: [batch_size, 20, 1] 仓位预测
            next_day_returns: [batch_size, 20] 次日收益
            
        Returns:
            评估指标字典
        """
        if position_predictions is None:
            return {
                'mean_cumulative_return': 0.0,
                'return_volatility': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        batch_size = position_predictions.shape[0]
        
        # 计算每个样本的累积收益
        cumulative_returns = []
        daily_returns_all = []
        max_drawdowns = []
        
        with torch.no_grad():
            for b in range(batch_size):
                positions = position_predictions[b, :, 0] / 10.0  # 标准化到[0,1]
                returns = next_day_returns[b, :]
                
                # 计算每日收益
                daily_returns = positions * returns
                daily_returns_all.extend(daily_returns.cpu().numpy())
                
                # 计算累积收益
                portfolio_value = 1.0
                portfolio_values = [portfolio_value]
                
                for daily_return in daily_returns:
                    portfolio_value *= (1.0 + daily_return.item())
                    portfolio_values.append(portfolio_value)
                
                cumulative_return = portfolio_value - 1.0
                cumulative_returns.append(cumulative_return)
                
                # 计算最大回撤
                peak = portfolio_values[0]
                max_drawdown = 0.0
                for value in portfolio_values[1:]:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                max_drawdowns.append(max_drawdown)
        
        # 计算平均指标
        mean_cumulative_return = np.mean(cumulative_returns)
        return_volatility = np.std(daily_returns_all) if daily_returns_all else 0.0
        mean_max_drawdown = np.mean(max_drawdowns)
        
        # 计算夏普比率
        if return_volatility > 0:
            sharpe_ratio = np.mean(daily_returns_all) / return_volatility * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        return {
            'mean_cumulative_return': mean_cumulative_return,
            'return_volatility': return_volatility,
            'max_drawdown': mean_max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def validate_step(self, sliding_window_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证步骤
        
        Args:
            sliding_window_data: 验证数据
            
        Returns:
            验证指标
        """
        self.model.eval()
        
        with torch.no_grad():
            result = self.train_step(sliding_window_data)
        
        # 移除loss_tensor（验证时不需要）
        if 'loss_tensor' in result:
            del result['loss_tensor']
        
        return result


def create_sliding_window_batches(
    features_list: torch.Tensor,      # [n_sequences, 20, 180, 11]
    price_targets_list: torch.Tensor, # [n_sequences, 20, 7]
    next_day_returns: torch.Tensor,   # [n_sequences, 20]
    batch_size: int = 2
) -> List[Dict[str, torch.Tensor]]:
    """
    创建滑动窗口批次数据
    
    Args:
        features_list: 特征数据
        price_targets_list: 价格目标
        next_day_returns: 次日收益
        batch_size: 批次大小
        
    Returns:
        批次数据列表
    """
    n_sequences = features_list.shape[0]
    batches = []
    
    for i in range(0, n_sequences, batch_size):
        end_idx = min(i + batch_size, n_sequences)
        
        batch_data = {
            'features': features_list[i:end_idx],
            'price_targets': price_targets_list[i:end_idx],
            'next_day_returns': next_day_returns[i:end_idx]
        }
        
        batches.append(batch_data)
    
    return batches

# -*- coding: utf-8 -*-
"""
金融时序预测专用损失函数集合

包含多种损失函数组合，专门针对股票价格预测任务优化：
1. 基础回归损失（MSE/MAE/Huber）
2. 方向损失（预测涨跌方向）
3. 趋势损失（价格变化趋势一致性）
4. 时间加权损失（近期预测更重要）
5. 排序损失（相对大小关系）
6. 波动率损失（价格波动模式）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DirectionLoss(nn.Module):
    """方向损失：专注于预测涨跌方向的准确性"""
    
    def __init__(self, weight_by_magnitude: bool = True):
        super().__init__()
        self.weight_by_magnitude = weight_by_magnitude
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算方向损失
        
        Args:
            predictions: [batch_size, seq_len] 预测价格
            targets: [batch_size, seq_len] 真实价格
            
        Returns:
            方向损失值
        """
        if predictions.shape[1] <= 1:
            return torch.tensor(0.0, device=predictions.device)
        
        # 计算相邻时间点的价格变化方向
        pred_changes = predictions[:, 1:] - predictions[:, :-1]  # [batch, seq_len-1]
        true_changes = targets[:, 1:] - targets[:, :-1]          # [batch, seq_len-1]
        
        # 方向向量：上涨=1，下跌=-1，不变=0
        pred_directions = torch.sign(pred_changes)
        true_directions = torch.sign(true_changes)
        
        # 方向不一致的惩罚
        direction_errors = (pred_directions != true_directions).float()
        
        if self.weight_by_magnitude:
            # 加权：变化幅度越大，方向错误的惩罚越大
            change_magnitude = torch.abs(true_changes)
            weighted_errors = direction_errors * (1 + change_magnitude)
            return weighted_errors.mean()
        else:
            return direction_errors.mean()


class TrendLoss(nn.Module):
    """趋势损失：确保整体价格变化趋势的一致性"""
    
    def __init__(self, order: int = 2):
        super().__init__()
        self.order = order  # 差分阶数：1=一阶导数，2=二阶导数（加速度）
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算趋势损失
        
        Args:
            predictions: [batch_size, seq_len] 预测价格
            targets: [batch_size, seq_len] 真实价格
            
        Returns:
            趋势损失值
        """
        if predictions.shape[1] <= self.order:
            return torch.tensor(0.0, device=predictions.device)
        
        if self.order == 1:
            # 一阶差分（速度）
            pred_trend = predictions[:, 1:] - predictions[:, :-1]
            true_trend = targets[:, 1:] - targets[:, :-1]
        elif self.order == 2:
            # 二阶差分（加速度/趋势变化）
            pred_trend = predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2]
            true_trend = targets[:, 2:] - 2 * targets[:, 1:-1] + targets[:, :-2]
        else:
            raise ValueError(f"不支持的差分阶数: {self.order}")
        
        # 趋势差异
        trend_loss = torch.mean((pred_trend - true_trend) ** 2)
        
        return trend_loss


class RankingLoss(nn.Module):
    """排序损失：保持预测值之间的相对大小关系"""
    
    def __init__(self, margin: float = 1.0, threshold: float = 0.01):
        super().__init__()
        self.margin = margin
        self.threshold = threshold
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算排序损失
        
        Args:
            predictions: [batch_size, seq_len] 预测价格
            targets: [batch_size, seq_len] 真实价格
            
        Returns:
            排序损失值
        """
        _, seq_len = predictions.shape
        
        if seq_len <= 1:
            return torch.tensor(0.0, device=predictions.device)
        
        total_loss = 0.0
        count = 0
        
        # 生成所有可能的配对
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # 真实值的大小关系
                true_diff = targets[:, i] - targets[:, j]  # [batch]
                pred_diff = predictions[:, i] - predictions[:, j]  # [batch]
                
                # 如果真实值有明显大小关系（差异>阈值），则约束预测值保持相同关系
                significant_diff = torch.abs(true_diff) > self.threshold
                
                if significant_diff.any():
                    # 使用hinge loss：如果关系错误则惩罚
                    ranking_error = torch.relu(
                        self.margin - true_diff * pred_diff / (torch.abs(true_diff) + 1e-8)
                    )
                    total_loss += ranking_error[significant_diff].mean()
                    count += 1
        
        return total_loss / max(count, 1)


class VolatilityLoss(nn.Module):
    """波动率损失：确保预测的价格波动模式与真实波动一致"""
    
    def __init__(self, window_size: int = 3):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算波动率损失
        
        Args:
            predictions: [batch_size, seq_len] 预测价格
            targets: [batch_size, seq_len] 真实价格
            
        Returns:
            波动率损失值
        """
        if predictions.shape[1] < self.window_size:
            return torch.tensor(0.0, device=predictions.device)
        
        # 计算滑动窗口内的标准差（波动率）
        pred_volatility = []
        true_volatility = []
        
        for i in range(predictions.shape[1] - self.window_size + 1):
            pred_window = predictions[:, i:i+self.window_size]
            true_window = targets[:, i:i+self.window_size]
            
            pred_vol = torch.std(pred_window, dim=1)
            true_vol = torch.std(true_window, dim=1)
            
            pred_volatility.append(pred_vol)
            true_volatility.append(true_vol)
        
        pred_volatility = torch.stack(pred_volatility, dim=1)  # [batch, windows]
        true_volatility = torch.stack(true_volatility, dim=1)  # [batch, windows]
        
        # 波动率差异
        volatility_loss = torch.mean((pred_volatility - true_volatility) ** 2)
        
        return volatility_loss


class TemporalWeightedLoss(nn.Module):
    """时间加权损失：近期预测比远期预测更重要"""
    
    def __init__(self, base_loss_fn: nn.Module, decay_factor: float = 0.9):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.decay_factor = decay_factor
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算时间加权损失
        
        Args:
            predictions: [batch_size, seq_len] 预测价格
            targets: [batch_size, seq_len] 真实价格
            
        Returns:
            时间加权损失值
        """
        seq_len = predictions.shape[1]
        device = predictions.device
        
        # 生成时间权重：近期权重高，远期权重低
        # 对应10个时间点：[1,2,3,4,5,10,15,20,25,30]天
        if seq_len == 10:
            # 专门为股票预测的10个时间点设计权重
            weights = torch.tensor([
                2.0, 1.8, 1.6, 1.4, 1.2,  # 前5天权重较高
                1.0, 0.8, 0.6, 0.4, 0.2   # 后5个时间点权重递减
            ], device=device)
        else:
            # 通用的指数衰减权重
            weights = torch.tensor([
                self.decay_factor ** i for i in range(seq_len)
            ], device=device)
        
        # 计算逐点损失
        if hasattr(self.base_loss_fn, 'reduction') and self.base_loss_fn.reduction == 'none':
            point_losses = self.base_loss_fn(predictions, targets)  # [batch, seq_len]
        else:
            # 如果基础损失函数不支持reduction='none'，手动计算
            point_losses = (predictions - targets) ** 2  # 默认使用MSE
        
        # 应用时间权重
        weighted_losses = point_losses * weights.unsqueeze(0)  # [batch, seq_len]
        
        return weighted_losses.mean()


class FinancialMultiLoss(nn.Module):
    """
    金融时序预测多损失函数组合
    
    整合多种损失函数来全面优化股票价格预测：
    - 基础回归损失：数值准确性
    - 方向损失：涨跌方向准确性
    - 趋势损失：价格变化趋势一致性
    - 时间加权：近期预测更重要
    - 排序损失：相对大小关系
    - 波动率损失：价格波动模式
    """
    
    def __init__(
        self,
        base_loss_type: str = 'mse',
        use_direction_loss: bool = True,
        use_trend_loss: bool = True,
        use_temporal_weighting: bool = True,
        use_ranking_loss: bool = False,
        use_volatility_loss: bool = False,
        # 损失权重
        base_weight: float = 1.0,
        direction_weight: float = 0.3,
        trend_weight: float = 0.2,
        ranking_weight: float = 0.1,
        volatility_weight: float = 0.1
    ):
        super().__init__()
        
        # 保存配置
        self.use_direction_loss = use_direction_loss
        self.use_trend_loss = use_trend_loss
        self.use_temporal_weighting = use_temporal_weighting
        self.use_ranking_loss = use_ranking_loss
        self.use_volatility_loss = use_volatility_loss
        
        # 损失权重
        self.base_weight = base_weight
        self.direction_weight = direction_weight
        self.trend_weight = trend_weight
        self.ranking_weight = ranking_weight
        self.volatility_weight = volatility_weight
        
        # 基础回归损失
        if base_loss_type == 'mse':
            base_loss_fn = nn.MSELoss(reduction='none')
        elif base_loss_type == 'mae':
            base_loss_fn = nn.L1Loss(reduction='none')
        elif base_loss_type == 'huber':
            base_loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"不支持的基础损失类型: {base_loss_type}")
        
        # 创建损失函数组件
        if use_temporal_weighting:
            self.base_loss = TemporalWeightedLoss(base_loss_fn)
        else:
            self.base_loss = base_loss_fn
            
        if use_direction_loss:
            self.direction_loss = DirectionLoss(weight_by_magnitude=True)
            
        if use_trend_loss:
            self.trend_loss = TrendLoss(order=2)
            
        if use_ranking_loss:
            self.ranking_loss = RankingLoss(margin=1.0, threshold=0.01)
            
        if use_volatility_loss:
            self.volatility_loss = VolatilityLoss(window_size=3)
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失函数
        
        Args:
            predictions: [batch_size, seq_len] 预测价格
            targets: [batch_size, seq_len] 真实价格
            
        Returns:
            损失字典，包含总损失和各组件损失
        """
        device = predictions.device
        
        # 1. 基础回归损失
        if self.use_temporal_weighting:
            base_loss = self.base_loss(predictions, targets)
        else:
            base_losses = self.base_loss(predictions, targets)
            base_loss = base_losses.mean()
        
        # 总损失从基础损失开始
        total_loss = self.base_weight * base_loss
        
        # 损失组件字典
        loss_components = {
            'base_loss': base_loss.item(),
            'direction_loss': 0.0,
            'trend_loss': 0.0,
            'ranking_loss': 0.0,
            'volatility_loss': 0.0
        }
        
        # 2. 方向损失
        if self.use_direction_loss:
            direction_loss = self.direction_loss(predictions, targets)
            total_loss += self.direction_weight * direction_loss
            loss_components['direction_loss'] = direction_loss.item()
        
        # 3. 趋势损失
        if self.use_trend_loss:
            trend_loss = self.trend_loss(predictions, targets)
            total_loss += self.trend_weight * trend_loss
            loss_components['trend_loss'] = trend_loss.item()
        
        # 4. 排序损失
        if self.use_ranking_loss:
            ranking_loss = self.ranking_loss(predictions, targets)
            total_loss += self.ranking_weight * ranking_loss
            loss_components['ranking_loss'] = ranking_loss.item()
        
        # 5. 波动率损失
        if self.use_volatility_loss:
            volatility_loss = self.volatility_loss(predictions, targets)
            total_loss += self.volatility_weight * volatility_loss
            loss_components['volatility_loss'] = volatility_loss.item()
        
        # 计算监控指标
        with torch.no_grad():
            mae = torch.mean(torch.abs(predictions - targets))
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
            
            # 方向准确率
            if predictions.shape[1] > 1:
                pred_direction = torch.sign(predictions[:, 1:] - predictions[:, :-1])
                true_direction = torch.sign(targets[:, 1:] - targets[:, :-1])
                direction_accuracy = torch.mean((pred_direction == true_direction).float())
            else:
                direction_accuracy = torch.tensor(0.0)
        
        # 返回完整的损失字典
        return {
            # 主损失（用于反向传播）
            'loss': total_loss,
            
            # 组件损失（用于监控）
            **loss_components,
            
            # 评估指标（用于监控）
            'mae': mae.item(),
            'rmse': rmse.item(),
            'direction_accuracy': direction_accuracy.item(),
        }

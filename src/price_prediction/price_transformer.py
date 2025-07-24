"""
价格预测Transformer网络
专门用于价格预测，与策略网络完全解耦
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import math

# 导入原有的组件
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .attention import MultiHeadLatentAttention
from .feedforward import TransformerBlock
from .utils import RMSNorm, precompute_freqs_cis


class PriceTransformer(nn.Module):
    """
    专门用于价格预测的Transformer网络
    输入：历史金融数据
    输出：未来价格预测 + 特征向量（供策略网络使用）
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.d_model
        self.n_features = 11  # 金融特征数量
        
        # 特征嵌入和归一化
        self.feature_embed = nn.Linear(self.n_features, self.d_model)
        self.feature_norm = nn.LayerNorm(self.n_features)
        
        # RoPE位置编码
        self.freqs_cis = precompute_freqs_cis(
            dim=self.d_model // args.n_heads,
            seq_len=args.max_seq_len,
            theta=args.rope_theta
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(args) for _ in range(args.n_layers)
        ])
        
        # 最终归一化
        self.norm = RMSNorm(self.d_model)
        
        # 价格预测头
        self.price_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_model // 2, args.prediction_horizon)  # 预测未来7天价格
        )
        
        # 特征提取头（供策略网络使用）
        self.feature_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_model, args.d_model)  # 保持维度
        )
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        financial_data: torch.Tensor,
        return_features: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            financial_data: [batch_size, seq_len, n_features] 金融数据
            return_features: 是否返回特征向量（供策略网络使用）
            return_dict: 是否返回字典格式
            
        Returns:
            包含价格预测和特征的字典
        """
        batch_size, seq_len, n_features = financial_data.shape
        device = financial_data.device
        
        # 验证输入特征数量
        assert n_features == self.n_features, f"期望{self.n_features}个特征，实际得到{n_features}个"
        
        # 1. 特征归一化和嵌入
        normalized_data = self.feature_norm(financial_data)
        hidden_states = self.feature_embed(normalized_data)  # [batch, seq_len, d_model]
        
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
        
        # 7. 特征提取（供策略网络使用）
        extracted_features = None
        if return_features:
            extracted_features = self.feature_head(last_hidden)
        
        if return_dict:
            result = {
                "price_predictions": price_predictions,
                "last_hidden": last_hidden
            }
            if extracted_features is not None:
                result["strategy_features"] = extracted_features
            return result
        else:
            outputs = (price_predictions,)
            if extracted_features is not None:
                outputs = outputs + (extracted_features,)
            return outputs
    
    def extract_features(self, financial_data: torch.Tensor) -> torch.Tensor:
        """
        专门用于提取特征的方法（供策略网络调用）
        
        Args:
            financial_data: [batch_size, seq_len, n_features]
            
        Returns:
            strategy_features: [batch_size, d_model] 策略网络所需的特征
        """
        with torch.no_grad():  # 特征提取时不需要梯度
            outputs = self.forward(financial_data, return_features=True, return_dict=True)
            return outputs["strategy_features"]
    
    def predict_prices(self, financial_data: torch.Tensor) -> torch.Tensor:
        """
        专门用于价格预测的方法
        
        Args:
            financial_data: [batch_size, seq_len, n_features]
            
        Returns:
            price_predictions: [batch_size, prediction_horizon]
        """
        outputs = self.forward(financial_data, return_features=False, return_dict=True)
        return outputs["price_predictions"]


class PricePredictionLoss(nn.Module):
    """价格预测专用损失函数"""
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def forward(
        self, 
        price_predictions: torch.Tensor, 
        price_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算价格预测损失
        
        Args:
            price_predictions: [batch_size, prediction_horizon] 预测价格
            price_targets: [batch_size, prediction_horizon] 真实价格
            
        Returns:
            损失字典
        """
        # 基础损失
        base_loss = self.loss_fn(price_predictions, price_targets)
        
        # 计算额外指标
        with torch.no_grad():
            # 平均绝对误差
            mae = torch.mean(torch.abs(price_predictions - price_targets))
            
            # 相对误差
            relative_error = torch.mean(
                torch.abs(price_predictions - price_targets) / (torch.abs(price_targets) + 1e-8)
            )
            
            # 方向准确率（预测涨跌方向的准确率）
            pred_direction = torch.sign(price_predictions[:, 1:] - price_predictions[:, :-1])
            true_direction = torch.sign(price_targets[:, 1:] - price_targets[:, :-1])
            direction_accuracy = torch.mean((pred_direction == true_direction).float())
        
        return {
            'loss': base_loss,
            'mae': mae.item(),
            'relative_error': relative_error.item(),
            'direction_accuracy': direction_accuracy.item()
        }


def create_price_transformer(args) -> PriceTransformer:
    """
    创建价格预测Transformer
    
    Args:
        args: 模型配置
        
    Returns:
        价格预测模型实例
    """
    return PriceTransformer(args)

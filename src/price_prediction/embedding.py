# -*- coding: utf-8 -*-
"""
金融特征嵌入层
用于价格预测Transformer模型的特征嵌入处理
"""

import torch
import torch.nn as nn


class FinancialEmbedding(nn.Module):
    """
    金融特征统一嵌入层

    设计思想：
    1. 每日交易数据作为一个整体进行嵌入
    2. 让模型自由学习20维特征间的关系
    3. 避免人为分组可能带来的限制

    20维特征包括：
    - 时间特征 (3维): 月、日、星期
    - 价格特征 (4维): open_rel, high_rel, low_rel, close_rel
    - 价格变化 (2维): 涨幅, 振幅
    - 成交量特征 (2维): volume_rel, volume_log
    - 金额特征 (2维): amount_rel, amount_log
    - 市场特征 (3维): 成交次数, 换手%, price_median
    - 金融特征 (4维): big_order_activity, chip_concentration, market_sentiment, price_volume_sync

    优势：
    1. 简单直接，符合"每日信息整体"的直觉
    2. 让模型自由学习特征间关系，不受人为分组限制
    3. 参数更少，计算更高效
    4. 无需d_model整除约束
    """
    
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model

        # 统一嵌入：20维特征 → d_model维
        self.feature_embedding = nn.Sequential(
            nn.Linear(20, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 可选的额外变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch, seq_len, 20] 输入特征

        Returns:
            embedded: [batch, seq_len, d_model] 嵌入后的特征
        """
        _, _, feature_dim = x.shape
        assert feature_dim == 20, f"Expected 20 features, got {feature_dim}"

        # 统一嵌入：每日20维特征作为整体进行嵌入
        embedded = self.feature_embedding(x)  # [batch, seq_len, d_model]

        # 可选的额外变换
        embedded = self.feature_transform(embedded)  # [batch, seq_len, d_model]

        return embedded
    
    def get_feature_importance(self, x: torch.Tensor) -> dict:
        """
        获取各特征组的重要性（用于分析）
        
        Args:
            x: [batch, seq_len, 20] 输入特征
            
        Returns:
            importance: 各特征组的重要性分数
        """
        with torch.no_grad():
            # 分组提取特征
            time_features = x[..., 0:3]
            price_features = x[..., 3:7]
            volume_features = x[..., 7:12]
            market_features = x[..., 12:16]
            financial_features = x[..., 16:20]
            
            # 计算各组的L2范数作为重要性指标
            importance = {
                'time': torch.norm(time_features, dim=-1).mean().item(),
                'price': torch.norm(price_features, dim=-1).mean().item(),
                'volume': torch.norm(volume_features, dim=-1).mean().item(),
                'market': torch.norm(market_features, dim=-1).mean().item(),
                'financial': torch.norm(financial_features, dim=-1).mean().item()
            }
            
        return importance


class BatchSequenceNorm(nn.Module):
    """
    Batch序列内标准化
    
    目的：
    1. 消除不同股票、不同时期的数值差异
    2. 在每个batch内，沿序列维度进行标准化
    3. 保持时序内的相对关系
    """
    
    def __init__(self, eps: float = 1e-6, learnable: bool = False):
        super().__init__()
        self.eps = eps
        self.learnable = learnable
        
        if learnable:
            # 可学习的缩放和偏移参数
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, seq_len, d_model] 输入特征
            
        Returns:
            normalized: [batch, seq_len, d_model] 标准化后的特征
        """
        # 沿序列维度计算均值和标准差
        mean = x.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
        std = x.std(dim=1, keepdim=True)    # [batch, 1, d_model]
        
        # 标准化
        normalized = (x - mean) / (std + self.eps)
        
        if self.learnable:
            normalized = normalized * self.weight + self.bias
        
        return normalized





class FinancialEmbeddingLayer(nn.Module):
    """
    完整的金融特征嵌入层
    包含特征嵌入和批标准化（位置编码由RoPE处理）
    """

    def __init__(
        self,
        d_model: int = 512,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        learnable_norm: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.use_batch_norm = use_batch_norm

        # 特征嵌入
        self.feature_embedding = FinancialEmbedding(d_model, dropout)

        # 批标准化
        if use_batch_norm:
            self.batch_norm = BatchSequenceNorm(learnable=learnable_norm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整的嵌入处理流程

        Args:
            x: [batch, seq_len, 20] 原始特征

        Returns:
            embedded: [batch, seq_len, d_model] 最终嵌入特征（位置信息由RoPE处理）
        """
        # 1. 特征嵌入
        embedded = self.feature_embedding(x)

        # 2. 批标准化（可选）
        if self.use_batch_norm:
            embedded = self.batch_norm(embedded)

        return embedded
    
    def get_embedding_stats(self, x: torch.Tensor) -> dict:
        """
        获取嵌入统计信息（用于调试）
        """
        with torch.no_grad():
            embedded = self.feature_embedding(x)
            
            stats = {
                'mean': embedded.mean().item(),
                'std': embedded.std().item(),
                'min': embedded.min().item(),
                'max': embedded.max().item(),
                'feature_importance': self.feature_embedding.get_feature_importance(x)
            }
            
        return stats


# 工厂函数
def create_financial_embedding(
    d_model: int = 512,
    dropout: float = 0.1,
    **kwargs
) -> FinancialEmbeddingLayer:
    """
    创建金融特征嵌入层的工厂函数

    Args:
        d_model: 模型维度
        dropout: dropout率
        **kwargs: 其他参数（如use_batch_norm等）

    Returns:
        embedding_layer: 金融特征嵌入层（位置信息由RoPE处理）
    """
    return FinancialEmbeddingLayer(
        d_model=d_model,
        dropout=dropout,
        **kwargs
    )

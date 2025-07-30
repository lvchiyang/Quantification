# -*- coding: utf-8 -*-
"""
金融特征嵌入层
用于价格预测Transformer模型的特征嵌入处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class FinancialEmbedding(nn.Module):
    """
    金融特征分组嵌入层
    
    设计思想：
    1. 不同类型特征分组处理，保持语义独立性
    2. 每组内部学习特征交互
    3. 最后拼接为统一向量
    
    特征分组（20维特征索引映射）：
    - 时间特征 (3维): [0-2] 月、日、星期
    - 价格特征 (4维): [3-6] open_rel, high_rel, low_rel, close_rel
    - 成交量特征 (5维): [7-11] volume_rel, volume_change, amount_rel, amount_change, 成交次数(标准化)
    - 市场特征 (4维): [12-15] 涨幅, 振幅, 换手%, price_median
    - 金融特征 (4维): [16-19] big_order_activity, chip_concentration, market_sentiment, price_volume_sync

    注意：成交次数已从市场特征组移至成交量特征组，因为：
    1. 成交次数与成交量密切相关，都反映市场交易活跃度
    2. 成交次数需要标准化处理，与volume特征处理方式一致
    3. 成交次数/成交量 = 平均每笔交易量，是重要的volume派生指标
    """
    
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 确保d_model能被5整除（5个特征组）
        assert d_model % 5 == 0, f"d_model ({d_model}) must be divisible by 5"
        
        group_dim = d_model // 5
        
        # 第1组：时间特征 (3维 → group_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(3, group_dim),
            nn.LayerNorm(group_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 第2组：价格特征 (4维 → group_dim)
        self.price_embed = nn.Sequential(
            nn.Linear(4, group_dim),
            nn.LayerNorm(group_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 第3组：成交量特征 (5维 → group_dim) - 包含成交次数
        self.volume_embed = nn.Sequential(
            nn.Linear(5, group_dim),
            nn.LayerNorm(group_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 第4组：市场特征 (4维 → group_dim)
        self.market_embed = nn.Sequential(
            nn.Linear(4, group_dim),
            nn.LayerNorm(group_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 第5组：金融特征 (4维 → group_dim)
        self.financial_embed = nn.Sequential(
            nn.Linear(4, group_dim),
            nn.LayerNorm(group_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 最终投影层（可选）
        self.final_projection = nn.Linear(d_model, d_model)
        
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
        
        # 分组提取特征
        time_features = x[..., 0:3]       # 月、日、星期
        price_features = x[..., 3:7]      # open_rel, high_rel, low_rel, close_rel
        volume_features = x[..., 7:12]    # volume_rel, volume_change, amount_rel, amount_change, 成交次数
        market_features = x[..., 12:16]   # 涨幅, 振幅, 换手%, price_median
        financial_features = x[..., 16:20]  # 4个金融特征
        
        # 分组嵌入
        time_emb = self.time_embed(time_features)
        price_emb = self.price_embed(price_features)
        volume_emb = self.volume_embed(volume_features)
        market_emb = self.market_embed(market_features)
        financial_emb = self.financial_embed(financial_features)
        
        # 拼接为统一向量 [batch, seq_len, d_model]
        unified_embedding = torch.cat([
            time_emb, price_emb, volume_emb, market_emb, financial_emb
        ], dim=-1)
        
        # 最终投影
        embedded = self.final_projection(unified_embedding)
        
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


class PositionalEncoding(nn.Module):
    """
    位置编码
    为序列添加位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + positional encoding: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :, :].transpose(0, 1)
        return self.dropout(x)


class FinancialEmbeddingLayer(nn.Module):
    """
    完整的金融特征嵌入层
    包含特征嵌入、批标准化和位置编码
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        max_seq_len: int = 200,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_pos_encoding: bool = True,
        learnable_norm: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_batch_norm = use_batch_norm
        self.use_pos_encoding = use_pos_encoding
        
        # 特征嵌入
        self.feature_embedding = FinancialEmbedding(d_model, dropout)
        
        # 批标准化
        if use_batch_norm:
            self.batch_norm = BatchSequenceNorm(learnable=learnable_norm)
        
        # 位置编码
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整的嵌入处理流程
        
        Args:
            x: [batch, seq_len, 20] 原始特征
            
        Returns:
            embedded: [batch, seq_len, d_model] 最终嵌入特征
        """
        # 1. 特征嵌入
        embedded = self.feature_embedding(x)
        
        # 2. 批标准化（可选）
        if self.use_batch_norm:
            embedded = self.batch_norm(embedded)
        
        # 3. 位置编码（可选）
        if self.use_pos_encoding:
            embedded = self.pos_encoding(embedded)
        
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
    max_seq_len: int = 180,
    dropout: float = 0.1,
    **kwargs
) -> FinancialEmbeddingLayer:
    """
    创建金融特征嵌入层的工厂函数
    
    Args:
        d_model: 模型维度
        max_seq_len: 最大序列长度
        dropout: dropout率
        **kwargs: 其他参数
        
    Returns:
        embedding_layer: 金融特征嵌入层
    """
    return FinancialEmbeddingLayer(
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=dropout,
        **kwargs
    )

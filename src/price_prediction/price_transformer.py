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
from .feedforward import get_ffn
from .utils import RMSNorm, precompute_freqs_cis
from .embedding import FinancialEmbeddingLayer


class TransformerBlock(nn.Module):
    """
    Transformer Block with Pre-RMSNorm

    专门为金融时序预测设计的 Transformer 层，结合了：
    - Multi-Head Latent Attention (MLA)
    - SwiGLU 前馈网络
    - RMSNorm 归一化
    - 残差连接

    结构：
    x -> RMSNorm -> MLA -> Add -> RMSNorm -> FFN -> Add
    """

    def __init__(self, args, layer_idx: int = 0):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        # 注意力层
        self.attn_norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)
        self.attn = MultiHeadLatentAttention(args)

        # 前馈网络层
        self.ffn_norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)
        self.ffn = get_ffn(args, ffn_type="swiglu")  # 使用 SwiGLU

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False  # 金融预测通常不需要因果掩码
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            freqs_cis: RoPE 频率复数 [seq_len, head_dim//2]
            attn_mask: 注意力掩码 [batch_size, seq_len, seq_len]
            is_causal: 是否使用因果掩码（金融预测通常设为False）

        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # Pre-RMSNorm + MLA + 残差连接
        attn_input = self.attn_norm(x)
        attn_output = self.attn(attn_input, freqs_cis, attn_mask, is_causal)
        x = x + attn_output

        # Pre-RMSNorm + FFN + 残差连接
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output

        return x


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
        self.n_features = 20  # 金融特征数量（根据序列处理器）
        
        # 金融特征嵌入层（使用RoPE，关闭传统位置编码）
        self.feature_embedding = FinancialEmbeddingLayer(
            d_model=self.d_model,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            use_batch_norm=True,
            use_pos_encoding=False  # 关闭传统位置编码，使用RoPE
        )
        
        # RoPE位置编码（用于注意力层）
        self.freqs_cis = precompute_freqs_cis(
            dim=self.d_model // args.n_heads,
            end=args.max_seq_len,  # 修复参数名：seq_len -> end
            theta=args.rope_theta
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(args) for _ in range(args.n_layers)
        ])
        
        # 最终归一化
        self.norm = RMSNorm(self.d_model)
        
        # 价格预测头 - 预测未来10个时间点
        self.price_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_model // 2, 10)  # 预测未来10个时间点：第1,2,3,4,5,10,15,20,25,30天
        )
        
        # 特征提取头（供策略网络使用）
        self.feature_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_model, self.d_model)  # 保持维度
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
            financial_data: [batch_size, seq_len, 20] 金融数据（20维特征）
            return_features: 是否返回特征向量（供策略网络使用）
            return_dict: 是否返回字典格式

        Returns:
            包含价格预测和特征的字典
            - price_predictions: [batch_size, 10] 未来10个时间点的价格预测
        """
        _, seq_len, n_features = financial_data.shape
        device = financial_data.device

        # 验证输入特征数量
        assert n_features == self.n_features, f"期望{self.n_features}个特征，实际得到{n_features}个"

        # 1. 金融特征嵌入（包含分组嵌入、批标准化、位置编码）
        hidden_states = self.feature_embedding(financial_data)  # [batch, seq_len, d_model]
        
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
            price_predictions: [batch_size, 10] 未来10个时间点的价格预测
        """
        outputs = self.forward(financial_data, return_features=False, return_dict=True)
        return outputs["price_predictions"]


class PricePredictionLoss(nn.Module):
    """
    股票价格预测专用多损失函数组合

    组合多种损失函数来提高预测效果：
    1. 基础回归损失（MSE/MAE/Huber）
    2. 方向损失（预测涨跌方向）
    3. 趋势损失（价格变化趋势一致性）
    4. 时间加权损失（近期预测更重要）
    5. 排序损失（相对大小关系）
    """

    def __init__(
        self,
        base_loss_type: str = 'mse',
        use_direction_loss: bool = True,
        use_trend_loss: bool = True,
        use_temporal_weighting: bool = True,
        use_ranking_loss: bool = False,
        # 损失权重
        base_weight: float = 1.0,
        direction_weight: float = 0.3,
        trend_weight: float = 0.2,
        ranking_weight: float = 0.1
    ):
        super().__init__()
        self.base_loss_type = base_loss_type
        self.use_direction_loss = use_direction_loss
        self.use_trend_loss = use_trend_loss
        self.use_temporal_weighting = use_temporal_weighting
        self.use_ranking_loss = use_ranking_loss

        # 损失权重
        self.base_weight = base_weight
        self.direction_weight = direction_weight
        self.trend_weight = trend_weight
        self.ranking_weight = ranking_weight

        # 基础回归损失
        if base_loss_type == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        elif base_loss_type == 'mae':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        elif base_loss_type == 'huber':
            self.base_loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"不支持的基础损失类型: {base_loss_type}")

        # 时间权重：近期预测更重要
        # 对应10个时间点：[1,2,3,4,5,10,15,20,25,30]天
        if use_temporal_weighting:
            # 权重递减：近期权重高，远期权重低
            self.register_buffer('temporal_weights', torch.tensor([
                2.0, 1.8, 1.6, 1.4, 1.2,  # 前5天权重较高
                1.0, 0.8, 0.6, 0.4, 0.2   # 后5个时间点权重递减
            ]))
        else:
            self.register_buffer('temporal_weights', torch.ones(10))
    
    def compute_direction_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算方向损失：预测涨跌方向的准确性"""
        if predictions.shape[1] <= 1:
            return torch.tensor(0.0, device=predictions.device)

        # 计算相邻时间点的价格变化方向
        pred_changes = predictions[:, 1:] - predictions[:, :-1]  # [batch, 9]
        true_changes = targets[:, 1:] - targets[:, :-1]          # [batch, 9]

        # 方向向量：上涨=1，下跌=-1，不变=0
        pred_directions = torch.sign(pred_changes)
        true_directions = torch.sign(true_changes)

        # 方向不一致的惩罚
        direction_errors = (pred_directions != true_directions).float()

        # 加权：变化幅度越大，方向错误的惩罚越大
        change_magnitude = torch.abs(true_changes)
        weighted_errors = direction_errors * (1 + change_magnitude)

        return weighted_errors.mean()

    def compute_trend_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算趋势损失：整体价格变化趋势的一致性"""
        if predictions.shape[1] <= 2:
            return torch.tensor(0.0, device=predictions.device)

        # 计算二阶差分（加速度/趋势变化）
        pred_trend = predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2]
        true_trend = targets[:, 2:] - 2 * targets[:, 1:-1] + targets[:, :-2]

        # 趋势差异
        trend_loss = torch.mean((pred_trend - true_trend) ** 2)

        return trend_loss

    def compute_ranking_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算排序损失：保持相对大小关系"""
        _, seq_len = predictions.shape

        # 生成所有可能的配对
        total_loss = 0.0
        count = 0

        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # 真实值的大小关系
                true_diff = targets[:, i] - targets[:, j]  # [batch]
                pred_diff = predictions[:, i] - predictions[:, j]  # [batch]

                # 如果真实值有明显大小关系（差异>阈值），则约束预测值保持相同关系
                threshold = 0.01  # 相对阈值
                significant_diff = torch.abs(true_diff) > threshold

                if significant_diff.any():
                    # 使用hinge loss：如果关系错误则惩罚
                    ranking_error = torch.relu(1 - true_diff * pred_diff / (torch.abs(true_diff) + 1e-8))
                    total_loss += ranking_error[significant_diff].mean()
                    count += 1

        return total_loss / max(count, 1)

    def forward(
        self,
        price_predictions: torch.Tensor,
        price_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失函数

        Args:
            price_predictions: [batch_size, 10] 预测价格（10个时间点）
            price_targets: [batch_size, 10] 真实价格（10个时间点）

        Returns:
            损失字典，包含总损失和各组件损失
        """
        device = price_predictions.device

        # 1. 基础回归损失（带时间权重）
        base_losses = self.base_loss_fn(price_predictions, price_targets)  # [batch, 10]

        if self.use_temporal_weighting:
            # 应用时间权重
            weighted_base_losses = base_losses * self.temporal_weights.to(device)
            base_loss = weighted_base_losses.mean()
        else:
            base_loss = base_losses.mean()

        # 总损失从基础损失开始
        total_loss = self.base_weight * base_loss

        # 2. 方向损失
        direction_loss = torch.tensor(0.0, device=device)
        if self.use_direction_loss:
            direction_loss = self.compute_direction_loss(price_predictions, price_targets)
            total_loss += self.direction_weight * direction_loss

        # 3. 趋势损失
        trend_loss = torch.tensor(0.0, device=device)
        if self.use_trend_loss:
            trend_loss = self.compute_trend_loss(price_predictions, price_targets)
            total_loss += self.trend_weight * trend_loss

        # 4. 排序损失
        ranking_loss = torch.tensor(0.0, device=device)
        if self.use_ranking_loss:
            ranking_loss = self.compute_ranking_loss(price_predictions, price_targets)
            total_loss += self.ranking_weight * ranking_loss
        
        # 计算监控指标
        with torch.no_grad():
            # 平均绝对误差
            mae = torch.mean(torch.abs(price_predictions - price_targets))

            # 均方根误差
            rmse = torch.sqrt(torch.mean((price_predictions - price_targets) ** 2))

            # 相对误差（百分比）
            relative_error = torch.mean(
                torch.abs(price_predictions - price_targets) / (torch.abs(price_targets) + 1e-8)
            ) * 100

            # 方向准确率（预测涨跌方向的准确率）
            if price_predictions.shape[1] > 1:
                pred_direction = torch.sign(price_predictions[:, 1:] - price_predictions[:, :-1])
                true_direction = torch.sign(price_targets[:, 1:] - price_targets[:, :-1])
                direction_accuracy = torch.mean((pred_direction == true_direction).float())
            else:
                direction_accuracy = torch.tensor(0.0)

            # 最大绝对误差
            max_error = torch.max(torch.abs(price_predictions - price_targets))

            # R²决定系数
            ss_res = torch.sum((price_targets - price_predictions) ** 2)
            ss_tot = torch.sum((price_targets - torch.mean(price_targets)) ** 2)
            r2_score = 1 - ss_res / (ss_tot + 1e-8)

        return {
            # 主损失（用于反向传播）
            'loss': total_loss,

            # 组件损失（用于监控）
            'base_loss': base_loss.item(),
            'direction_loss': direction_loss.item(),
            'trend_loss': trend_loss.item(),
            'ranking_loss': ranking_loss.item(),

            # 评估指标（用于监控）
            'mae': mae.item(),
            'rmse': rmse.item(),
            'relative_error': relative_error.item(),
            'direction_accuracy': direction_accuracy.item(),
            'max_error': max_error.item(),
            'r2_score': r2_score.item()
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

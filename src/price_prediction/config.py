"""
价格预测网络配置
专门为Transformer架构设计的配置参数
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class PricePredictionConfig:
    """
    价格预测网络配置
    
    专门为Transformer架构优化：
    - 长序列处理能力
    - 价格预测专用参数
    - MLA注意力机制配置
    """
    
    # 模型架构参数
    d_model: int = 512            # 模型维度
    n_layers: int = 8             # Transformer层数
    n_heads: int = 8              # 注意力头数
    
    # MLA 相关参数
    kv_lora_rank: int = 256       # K/V 压缩维度
    v_head_dim: int = 64          # 值头维度
    
    # 前馈网络参数
    intermediate_size: int = 2048  # FFN中间层维度
    dropout: float = 0.1          # Dropout概率
    layer_norm_eps: float = 1e-6  # LayerNorm epsilon
    
    # 数据参数
    n_features: int = 13          # 输入特征数
    sequence_length: int = 180    # 输入序列长度
    prediction_horizon: int = 7   # 预测时间跨度
    max_seq_len: int = 512        # 最大序列长度（用于RoPE）
    
    # 训练参数
    batch_size: int = 4           # 批次大小
    learning_rate: float = 1e-4   # 学习率
    weight_decay: float = 0.01    # 权重衰减
    max_epochs: int = 100         # 最大训练轮数
    warmup_steps: int = 1000      # 预热步数
    
    # 损失函数参数
    loss_type: str = "mse"        # 损失函数类型 ("mse", "mae", "huber")
    huber_delta: float = 1.0      # Huber损失的delta参数
    
    # 数据处理参数
    data_dir: str = "processed_data_2025-07-29"
    large_value_transform: str = "relative_change"
    normalize_features: bool = True
    
    # 模型保存参数
    save_dir: str = "checkpoints/price_prediction"
    save_every_n_epochs: int = 10
    early_stopping_patience: int = 20
    
    def __post_init__(self):
        """验证配置参数"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.sequence_length <= self.max_seq_len, "sequence_length must <= max_seq_len"
        
        # 计算查询/键头维度
        self.qk_head_dim = self.d_model // self.n_heads
        assert self.qk_head_dim % 2 == 0, "qk_head_dim must be even for RoPE"
        
        print(f"价格预测模型配置:")
        print(f"  模型维度: {self.d_model}")
        print(f"  层数: {self.n_layers}")
        print(f"  注意力头数: {self.n_heads}")
        print(f"  序列长度: {self.sequence_length}")
        print(f"  预测时间跨度: {self.prediction_horizon}")
        print(f"  特征数: {self.n_features}")

class PricePredictionConfigs:
    """预定义的价格预测配置"""
    
    @staticmethod
    def tiny():
        """轻量级配置，用于快速测试"""
        return PricePredictionConfig(
            d_model=256,
            n_layers=4,
            n_heads=4,
            kv_lora_rank=128,
            v_head_dim=64,
            intermediate_size=512,
            batch_size=2,
            max_epochs=50
        )
    
    @staticmethod
    def small():
        """小型配置，适合个人电脑"""
        return PricePredictionConfig(
            d_model=512,
            n_layers=6,
            n_heads=8,
            kv_lora_rank=256,
            v_head_dim=64,
            intermediate_size=1024,
            batch_size=4,
            max_epochs=100
        )
    
    @staticmethod
    def base():
        """基础配置（默认）"""
        return PricePredictionConfig()
    
    @staticmethod
    def large():
        """大型配置，适合服务器训练"""
        return PricePredictionConfig(
            d_model=1024,
            n_layers=12,
            n_heads=16,
            kv_lora_rank=512,
            v_head_dim=64,
            intermediate_size=4096,
            batch_size=8,
            max_epochs=200,
            learning_rate=5e-5
        )
    
    @staticmethod
    def for_long_sequence():
        """长序列专用配置"""
        return PricePredictionConfig(
            d_model=768,
            n_layers=8,
            n_heads=12,
            sequence_length=360,  # 更长的序列
            max_seq_len=1024,
            kv_lora_rank=384,
            intermediate_size=3072,
            batch_size=2,  # 减小批次以适应长序列
            learning_rate=8e-5
        )
    
    @staticmethod
    def for_multi_step_prediction():
        """多步预测专用配置"""
        return PricePredictionConfig(
            d_model=512,
            n_layers=10,
            n_heads=8,
            prediction_horizon=14,  # 预测未来14天
            kv_lora_rank=256,
            intermediate_size=2048,
            loss_type="mae",  # 多步预测使用MAE
            learning_rate=1e-4
        )

"""
模型配置文件
定义 Decoder-only Transformer 模型的各种超参数
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    """模型配置参数"""

    # 模型架构参数
    d_model: int = 1024           # 隐藏维度 h
    n_layers: int = 24            # Transformer 层数
    n_heads: int = 16             # 查询头数

    # MLA 相关参数
    kv_lora_rank: int = 512       # MLA 中 K/V 压缩维度
    qk_rope_head_dim: int = 64    # RoPE 维度
    qk_nope_head_dim: int = 128   # 非 RoPE 查询/键维度
    v_head_dim: int = 128         # 值头维度

    # FFN 参数
    intermediate_size: int = 2816 # SwiGLU 中间维度 ≈ 2/3 * 4h
    
    # 金融数据参数
    n_features: int = 11          # 输入特征数（开盘、最高、最低、收盘等）
    n_predictions: int = 7        # 价格预测时间点数量
    n_trading_days: int = 20      # 交易策略预测天数
    max_seq_len: int = 2048       # 最大序列长度

    # 交易策略参数
    position_range_min: int = 0   # 仓位最小值（空仓）
    position_range_max: int = 10  # 仓位最大值（满仓）
    enable_trading_strategy: bool = True  # 是否启用交易策略学习
    sliding_window_mode: bool = True      # 是否使用滑动窗口模式
    position_method: str = 'gumbel_softmax'  # 仓位离散化方法: 'gumbel_softmax', 'straight_through', 'concrete'

    # 递归状态参数
    enable_stateful_training: bool = True     # 是否启用状态化训练
    strategy_state_dim: int = 256             # 策略状态维度
    state_update_method: str = 'gru'          # 状态更新方法: 'gru', 'lstm', 'attention'
    state_dropout: float = 0.1                # 状态更新的dropout

    # 市场分类参数
    market_classification_method: str = 'comprehensive'  # 市场分类方法
    bull_threshold: float = 0.008             # 牛市阈值
    bear_threshold: float = -0.008            # 熊市阈值
    market_window_size: int = 5               # 市场状态判断窗口大小
    
    # 正则化
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    # RoPE 参数
    rope_theta: float = 1e4

    # 训练参数
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # 多任务损失函数权重
    price_loss_weight: float = 1.0      # 价格预测损失权重
    trading_loss_weight: float = 0.1    # 交易策略损失权重

    # 信息比率损失参数
    information_ratio_weight: float = 1.0    # 信息比率权重
    opportunity_cost_weight: float = 0.1     # 机会成本权重
    risk_adjustment_weight: float = 0.05     # 风险调整权重
    state_regularization_weight: float = 0.001  # 状态正则化权重

    # 其他
    tie_word_embeddings: bool = False  # 是否共享词嵌入权重

    def __post_init__(self):
        """验证配置参数的合理性"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.qk_rope_head_dim % 2 == 0, "qk_rope_head_dim must be even for RoPE"
        assert self.kv_lora_rank > 0, "kv_lora_rank must be positive"
        assert self.intermediate_size > 0, "intermediate_size must be positive"
        
        # 计算总的查询/键维度
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        # 验证维度匹配
        total_qk_dim = self.n_heads * self.qk_head_dim
        total_v_dim = self.n_heads * self.v_head_dim
        
        print(f"Model configuration:")
        print(f"  d_model: {self.d_model}")
        print(f"  n_layers: {self.n_layers}")
        print(f"  n_heads: {self.n_heads}")
        print(f"  qk_head_dim: {self.qk_head_dim} (rope: {self.qk_rope_head_dim}, nope: {self.qk_nope_head_dim})")
        print(f"  v_head_dim: {self.v_head_dim}")
        print(f"  kv_lora_rank: {self.kv_lora_rank}")
        print(f"  intermediate_size: {self.intermediate_size}")
        print(f"  n_features: {self.n_features}")
        print(f"  n_predictions: {self.n_predictions}")
        print(f"  max_seq_len: {self.max_seq_len}")


# 预定义的模型配置
class ModelConfigs:
    """预定义的模型配置"""

    @staticmethod
    def tiny():
        """微型模型配置，用于测试"""
        return ModelArgs(
            d_model=256,
            n_layers=4,
            n_heads=4,
            kv_lora_rank=128,
            qk_rope_head_dim=32,
            qk_nope_head_dim=32,
            v_head_dim=64,
            intermediate_size=512,
            n_features=11,
            n_predictions=7,
            n_trading_days=20,
            max_seq_len=512
        )
    
    @staticmethod
    def small():
        """小型模型配置"""
        return ModelArgs(
            d_model=512,
            n_layers=8,
            n_heads=8,
            kv_lora_rank=256,
            qk_rope_head_dim=32,
            qk_nope_head_dim=32,
            v_head_dim=64,
            intermediate_size=1024,
            max_seq_len=1024
        )

    @staticmethod
    def base():
        """基础模型配置（默认）"""
        return ModelArgs()

    @staticmethod
    def large():
        """大型模型配置"""
        return ModelArgs(
            d_model=2048,
            n_layers=32,
            n_heads=32,
            kv_lora_rank=1024,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
            intermediate_size=5632,
            max_seq_len=4096
        )

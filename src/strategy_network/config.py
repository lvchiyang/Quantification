"""
策略网络配置
专门为GRU架构设计的配置参数
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class StrategyNetworkConfig:
    """
    策略网络配置
    
    专门为GRU架构优化：
    - 短序列递归处理
    - 交易策略专用参数
    - 收益优化配置
    """
    
    # 模型架构参数
    input_dim: int = 14           # 输入特征维度
    hidden_dim: int = 128         # GRU隐藏层维度
    num_layers: int = 2           # GRU层数
    dropout: float = 0.1          # Dropout概率
    
    # 策略参数
    trading_horizon: int = 20     # 交易决策时间跨度
    position_range_min: int = 0   # 最小仓位（空仓）
    position_range_max: int = 10  # 最大仓位（满仓）
    position_method: str = "gumbel_softmax"  # 仓位离散化方法
    
    # 状态化训练参数
    enable_stateful_training: bool = True
    strategy_state_dim: int = 128
    state_update_method: str = "gru"  # 'gru', 'lstm', 'attention'
    
    # 损失函数权重
    information_ratio_weight: float = 1.0      # 信息比率权重
    opportunity_cost_weight: float = 0.1       # 机会成本权重
    risk_adjustment_weight: float = 0.05       # 风险调整权重
    transaction_cost_weight: float = 0.02      # 交易成本权重
    
    # 市场分类参数
    enable_market_classification: bool = True
    market_classification_method: str = "adaptive"  # 'simple', 'technical', 'adaptive'
    
    # 训练参数
    batch_size: int = 8           # 批次大小
    learning_rate: float = 1e-3   # 学习率
    weight_decay: float = 0.01    # 权重衰减
    max_epochs: int = 200         # 最大训练轮数
    warmup_steps: int = 500       # 预热步数
    
    # Gumbel-Softmax参数
    gumbel_temperature: float = 1.0    # Gumbel-Softmax温度
    gumbel_hard: bool = False          # 是否使用硬采样
    temperature_decay: float = 0.99    # 温度衰减率
    min_temperature: float = 0.1       # 最小温度
    
    # 数据处理参数
    data_dir: str = "processed_data_2025-07-29"
    feature_extraction_length: int = 180
    large_value_transform: str = "relative_change"
    
    # 模型保存参数
    save_dir: str = "checkpoints/strategy_network"
    save_every_n_epochs: int = 20
    early_stopping_patience: int = 30
    
    # 风险控制参数
    max_position_change: float = 0.3   # 单日最大仓位变化
    risk_free_rate: float = 0.03       # 无风险利率
    max_drawdown_threshold: float = 0.2 # 最大回撤阈值
    
    def __post_init__(self):
        """验证配置参数"""
        assert self.position_range_min >= 0, "position_range_min must >= 0"
        assert self.position_range_max > self.position_range_min, "position_range_max must > position_range_min"
        assert self.trading_horizon > 0, "trading_horizon must > 0"
        assert 0 < self.gumbel_temperature <= 10, "gumbel_temperature must in (0, 10]"
        
        # 计算仓位档位数
        self.n_position_levels = self.position_range_max - self.position_range_min + 1
        
        print(f"策略网络配置:")
        print(f"  输入维度: {self.input_dim}")
        print(f"  隐藏维度: {self.hidden_dim}")
        print(f"  GRU层数: {self.num_layers}")
        print(f"  交易时间跨度: {self.trading_horizon}")
        print(f"  仓位档位: {self.n_position_levels} ({self.position_range_min}-{self.position_range_max})")
        print(f"  状态化训练: {self.enable_stateful_training}")

class StrategyNetworkConfigs:
    """预定义的策略网络配置"""
    
    @staticmethod
    def tiny():
        """轻量级配置，用于快速测试"""
        return StrategyNetworkConfig(
            input_dim=10,
            hidden_dim=64,
            num_layers=1,
            trading_horizon=10,
            batch_size=4,
            max_epochs=50,
            strategy_state_dim=64
        )
    
    @staticmethod
    def small():
        """小型配置，适合个人电脑"""
        return StrategyNetworkConfig(
            input_dim=12,
            hidden_dim=96,
            num_layers=2,
            trading_horizon=15,
            batch_size=6,
            max_epochs=100,
            strategy_state_dim=96
        )
    
    @staticmethod
    def base():
        """基础配置（默认）"""
        return StrategyNetworkConfig()
    
    @staticmethod
    def large():
        """大型配置，适合服务器训练"""
        return StrategyNetworkConfig(
            input_dim=16,
            hidden_dim=256,
            num_layers=3,
            trading_horizon=30,
            batch_size=16,
            max_epochs=300,
            strategy_state_dim=256,
            learning_rate=5e-4
        )
    
    @staticmethod
    def conservative():
        """保守策略配置"""
        return StrategyNetworkConfig(
            position_range_max=5,  # 最大50%仓位
            risk_adjustment_weight=0.2,  # 更高的风险权重
            transaction_cost_weight=0.05,  # 更高的交易成本
            max_position_change=0.2,  # 更小的仓位变化
            max_drawdown_threshold=0.1,  # 更严格的回撤控制
            information_ratio_weight=0.8
        )
    
    @staticmethod
    def aggressive():
        """激进策略配置"""
        return StrategyNetworkConfig(
            position_range_max=10,  # 满仓
            risk_adjustment_weight=0.02,  # 较低的风险权重
            transaction_cost_weight=0.01,  # 较低的交易成本
            max_position_change=0.5,  # 较大的仓位变化
            max_drawdown_threshold=0.3,  # 较宽松的回撤控制
            information_ratio_weight=1.2,
            learning_rate=2e-3
        )
    
    @staticmethod
    def high_frequency():
        """高频交易配置"""
        return StrategyNetworkConfig(
            trading_horizon=5,  # 短期决策
            hidden_dim=64,
            num_layers=1,
            batch_size=32,
            learning_rate=5e-3,
            transaction_cost_weight=0.1,  # 高频交易成本更重要
            gumbel_temperature=0.5,  # 更确定的决策
            max_position_change=0.1  # 频繁但小幅调整
        )
    
    @staticmethod
    def long_term():
        """长期投资配置"""
        return StrategyNetworkConfig(
            trading_horizon=60,  # 长期决策
            feature_extraction_length=360,  # 更长的历史
            hidden_dim=192,
            num_layers=3,
            batch_size=4,
            learning_rate=1e-4,
            opportunity_cost_weight=0.2,  # 机会成本更重要
            risk_adjustment_weight=0.1,
            max_position_change=0.2
        )

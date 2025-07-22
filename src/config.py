"""
模型配置类
定义 Decoder-only Transformer 模型的所有超参数
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
    
    # 词汇表和序列长度
    vocab_size: int = 32000
    max_seq_len: int = 2048
    
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
    
    # 其他
    tie_word_embeddings: bool = False  # 是否共享输入输出嵌入
    
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
        print(f"  vocab_size: {self.vocab_size}")
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
            vocab_size=32000,
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
            vocab_size=32000,
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
            vocab_size=32000,
            max_seq_len=4096
        )

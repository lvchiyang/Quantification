"""
ģ��������
���� Decoder-only Transformer ģ�͵����г�����
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    """ģ�����ò���"""
    
    # ģ�ͼܹ�����
    d_model: int = 1024           # ����ά�� h
    n_layers: int = 24            # Transformer ����
    n_heads: int = 16             # ��ѯͷ��
    
    # MLA ��ز���
    kv_lora_rank: int = 512       # MLA �� K/V ѹ��ά��
    qk_rope_head_dim: int = 64    # RoPE ά��
    qk_nope_head_dim: int = 128   # �� RoPE ��ѯ/��ά��
    v_head_dim: int = 128         # ֵͷά��
    
    # FFN ����
    intermediate_size: int = 2816 # SwiGLU �м�ά�� �� 2/3 * 4h
    
    # �ʻ������г���
    vocab_size: int = 32000
    max_seq_len: int = 2048
    
    # ����
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    
    # RoPE ����
    rope_theta: float = 1e4
    
    # ѵ������
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    # ����
    tie_word_embeddings: bool = False  # �Ƿ����������Ƕ��
    
    def __post_init__(self):
        """��֤���ò����ĺ�����"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.qk_rope_head_dim % 2 == 0, "qk_rope_head_dim must be even for RoPE"
        assert self.kv_lora_rank > 0, "kv_lora_rank must be positive"
        assert self.intermediate_size > 0, "intermediate_size must be positive"
        
        # �����ܵĲ�ѯ/��ά��
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        
        # ��֤ά��ƥ��
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


# Ԥ�����ģ������
class ModelConfigs:
    """Ԥ�����ģ������"""
    
    @staticmethod
    def tiny():
        """΢��ģ�����ã����ڲ���"""
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
        """С��ģ������"""
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
        """����ģ�����ã�Ĭ�ϣ�"""
        return ModelArgs()
    
    @staticmethod
    def large():
        """����ģ������"""
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

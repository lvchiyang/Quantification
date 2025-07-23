#!/usr/bin/env python3
"""
测试 reshape_for_broadcast 函数的修复
"""

import torch
import sys
import os

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import reshape_for_broadcast, precompute_freqs_cis

def test_reshape_for_broadcast():
    """测试 reshape_for_broadcast 函数"""
    print("🧪 测试 reshape_for_broadcast 函数修复...")
    
    # 测试参数
    batch_size = 2
    seq_len = 10
    n_heads = 8
    head_dim = 64
    
    # 创建测试数据
    # x 的形状: [batch_size, seq_len, n_heads, head_dim]
    x = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # freqs_cis 的形状: [seq_len, head_dim//2]
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    
    print(f"📊 输入形状:")
    print(f"  x.shape: {x.shape}")
    print(f"  freqs_cis.shape: {freqs_cis.shape}")
    
    try:
        # 测试修复后的函数
        reshaped_freqs = reshape_for_broadcast(freqs_cis, x)
        print(f"✅ 广播后形状: {reshaped_freqs.shape}")
        
        # 验证广播兼容性
        expected_shape = [1, seq_len, 1, head_dim // 2]
        assert list(reshaped_freqs.shape) == expected_shape, f"形状不匹配: 期望{expected_shape}, 实际{list(reshaped_freqs.shape)}"
        
        # 测试广播是否工作
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        print(f"  x_complex.shape: {x_complex.shape}")
        
        # 尝试广播乘法
        result = x_complex * reshaped_freqs
        print(f"  广播乘法结果形状: {result.shape}")
        
        print("✅ reshape_for_broadcast 函数修复成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

def test_edge_cases():
    """测试边界情况"""
    print("\n🧪 测试边界情况...")
    
    # 测试不匹配的序列长度
    try:
        x = torch.randn(2, 10, 8, 64)
        freqs_cis = torch.randn(5, 32)  # 序列长度不匹配
        reshape_for_broadcast(freqs_cis, x)
        print("❌ 应该抛出序列长度不匹配错误")
        return False
    except AssertionError as e:
        print(f"✅ 正确捕获序列长度不匹配: {e}")
    
    # 测试不匹配的头维度
    try:
        x = torch.randn(2, 10, 8, 64)
        freqs_cis = torch.randn(10, 16)  # 头维度不匹配 (应该是32)
        reshape_for_broadcast(freqs_cis, x)
        print("❌ 应该抛出头维度不匹配错误")
        return False
    except AssertionError as e:
        print(f"✅ 正确捕获头维度不匹配: {e}")
    
    # 测试维度不足
    try:
        x = torch.randn(10, 64)  # 只有2维
        freqs_cis = torch.randn(10, 32)
        reshape_for_broadcast(freqs_cis, x)
        print("❌ 应该抛出维度不足错误")
        return False
    except AssertionError as e:
        print(f"✅ 正确捕获维度不足错误: {e}")
    
    return True

if __name__ == "__main__":
    print("🚀 开始测试 reshape_for_broadcast 修复...")
    
    success1 = test_reshape_for_broadcast()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n🎉 所有测试通过! reshape_for_broadcast 修复成功!")
    else:
        print("\n❌ 部分测试失败")
        sys.exit(1)

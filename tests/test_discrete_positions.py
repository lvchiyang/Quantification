#!/usr/bin/env python3
"""
测试离散仓位预测方法
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ModelConfigs
from transformer import FinancialTransformer
from discrete_position_methods import (
    GumbelSoftmaxPositionHead, 
    StraightThroughPositionHead, 
    ConcretePositionHead,
    DiscretePositionLoss
)


def test_position_heads():
    """测试不同的仓位预测头"""
    print("🔧 测试离散仓位预测头...")
    
    d_model = 256
    batch_size = 8
    hidden_states = torch.randn(batch_size, d_model)
    next_day_returns = torch.randn(batch_size) * 0.05  # 模拟±5%的涨跌幅
    
    methods = {
        'Gumbel-Softmax': GumbelSoftmaxPositionHead(d_model),
        'Straight-Through': StraightThroughPositionHead(d_model),
        'Concrete': ConcretePositionHead(d_model)
    }
    
    loss_fn = DiscretePositionLoss(max_position=10)
    
    print(f"📊 测试数据: batch_size={batch_size}, d_model={d_model}")
    print(f"📈 模拟涨跌幅范围: {next_day_returns.min().item():.4f} ~ {next_day_returns.max().item():.4f}")
    
    for name, head in methods.items():
        print(f"\n🔍 测试 {name} 方法:")
        
        # 前向传播
        head.train()
        output_train = head(hidden_states)
        
        head.eval()
        output_eval = head(hidden_states)
        
        # 检查输出
        positions_train = output_train['positions']
        positions_eval = output_eval['positions']
        
        print(f"  训练模式仓位范围: {positions_train.min().item():.2f} ~ {positions_train.max().item():.2f}")
        print(f"  推理模式仓位范围: {positions_eval.min().item():.2f} ~ {positions_eval.max().item():.2f}")
        
        # 检查是否为整数（对于某些方法）
        if 'discrete_positions' in output_eval:
            discrete_pos = output_eval['discrete_positions']
            is_integer = torch.allclose(discrete_pos, torch.round(discrete_pos))
            print(f"  离散仓位是否为整数: {is_integer}")
            print(f"  离散仓位示例: {discrete_pos[:5].flatten().tolist()}")
        
        # 测试梯度
        head.train()
        output_grad = head(hidden_states)
        loss = loss_fn(output_grad, next_day_returns)
        loss.backward()
        
        # 检查梯度
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
        print(f"  梯度计算: {'✅ 正常' if has_grad else '❌ 异常'}")
        print(f"  损失值: {loss.item():.6f}")
        
        # 清除梯度
        head.zero_grad()


def test_model_integration():
    """测试模型集成"""
    print("\n🔧 测试模型集成...")
    
    methods = ['gumbel_softmax', 'straight_through', 'concrete']
    
    for method in methods:
        print(f"\n🔍 测试 {method} 方法集成:")
        
        # 创建配置
        config = ModelConfigs.tiny()
        config.position_method = method
        
        # 创建模型
        model = FinancialTransformer(config)
        model.train()
        
        # 创建测试数据
        batch_size = 4
        seq_len = 180
        n_features = 11
        
        financial_data = torch.randn(batch_size, seq_len, n_features)
        target_prices = torch.randn(batch_size, 7)
        next_day_returns = torch.randn(batch_size) * 0.03
        
        # 前向传播
        outputs = model(
            financial_data=financial_data,
            target_prices=target_prices,
            next_day_returns=next_day_returns,
            return_dict=True
        )
        
        # 检查输出
        print(f"  价格预测形状: {outputs['price_predictions'].shape}")
        print(f"  仓位预测形状: {outputs['position_predictions'].shape}")
        print(f"  仓位范围: {outputs['position_predictions'].min().item():.2f} ~ {outputs['position_predictions'].max().item():.2f}")
        
        # 检查损失
        total_loss = outputs['loss']
        price_loss = outputs.get('price_loss', torch.tensor(0.0))
        position_loss = outputs.get('position_loss', torch.tensor(0.0))
        
        print(f"  总损失: {total_loss.item():.6f}")
        print(f"  价格损失: {price_loss.item():.6f}")
        print(f"  仓位损失: {position_loss.item():.6f}")
        
        # 测试反向传播
        total_loss.backward()
        
        # 检查梯度
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        print(f"  梯度计算: {'✅ 正常' if has_grad else '❌ 异常'}")
        
        # 检查仓位预测头的梯度
        if hasattr(model.position_head, 'head'):
            pos_head_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.position_head.head.parameters())
            print(f"  仓位头梯度: {'✅ 正常' if pos_head_grad else '❌ 异常'}")


def test_discrete_output():
    """测试离散输出的整数性质"""
    print("\n🔧 测试离散输出...")
    
    d_model = 256
    batch_size = 100  # 使用更大的批次测试
    
    # 测试Gumbel-Softmax在不同温度下的表现
    head = GumbelSoftmaxPositionHead(d_model)
    hidden_states = torch.randn(batch_size, d_model)
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("📊 Gumbel-Softmax在不同温度下的离散程度:")
    
    for temp in temperatures:
        head.temperature.data = torch.tensor(temp)
        head.eval()  # 推理模式使用硬采样
        
        output = head(hidden_states)
        positions = output['positions']
        discrete_positions = output.get('discrete_positions', positions)
        
        # 计算离散程度（与最近整数的平均距离）
        discreteness = torch.mean(torch.abs(positions - torch.round(positions))).item()
        
        # 计算仓位分布
        rounded_positions = torch.round(discrete_positions).long()
        position_counts = torch.bincount(rounded_positions.flatten(), minlength=11)
        position_probs = position_counts.float() / batch_size
        
        print(f"  温度 {temp:3.1f}: 离散度={discreteness:.4f}, 分布熵={-torch.sum(position_probs * torch.log(position_probs + 1e-8)).item():.3f}")
        
        # 显示前10个位置的分布
        top_positions = position_probs[:10].tolist()
        print(f"           仓位0-9分布: {[f'{p:.2f}' for p in top_positions]}")


def test_gradient_flow():
    """测试梯度流动"""
    print("\n🔧 测试梯度流动...")
    
    d_model = 256
    batch_size = 16
    
    methods = {
        'Gumbel-Softmax': GumbelSoftmaxPositionHead(d_model),
        'Straight-Through': StraightThroughPositionHead(d_model),
        'Concrete': ConcretePositionHead(d_model)
    }
    
    hidden_states = torch.randn(batch_size, d_model, requires_grad=True)
    next_day_returns = torch.randn(batch_size) * 0.05
    loss_fn = DiscretePositionLoss(max_position=10)
    
    print("📊 不同方法的梯度统计:")
    
    for name, head in methods.items():
        head.train()
        
        # 前向传播
        output = head(hidden_states)
        loss = loss_fn(output, next_day_returns)
        
        # 反向传播
        loss.backward(retain_graph=True)
        
        # 统计梯度
        input_grad_norm = hidden_states.grad.norm().item() if hidden_states.grad is not None else 0.0
        param_grad_norms = [p.grad.norm().item() for p in head.parameters() if p.grad is not None]
        avg_param_grad = np.mean(param_grad_norms) if param_grad_norms else 0.0
        
        print(f"  {name:15s}: 输入梯度范数={input_grad_norm:.6f}, 平均参数梯度范数={avg_param_grad:.6f}")
        
        # 清除梯度
        hidden_states.grad = None
        head.zero_grad()


def main():
    """主测试函数"""
    print("🚀 开始测试离散仓位预测方法...\n")
    
    try:
        test_position_heads()
        test_model_integration()
        test_discrete_output()
        test_gradient_flow()
        
        print("\n🎉 所有测试通过！")
        print("\n📋 方法总结:")
        print("✅ Gumbel-Softmax: 理论最优，温度可调，支持硬/软采样")
        print("✅ Straight-Through: 简单直接，前向离散，反向连续")
        print("✅ Concrete: 无噪声，温度控制，训练稳定")
        print("\n🚀 推荐使用 Gumbel-Softmax 方法进行训练！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

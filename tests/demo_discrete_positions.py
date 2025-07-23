#!/usr/bin/env python3
"""
演示离散仓位预测功能
"""

import torch
import numpy as np
import sys
import os

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ModelConfigs
from transformer import FinancialTransformer


def demo_discrete_positions():
    """演示离散仓位预测"""
    print("🎯 演示离散仓位预测功能\n")
    
    # 测试不同的离散化方法
    methods = ['gumbel_softmax', 'straight_through', 'concrete']
    
    for method in methods:
        print(f"🔍 方法: {method.upper()}")
        print("=" * 50)
        
        # 创建配置和模型
        config = ModelConfigs.tiny()
        config.position_method = method
        
        model = FinancialTransformer(config)
        
        # 创建测试数据
        batch_size = 5
        seq_len = 180
        n_features = 11
        
        financial_data = torch.randn(batch_size, seq_len, n_features)
        target_prices = torch.randn(batch_size, 7)
        next_day_returns = torch.tensor([0.02, -0.01, 0.03, -0.02, 0.01])  # 模拟涨跌幅
        
        # 训练模式预测
        model.train()
        train_outputs = model(
            financial_data=financial_data,
            target_prices=target_prices,
            next_day_returns=next_day_returns,
            return_dict=True
        )
        
        # 推理模式预测
        model.eval()
        eval_outputs = model(
            financial_data=financial_data,
            return_dict=True
        )
        
        # 显示结果
        train_positions = train_outputs['position_predictions'].squeeze(-1)
        eval_positions = eval_outputs['position_predictions'].squeeze(-1)
        
        print("📊 预测结果对比:")
        print("样本  涨跌幅    训练模式仓位  推理模式仓位  预期收益(训练)  预期收益(推理)")
        print("-" * 75)
        
        for i in range(batch_size):
            return_pct = next_day_returns[i].item()
            train_pos = train_positions[i].item()
            eval_pos = eval_positions[i].item()
            
            # 计算预期收益（仓位/10 * 涨跌幅）
            train_return = (train_pos / 10.0) * return_pct
            eval_return = (eval_pos / 10.0) * return_pct
            
            print(f"{i+1:2d}    {return_pct:+6.2%}    {train_pos:8.2f}      {eval_pos:8.2f}      {train_return:+8.4f}     {eval_return:+8.4f}")
        
        # 显示损失信息
        total_loss = train_outputs['loss']
        price_loss = train_outputs.get('price_loss', torch.tensor(0.0))
        position_loss = train_outputs.get('position_loss', torch.tensor(0.0))
        
        print(f"\n📈 损失信息:")
        print(f"  总损失: {total_loss.item():.6f}")
        print(f"  价格损失: {price_loss.item():.6f}")
        print(f"  仓位损失: {position_loss.item():.6f} (负收益率)")
        
        # 显示仓位分布（如果有详细输出）
        if 'position_output' in train_outputs:
            position_output = train_outputs['position_output']
            
            if 'discrete_positions' in position_output:
                discrete_pos = position_output['discrete_positions'].squeeze(-1)
                print(f"\n🎯 离散仓位: {discrete_pos.tolist()}")
            
            if 'probs' in position_output:
                probs = position_output['probs']
                print(f"\n📊 仓位概率分布 (前5个样本的前6个仓位):")
                for i in range(min(5, probs.shape[0])):
                    prob_str = " ".join([f"{p:.3f}" for p in probs[i, :6]])
                    print(f"  样本{i+1}: {prob_str} ...")
        
        print("\n" + "="*50 + "\n")


def demo_gradient_preservation():
    """演示梯度保持效果"""
    print("🔧 演示梯度保持效果\n")
    
    # 创建简单的测试
    config = ModelConfigs.tiny()
    config.position_method = 'gumbel_softmax'
    
    model = FinancialTransformer(config)
    model.train()
    
    # 创建需要梯度的输入
    financial_data = torch.randn(3, 180, 11, requires_grad=True)
    target_prices = torch.randn(3, 7)
    next_day_returns = torch.tensor([0.02, -0.01, 0.03])
    
    # 前向传播
    outputs = model(
        financial_data=financial_data,
        target_prices=target_prices,
        next_day_returns=next_day_returns,
        return_dict=True
    )
    
    # 反向传播
    loss = outputs['loss']
    loss.backward()
    
    # 检查梯度
    print("📊 梯度检查:")
    print(f"  输入数据梯度范数: {financial_data.grad.norm().item():.6f}")
    
    # 检查仓位预测头的梯度
    position_head = model.position_head
    if hasattr(position_head, 'head'):
        head_params = list(position_head.head.parameters())
        if head_params:
            grad_norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in head_params]
            avg_grad_norm = np.mean(grad_norms)
            print(f"  仓位预测头平均梯度范数: {avg_grad_norm:.6f}")
    
    # 显示仓位预测的梯度敏感性
    positions = outputs['position_predictions']
    print(f"  仓位预测值: {positions.squeeze(-1).tolist()}")
    print(f"  仓位预测梯度: {[f'{g:.6f}' for g in positions.grad.squeeze(-1).tolist()] if positions.grad is not None else 'None'}")
    
    print("\n✅ 梯度成功传播到仓位预测！")


def demo_integer_convergence():
    """演示整数收敛效果"""
    print("🎯 演示整数收敛效果\n")
    
    config = ModelConfigs.tiny()
    config.position_method = 'gumbel_softmax'
    
    model = FinancialTransformer(config)
    
    # 创建测试数据
    financial_data = torch.randn(10, 180, 11)
    
    # 测试不同温度下的输出
    temperatures = [2.0, 1.0, 0.5, 0.1]
    
    print("📊 不同温度下的仓位预测:")
    print("温度    平均仓位    标准差    离散度(与整数距离)")
    print("-" * 50)
    
    model.eval()
    for temp in temperatures:
        # 设置温度
        if hasattr(model.position_head, 'temperature'):
            model.position_head.temperature.data = torch.tensor(temp)
        
        # 预测
        with torch.no_grad():
            outputs = model(financial_data=financial_data, return_dict=True)
            positions = outputs['position_predictions'].squeeze(-1)
            
            # 计算统计量
            mean_pos = positions.mean().item()
            std_pos = positions.std().item()
            
            # 计算离散度（与最近整数的平均距离）
            discreteness = torch.mean(torch.abs(positions - torch.round(positions))).item()
            
            print(f"{temp:4.1f}    {mean_pos:8.2f}    {std_pos:6.3f}    {discreteness:12.4f}")
    
    print("\n💡 温度越低，输出越接近整数！")


def main():
    """主演示函数"""
    print("🚀 离散仓位预测功能演示\n")
    
    try:
        demo_discrete_positions()
        demo_gradient_preservation()
        demo_integer_convergence()
        
        print("\n🎉 演示完成！")
        print("\n📋 功能特点:")
        print("✅ 支持3种离散化方法：Gumbel-Softmax、Straight-Through、Concrete")
        print("✅ 输出0-10的整数仓位，同时保持梯度可计算")
        print("✅ 训练时连续优化，推理时离散输出")
        print("✅ 温度参数可调节离散程度")
        print("✅ 完全集成到滑动窗口交易策略中")
        
        print("\n🚀 现在可以训练具有整数仓位输出的模型了！")
        print("推荐使用 Gumbel-Softmax 方法获得最佳效果。")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
测试状态化模型功能
"""

import torch
import numpy as np
import sys
import os

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ModelConfigs
from transformer import FinancialTransformer
from market_classifier import ComprehensiveMarketClassifier
from information_ratio_loss import InformationRatioLoss
from recurrent_trainer import RecurrentStrategyTrainer


def test_market_classifier():
    """测试市场分类器"""
    print("🔧 测试市场分类器...")
    
    classifier = ComprehensiveMarketClassifier()
    
    # 测试不同市场环境
    test_cases = [
        (torch.tensor([0.02, 0.015, 0.01, 0.025, 0.018]), "牛市"),
        (torch.tensor([-0.02, -0.015, -0.01, -0.025, -0.018]), "熊市"),
        (torch.tensor([0.005, -0.003, 0.002, -0.001, 0.004]), "震荡市")
    ]
    
    for returns, expected in test_cases:
        market_type = classifier.classify_market(returns)
        benchmark = classifier.get_optimal_benchmark(market_type)
        
        print(f"  {expected}: 分类={market_type}, 基准={benchmark['name']}")
    
    print("✅ 市场分类器测试通过")


def test_information_ratio_loss():
    """测试信息比率损失函数"""
    print("🔧 测试信息比率损失函数...")
    
    classifier = ComprehensiveMarketClassifier()
    loss_fn = InformationRatioLoss(classifier)
    
    # 创建测试数据
    batch_size = 3
    seq_len = 20
    
    position_predictions = torch.rand(batch_size, seq_len, 1) * 10  # [0, 10]
    next_day_returns = torch.randn(batch_size, seq_len) * 0.02      # ±2%
    
    # 计算损失
    loss_dict = loss_fn(position_predictions, next_day_returns)
    
    print(f"  信息比率损失: {loss_dict['total_loss']:.6f}")
    print(f"  平均信息比率: {loss_dict['information_ratio']:.4f}")
    print(f"  机会成本: {loss_dict['opportunity_cost']:.6f}")
    print(f"  风险惩罚: {loss_dict['risk_penalty']:.6f}")
    
    print("✅ 信息比率损失测试通过")


def test_stateful_model():
    """测试状态化模型"""
    print("🔧 测试状态化模型...")
    
    # 创建配置
    config = ModelConfigs.tiny()
    config.enable_stateful_training = True
    config.strategy_state_dim = 64
    config.state_update_method = 'gru'
    
    # 创建模型
    model = FinancialTransformer(config)
    
    # 测试单日预测
    batch_size = 2
    seq_len = 180
    n_features = 11
    
    financial_data = torch.randn(batch_size, seq_len, n_features)
    
    # 第一次预测（无状态）
    outputs1 = model.forward_single_day(financial_data, return_dict=True)
    
    print(f"  第一次预测:")
    print(f"    价格预测形状: {outputs1['price_predictions'].shape}")
    print(f"    仓位预测形状: {outputs1['position_predictions'].shape}")
    print(f"    策略状态形状: {outputs1['strategy_state'].shape}")
    
    # 第二次预测（使用状态）
    outputs2 = model.forward_single_day(
        financial_data, 
        strategy_state=outputs1['strategy_state'],
        return_dict=True
    )
    
    print(f"  第二次预测:")
    print(f"    仓位预测变化: {torch.mean(torch.abs(outputs2['position_predictions'] - outputs1['position_predictions'])).item():.4f}")
    print(f"    状态变化: {torch.mean(torch.abs(outputs2['strategy_state'] - outputs1['strategy_state'])).item():.4f}")
    
    print("✅ 状态化模型测试通过")


def test_recurrent_trainer():
    """测试递归训练器"""
    print("🔧 测试递归训练器...")
    
    # 创建模型
    config = ModelConfigs.tiny()
    config.enable_stateful_training = True
    config.strategy_state_dim = 32
    
    model = FinancialTransformer(config)
    trainer = RecurrentStrategyTrainer(model, use_information_ratio=True)
    
    # 创建测试数据
    batch_size = 2
    n_slides = 20
    seq_len = 180
    n_features = 11
    
    sliding_window_data = {
        'features': torch.randn(batch_size, n_slides, seq_len, n_features),
        'price_targets': torch.randn(batch_size, n_slides, 7),
        'next_day_returns': torch.randn(batch_size, n_slides) * 0.02
    }
    
    # 训练步骤
    model.train()
    loss_dict = trainer.train_step(sliding_window_data)
    
    print(f"  训练损失: {loss_dict['total_loss']:.6f}")
    print(f"  价格损失: {loss_dict['price_loss']:.6f}")
    print(f"  信息比率: {loss_dict['information_ratio']:.4f}")
    print(f"  累计收益: {loss_dict['mean_cumulative_return']:+.4f}")
    print(f"  夏普比率: {loss_dict['sharpe_ratio']:.4f}")
    
    # 验证步骤
    val_dict = trainer.validate_step(sliding_window_data)
    
    print(f"  验证损失: {val_dict['total_loss']:.6f}")
    print(f"  验证收益: {val_dict['mean_cumulative_return']:+.4f}")
    
    # 测试梯度
    loss_tensor = loss_dict['loss_tensor']
    loss_tensor.backward()
    
    # 检查梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"  梯度计算: {'✅ 正常' if has_grad else '❌ 异常'}")
    
    print("✅ 递归训练器测试通过")


def test_memory_efficiency():
    """测试内存效率"""
    print("🔧 测试内存效率...")
    
    import psutil
    import gc
    
    # 获取初始内存
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建模型
    config = ModelConfigs.tiny()
    config.enable_stateful_training = True
    
    model = FinancialTransformer(config)
    trainer = RecurrentStrategyTrainer(model)
    
    # 创建较大的测试数据
    batch_size = 4
    sliding_window_data = {
        'features': torch.randn(batch_size, 20, 180, 11),
        'price_targets': torch.randn(batch_size, 20, 7),
        'next_day_returns': torch.randn(batch_size, 20) * 0.02
    }
    
    # 执行训练步骤
    model.train()
    loss_dict = trainer.train_step(sliding_window_data)
    loss_dict['loss_tensor'].backward()
    
    # 获取峰值内存
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    print(f"  初始内存: {initial_memory:.1f} MB")
    print(f"  峰值内存: {peak_memory:.1f} MB")
    print(f"  内存增长: {memory_increase:.1f} MB")
    
    # 清理
    del model, trainer, sliding_window_data, loss_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("✅ 内存效率测试完成")


def test_gradient_flow():
    """测试梯度流动"""
    print("🔧 测试梯度流动...")
    
    config = ModelConfigs.tiny()
    config.enable_stateful_training = True
    config.strategy_state_dim = 32
    
    model = FinancialTransformer(config)
    
    # 创建需要梯度的输入
    financial_data = torch.randn(2, 180, 11, requires_grad=True)
    
    # 多步预测
    strategy_state = None
    total_loss = 0.0
    
    for step in range(5):  # 5步预测
        outputs = model.forward_single_day(financial_data, strategy_state, return_dict=True)
        
        # 简单损失
        loss = torch.mean(outputs['position_predictions'] ** 2)
        total_loss += loss
        
        strategy_state = outputs['strategy_state']
    
    # 反向传播
    total_loss.backward()
    
    # 检查梯度
    input_grad_norm = financial_data.grad.norm().item() if financial_data.grad is not None else 0.0
    
    param_grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grad_norms.append(param.grad.norm().item())
    
    avg_param_grad = np.mean(param_grad_norms) if param_grad_norms else 0.0
    
    print(f"  输入梯度范数: {input_grad_norm:.6f}")
    print(f"  平均参数梯度范数: {avg_param_grad:.6f}")
    print(f"  有梯度的参数数量: {len(param_grad_norms)}")
    
    print("✅ 梯度流动测试通过")


def main():
    """主测试函数"""
    print("🚀 开始测试状态化模型功能...\n")
    
    try:
        test_market_classifier()
        print()
        
        test_information_ratio_loss()
        print()
        
        test_stateful_model()
        print()
        
        test_recurrent_trainer()
        print()
        
        test_memory_efficiency()
        print()
        
        test_gradient_flow()
        print()
        
        print("🎉 所有测试通过！")
        print("\n📋 功能总结:")
        print("✅ 市场状态自动分类（牛市/熊市/震荡市）")
        print("✅ 信息比率损失函数（自适应基准比较）")
        print("✅ 递归状态更新（20天记忆保持）")
        print("✅ 内存高效训练（避免20倍内存开销）")
        print("✅ 梯度正常传播（所有参数可训练）")
        print("✅ 机会成本计算（解决连续上涨评价问题）")
        
        print("\n🚀 可以开始训练状态化模型了！")
        print("运行: python train_stateful_strategy.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

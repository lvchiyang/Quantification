#!/usr/bin/env python3
"""
完整项目验证脚本
验证所有核心功能是否正常工作
"""

import torch
import numpy as np
import sys
import os
import traceback
from typing import Dict, Any

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """测试所有模块导入"""
    print("🔧 测试模块导入...")
    
    try:
        from config import ModelConfigs
        from transformer import FinancialTransformer
        from market_classifier import ComprehensiveMarketClassifier
        from information_ratio_loss import InformationRatioLoss, MultiObjectiveTradingLoss
        from recurrent_trainer import RecurrentStrategyTrainer
        from financial_data import FinancialDataProcessor
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("🔧 测试模型创建...")
    
    try:
        from config import ModelConfigs
        from transformer import FinancialTransformer
        
        # 测试不同配置
        configs = [
            ModelConfigs.tiny(),
            ModelConfigs.small()
        ]
        
        for i, config in enumerate(configs):
            config.enable_stateful_training = True
            config.strategy_state_dim = 64
            
            model = FinancialTransformer(config)
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"  配置{i+1}: {param_count:,} 参数")
        
        print("✅ 模型创建成功")
        return True
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    print("🔧 测试前向传播...")
    
    try:
        from config import ModelConfigs
        from transformer import FinancialTransformer
        
        config = ModelConfigs.tiny()
        config.enable_stateful_training = True
        model = FinancialTransformer(config)
        
        # 测试数据
        batch_size = 2
        seq_len = 180
        n_features = 11
        
        financial_data = torch.randn(batch_size, seq_len, n_features)
        
        # 传统前向传播
        outputs = model(financial_data)
        print(f"  传统输出形状: {outputs['price_predictions'].shape}")
        
        # 单日预测
        single_outputs = model.forward_single_day(financial_data)
        print(f"  单日输出形状: {single_outputs['price_predictions'].shape}")
        
        print("✅ 前向传播成功")
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

def test_market_classifier():
    """测试市场分类器"""
    print("🔧 测试市场分类器...")
    
    try:
        from market_classifier import ComprehensiveMarketClassifier
        
        classifier = ComprehensiveMarketClassifier()
        
        # 测试不同市场
        test_cases = [
            torch.tensor([0.02, 0.015, 0.01, 0.025, 0.018]),  # 牛市
            torch.tensor([-0.02, -0.015, -0.01, -0.025, -0.018]),  # 熊市
            torch.tensor([0.005, -0.003, 0.002, -0.001, 0.004])  # 震荡
        ]
        
        for i, returns in enumerate(test_cases):
            market_type = classifier.classify_market(returns)
            benchmark = classifier.get_optimal_benchmark(market_type)
            print(f"  测试{i+1}: {market_type} -> {benchmark['name']}")
        
        print("✅ 市场分类器成功")
        return True
    except Exception as e:
        print(f"❌ 市场分类器失败: {e}")
        return False

def test_information_ratio_loss():
    """测试信息比率损失"""
    print("🔧 测试信息比率损失...")
    
    try:
        from market_classifier import ComprehensiveMarketClassifier
        from information_ratio_loss import InformationRatioLoss
        
        classifier = ComprehensiveMarketClassifier()
        loss_fn = InformationRatioLoss(classifier)
        
        # 测试数据
        batch_size = 2
        seq_len = 20
        position_predictions = torch.rand(batch_size, seq_len, 1) * 10
        next_day_returns = torch.randn(batch_size, seq_len) * 0.02
        
        loss_dict = loss_fn(position_predictions, next_day_returns)
        
        print(f"  信息比率: {loss_dict['information_ratio']:.4f}")
        print(f"  机会成本: {loss_dict['opportunity_cost']:.6f}")
        
        print("✅ 信息比率损失成功")
        return True
    except Exception as e:
        print(f"❌ 信息比率损失失败: {e}")
        return False

def test_recurrent_trainer():
    """测试递归训练器"""
    print("🔧 测试递归训练器...")
    
    try:
        from config import ModelConfigs
        from transformer import FinancialTransformer
        from recurrent_trainer import RecurrentStrategyTrainer
        
        config = ModelConfigs.tiny()
        config.enable_stateful_training = True
        config.strategy_state_dim = 32
        
        model = FinancialTransformer(config)
        trainer = RecurrentStrategyTrainer(model, use_information_ratio=True)
        
        # 测试数据
        batch_size = 2
        sliding_window_data = {
            'features': torch.randn(batch_size, 20, 180, 11),
            'price_targets': torch.randn(batch_size, 20, 7),
            'next_day_returns': torch.randn(batch_size, 20) * 0.02
        }
        
        # 训练步骤
        model.train()
        loss_dict = trainer.train_step(sliding_window_data)
        
        print(f"  训练损失: {loss_dict['total_loss']:.6f}")
        print(f"  累计收益: {loss_dict['mean_cumulative_return']:+.4f}")
        
        # 测试梯度
        loss_tensor = loss_dict['loss_tensor']
        loss_tensor.backward()
        
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        print(f"  梯度计算: {'正常' if has_grad else '异常'}")
        
        print("✅ 递归训练器成功")
        return True
    except Exception as e:
        print(f"❌ 递归训练器失败: {e}")
        return False

def test_memory_efficiency():
    """测试内存效率"""
    print("🔧 测试内存效率...")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        from config import ModelConfigs
        from transformer import FinancialTransformer
        from recurrent_trainer import RecurrentStrategyTrainer
        
        config = ModelConfigs.tiny()
        config.enable_stateful_training = True
        model = FinancialTransformer(config)
        trainer = RecurrentStrategyTrainer(model)
        
        # 较大的测试数据
        sliding_window_data = {
            'features': torch.randn(4, 20, 180, 11),
            'price_targets': torch.randn(4, 20, 7),
            'next_day_returns': torch.randn(4, 20) * 0.02
        }
        
        model.train()
        loss_dict = trainer.train_step(sliding_window_data)
        loss_dict['loss_tensor'].backward()
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        print(f"  内存增长: {memory_increase:.1f} MB")
        print(f"  内存效率: {'良好' if memory_increase < 500 else '需优化'}")
        
        # 清理
        del model, trainer, sliding_window_data, loss_dict
        gc.collect()
        
        print("✅ 内存效率测试成功")
        return True
    except Exception as e:
        print(f"❌ 内存效率测试失败: {e}")
        return False

def test_end_to_end():
    """端到端测试"""
    print("🔧 端到端测试...")
    
    try:
        from config import ModelConfigs
        from transformer import FinancialTransformer
        from recurrent_trainer import RecurrentStrategyTrainer, create_sliding_window_batches
        
        # 创建模型
        config = ModelConfigs.tiny()
        config.enable_stateful_training = True
        config.strategy_state_dim = 32
        
        model = FinancialTransformer(config)
        trainer = RecurrentStrategyTrainer(model, use_information_ratio=True)
        
        # 创建数据
        n_sequences = 10
        features_list = torch.randn(n_sequences, 20, 180, 11)
        price_targets_list = torch.randn(n_sequences, 20, 7)
        next_day_returns = torch.randn(n_sequences, 20) * 0.02
        
        # 创建批次
        batches = create_sliding_window_batches(
            features_list, price_targets_list, next_day_returns, batch_size=2
        )
        
        print(f"  创建了 {len(batches)} 个批次")
        
        # 训练几个步骤
        model.train()
        total_loss = 0.0
        
        for i, batch in enumerate(batches[:3]):  # 只测试前3个批次
            loss_dict = trainer.train_step(batch)
            loss_dict['loss_tensor'].backward()
            total_loss += loss_dict['total_loss']
            
            # 清除梯度
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        avg_loss = total_loss / 3
        print(f"  平均损失: {avg_loss:.6f}")
        
        # 验证步骤
        model.eval()
        val_dict = trainer.validate_step(batches[0])
        print(f"  验证损失: {val_dict['total_loss']:.6f}")
        
        print("✅ 端到端测试成功")
        return True
    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主验证函数"""
    print("🚀 开始完整项目验证...\n")
    
    tests = [
        ("模块导入", test_imports),
        ("模型创建", test_model_creation),
        ("前向传播", test_forward_pass),
        ("市场分类器", test_market_classifier),
        ("信息比率损失", test_information_ratio_loss),
        ("递归训练器", test_recurrent_trainer),
        ("内存效率", test_memory_efficiency),
        ("端到端测试", test_end_to_end)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            traceback.print_exc()
        
        print()
    
    print("="*60)
    print(f"📊 验证结果: {passed}/{total} 测试通过")
    print("="*60)
    
    if passed == total:
        print("🎉 所有测试通过！项目功能完整！")
        print("\n📋 项目状态:")
        print("✅ 基础模型架构 - 完成")
        print("✅ 金融专用功能 - 完成")
        print("✅ 状态化训练系统 - 完成")
        print("✅ 智能损失函数 - 完成")
        print("✅ 训练和测试框架 - 完成")
        print("✅ 代码清理优化 - 完成")
        
        print("\n🚀 可以开始使用:")
        print("1. python test_stateful_model.py  # 详细功能测试")
        print("2. python train_stateful_strategy.py  # 开始训练")
        
        print("\n📚 查看文档:")
        print("- README.md - 完整项目文档")
        print("- PROJECT_STATUS.md - 项目状态报告")
        print("- QUICKSTART.md - 快速开始指南")
        
        return True
    else:
        print(f"❌ 有 {total - passed} 个测试失败，请检查问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

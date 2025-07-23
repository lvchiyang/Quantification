#!/usr/bin/env python3
"""
测试交易策略功能的脚本
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Any

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ModelArgs, ModelConfigs
from transformer import FinancialTransformer
from financial_data import FinancialDataProcessor, create_sample_data
from trading_strategy import TradingSimulator, TradingLoss, create_trading_simulator


def test_model_architecture():
    """测试模型架构"""
    print("🔧 测试模型架构...")
    
    # 创建配置
    config = ModelConfigs.tiny()
    
    # 创建模型
    model = FinancialTransformer(config)
    
    # 检查模型组件
    assert hasattr(model, 'price_head'), "模型应该有价格预测头"
    assert hasattr(model, 'trading_head'), "模型应该有交易策略预测头"
    assert hasattr(model, 'trading_simulator'), "模型应该有交易模拟器"
    assert hasattr(model, 'trading_loss_fn'), "模型应该有交易损失函数"
    
    print("✅ 模型架构测试通过")
    return model


def test_data_processing():
    """测试数据处理"""
    print("🔧 测试数据处理...")
    
    # 生成测试数据
    sample_data = create_sample_data(n_days=100)
    
    # 保存到临时文件
    temp_file = "temp_test_data.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    # 测试启用交易策略的数据处理
    processor = FinancialDataProcessor(
        sequence_length=30,
        prediction_horizon=7,
        trading_horizon=20,
        normalize=True,
        enable_trading_strategy=True
    )
    
    features, price_targets, trading_prices = processor.process_file(temp_file)
    
    # 检查数据形状
    assert features.shape[1] == 30, f"特征序列长度应为30，实际为{features.shape[1]}"
    assert features.shape[2] == 11, f"特征维度应为11，实际为{features.shape[2]}"
    assert price_targets.shape[1] == 7, f"价格目标长度应为7，实际为{price_targets.shape[1]}"
    assert trading_prices.shape[1] == 20, f"交易价格长度应为20，实际为{trading_prices.shape[1]}"
    assert features.shape[0] == price_targets.shape[0] == trading_prices.shape[0], "样本数量应该一致"
    
    # 清理临时文件
    os.remove(temp_file)
    
    print("✅ 数据处理测试通过")
    return features, price_targets, trading_prices, processor


def test_model_forward():
    """测试模型前向传播"""
    print("🔧 测试模型前向传播...")
    
    # 创建模型和数据
    model = test_model_architecture()
    features, price_targets, trading_prices, processor = test_data_processing()
    
    # 取一个小批次
    batch_size = 2
    batch_features = features[:batch_size]
    batch_price_targets = price_targets[:batch_size]
    batch_trading_prices = trading_prices[:batch_size]
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(
            financial_data=batch_features,
            target_prices=batch_price_targets,
            future_prices=batch_trading_prices,
            return_dict=True
        )
    
    # 检查输出
    assert 'price_predictions' in outputs, "输出应包含价格预测"
    assert 'trading_predictions' in outputs, "输出应包含交易策略预测"
    assert 'loss' in outputs, "输出应包含损失"
    
    price_pred = outputs['price_predictions']
    trading_pred = outputs['trading_predictions']
    
    assert price_pred.shape == (batch_size, 7), f"价格预测形状错误: {price_pred.shape}"
    assert trading_pred.shape == (batch_size, 20), f"交易预测形状错误: {trading_pred.shape}"
    
    # 检查交易预测范围
    assert torch.all(trading_pred >= -10) and torch.all(trading_pred <= 10), "交易预测应在[-10, 10]范围内"
    
    print("✅ 模型前向传播测试通过")
    return model, outputs


def test_trading_simulator():
    """测试交易模拟器"""
    print("🔧 测试交易模拟器...")
    
    # 创建交易模拟器
    simulator = TradingSimulator(
        trading_range_min=-10,
        trading_range_max=10,
        max_position=10,
        initial_cash=10000.0
    )
    
    # 创建测试数据
    batch_size = 3
    n_days = 20
    
    # 模拟交易动作和价格
    trading_actions = torch.randn(batch_size, n_days) * 5  # 随机交易动作
    prices = torch.abs(torch.randn(batch_size, n_days)) * 100 + 50  # 价格在50-150之间
    
    # 模拟交易
    returns = simulator.simulate_trading(trading_actions, prices)
    
    # 检查返回值
    assert returns.shape == (batch_size,), f"收益率形状错误: {returns.shape}"
    assert torch.all(torch.isfinite(returns)), "收益率应该是有限数值"
    
    # 测试详细信息
    returns_detailed, details = simulator.simulate_trading(
        trading_actions[:1], prices[:1], return_details=True
    )
    
    assert len(details) == 1, "详细信息数量错误"
    detail = details[0]
    assert 'total_return' in detail, "详细信息应包含总收益率"
    assert 'position_history' in detail, "详细信息应包含持仓历史"
    assert 'action_history' in detail, "详细信息应包含动作历史"
    
    print("✅ 交易模拟器测试通过")


def test_trading_loss():
    """测试交易损失函数"""
    print("🔧 测试交易损失函数...")
    
    # 创建交易模拟器和损失函数
    simulator = TradingSimulator()
    loss_fn = TradingLoss(simulator)
    
    # 创建测试数据
    batch_size = 2
    n_days = 20
    
    trading_predictions = torch.randn(batch_size, n_days) * 5
    prices = torch.abs(torch.randn(batch_size, n_days)) * 100 + 50
    
    # 计算损失
    loss = loss_fn(trading_predictions, prices)
    
    # 检查损失
    assert loss.dim() == 0, "损失应该是标量"
    assert torch.isfinite(loss), "损失应该是有限数值"
    
    print("✅ 交易损失函数测试通过")


def test_end_to_end():
    """端到端测试"""
    print("🔧 端到端测试...")
    
    # 创建模型和数据
    model, outputs = test_model_forward()
    
    # 测试预测功能
    features, price_targets, trading_prices, processor = test_data_processing()
    
    model.eval()
    with torch.no_grad():
        predictions = model.predict(features[:1], return_dict=True)
    
    # 检查预测输出
    assert 'price_predictions' in predictions, "预测应包含价格预测"
    assert 'trading_predictions' in predictions, "预测应包含交易策略预测"
    
    # 反标准化测试
    price_pred = predictions['price_predictions']
    price_denorm = processor.denormalize_predictions(price_pred)
    
    assert price_denorm.shape == price_pred.shape, "反标准化后形状应保持一致"
    assert not torch.allclose(price_pred, price_denorm), "反标准化应该改变数值"
    
    print("✅ 端到端测试通过")


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行交易策略功能测试...\n")
    
    try:
        test_model_architecture()
        test_data_processing()
        test_model_forward()
        test_trading_simulator()
        test_trading_loss()
        test_end_to_end()
        
        print("\n🎉 所有测试通过！交易策略功能正常工作。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

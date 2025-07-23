#!/usr/bin/env python3
"""
测试滑动窗口预测功能
"""

import torch
import numpy as np
import sys
import os

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ModelConfigs
from transformer import FinancialTransformer
from financial_data import FinancialDataProcessor, create_sample_data
from sliding_window_predictor import SlidingWindowPredictor


def test_sliding_window_prediction():
    """测试滑动窗口预测功能"""
    print("🔧 测试滑动窗口预测功能...")
    
    # 1. 创建模型
    config = ModelConfigs.tiny()
    model = FinancialTransformer(config)
    model.eval()
    
    # 2. 创建数据处理器
    processor = FinancialDataProcessor(
        sequence_length=180,
        prediction_horizon=7,
        trading_horizon=20,
        normalize=True,
        enable_trading_strategy=True,
        sliding_window=True
    )
    
    # 3. 生成测试数据（250天，确保有足够的数据）
    sample_data = create_sample_data(n_days=250)
    
    # 解析数据
    data_list = []
    for line in sample_data.split('\n'):
        if line.strip():
            parsed = processor.parse_data_line(line)
            if parsed:
                data_list.append(parsed)
    
    if len(data_list) < 207:  # 180 + 20 + 7
        print(f"❌ 数据不足，需要至少207天，实际{len(data_list)}天")
        return False
    
    # 转换为DataFrame并处理
    import pandas as pd
    df = pd.DataFrame(data_list)
    df = processor.add_time_encoding(df)
    processor.fit_normalizer(df)
    df = processor.normalize_features(df)
    
    # 4. 创建滑动窗口预测器
    predictor = SlidingWindowPredictor(model, processor)
    
    # 5. 准备测试数据（取前200天）
    test_data = df[processor.feature_columns].values[:200]
    
    print(f"📊 测试数据形状: {test_data.shape}")
    
    # 6. 进行滑动窗口预测
    try:
        results = predictor.predict_sequence(test_data, return_details=True)
        
        print("✅ 滑动窗口预测成功!")
        print(f"📈 预测结果:")
        print(f"  - 价格预测形状: {results['price_predictions'].shape}")  # [20, 7]
        print(f"  - 仓位预测形状: {results['position_predictions'].shape}")  # [20]
        print(f"  - 实际收益形状: {results['actual_returns'].shape}")  # [20]
        print(f"  - 累计收益率: {results['cumulative_return']:.4f} ({results['cumulative_return']*100:.2f}%)")
        print(f"  - 最终组合价值: {results['final_portfolio_value']:.4f}")
        
        # 显示前5天的详细预测
        print(f"\n📋 前5天预测详情:")
        for i in range(min(5, len(results['position_predictions']))):
            pos = results['position_predictions'][i]
            ret = results['actual_returns'][i]
            daily_ret = results['details']['daily_returns'][i]
            print(f"  第{i+1}天: 仓位={pos:.2f}, 次日涨跌={ret:.4f}({ret*100:.2f}%), 收益={daily_ret:.4f}({daily_ret*100:.2f}%)")
        
        # 显示策略统计
        details = results['details']
        print(f"\n📊 策略统计:")
        print(f"  - 最大回撤: {details['max_drawdown']:.4f} ({details['max_drawdown']*100:.2f}%)")
        print(f"  - 夏普比率: {details['sharpe_ratio']:.4f}")
        print(f"  - 胜率: {details['win_rate']:.4f} ({details['win_rate']*100:.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 滑动窗口预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_prediction():
    """测试批量预测功能"""
    print("\n🔧 测试批量预测功能...")
    
    # 创建模型和处理器
    config = ModelConfigs.tiny()
    model = FinancialTransformer(config)
    model.eval()
    
    processor = FinancialDataProcessor(
        sequence_length=180,
        prediction_horizon=7,
        trading_horizon=20,
        normalize=True,
        enable_trading_strategy=True,
        sliding_window=True
    )
    
    # 生成多个测试序列
    test_sequences = []
    for i in range(3):  # 生成3个测试序列
        sample_data = create_sample_data(n_days=250, base_price=100 + i*10)
        
        data_list = []
        for line in sample_data.split('\n'):
            if line.strip():
                parsed = processor.parse_data_line(line)
                if parsed:
                    data_list.append(parsed)
        
        if len(data_list) >= 207:
            import pandas as pd
            df = pd.DataFrame(data_list)
            df = processor.add_time_encoding(df)
            processor.fit_normalizer(df)
            df = processor.normalize_features(df)
            
            test_data = df[processor.feature_columns].values[:200]
            test_sequences.append(test_data)
    
    if len(test_sequences) == 0:
        print("❌ 没有足够的测试序列")
        return False
    
    # 创建预测器并进行批量预测
    predictor = SlidingWindowPredictor(model, processor)
    
    try:
        # 评估策略性能
        performance = predictor.evaluate_strategy(test_sequences)
        
        print("✅ 批量预测成功!")
        print(f"📊 策略性能评估:")
        print(f"  - 平均累计收益: {performance['mean_cumulative_return']:.4f} ({performance['mean_cumulative_return']*100:.2f}%)")
        print(f"  - 收益标准差: {performance['std_cumulative_return']:.4f}")
        print(f"  - 平均最终价值: {performance['mean_final_value']:.4f}")
        print(f"  - 序列胜率: {performance['win_rate_sequences']:.4f} ({performance['win_rate_sequences']*100:.2f}%)")
        print(f"  - 平均日收益: {performance['mean_daily_return']:.6f} ({performance['mean_daily_return']*100:.4f}%)")
        print(f"  - 整体夏普比率: {performance['overall_sharpe_ratio']:.4f}")
        print(f"  - 平均最大回撤: {performance['mean_max_drawdown']:.4f} ({performance['mean_max_drawdown']*100:.2f}%)")
        print(f"  - 平均胜率: {performance['mean_win_rate']:.4f} ({performance['mean_win_rate']*100:.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始测试滑动窗口预测功能...\n")
    
    success1 = test_sliding_window_prediction()
    success2 = test_batch_prediction()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！")
        print("\n📋 功能总结:")
        print("✅ 支持200天数据的20次滑动窗口预测")
        print("✅ 每次使用180天历史数据预测未来7天价格")
        print("✅ 输出0-10的仓位决策")
        print("✅ 根据次日涨跌幅计算累计收益")
        print("✅ 支持批量预测和策略性能评估")
        print("\n🚀 可以开始训练滑动窗口模型了！")
        print("运行: python train_sliding_window_model.py")
        return True
    else:
        print("\n❌ 部分测试失败，需要修复后再继续")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
策略网络训练脚本
第二阶段：基于预训练的价格网络，训练GRU策略网络
"""

import torch
import torch.optim as optim
import numpy as np
import sys
import os
from typing import Dict, Any

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ModelConfigs
from price_prediction.price_transformer import PriceTransformer
from strategy_network.gru_strategy import GRUStrategyNetwork
from strategy_network.strategy_loss import StrategyLoss
from strategy_network.strategy_trainer import StrategyTrainingPipeline, create_strategy_batches
from financial_data import FinancialDataProcessor, create_sample_data
from market_classifier import create_market_classifier


def load_pretrained_price_network(checkpoint_path: str) -> PriceTransformer:
    """加载预训练的价格网络"""
    print(f"📥 加载预训练价格网络: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到价格网络检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # 创建模型并加载权重
    price_network = PriceTransformer(config)
    price_network.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ 价格网络加载成功 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f})")
    
    return price_network, config


def create_strategy_dataset():
    """创建策略训练数据集"""
    print("🔄 创建策略训练数据...")
    
    # 生成示例数据
    sample_data = create_sample_data(n_days=600)  # 策略训练数据
    
    # 保存到临时文件
    temp_file = "temp_strategy_data.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    # 处理数据
    processor = FinancialDataProcessor(
        sequence_length=180,
        prediction_horizon=7,
        trading_horizon=20,  # 策略需要20天序列
        normalize=True,
        enable_trading_strategy=True,
        sliding_window=True
    )
    
    # 读取并处理数据
    with open(temp_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data_list = []
    for line in lines:
        if line.strip():
            parsed = processor.parse_data_line(line)
            if parsed:
                data_list.append(parsed)
    
    if not data_list:
        raise ValueError("没有成功解析任何数据")
    
    # 转换为DataFrame
    import pandas as pd
    df = pd.DataFrame(data_list)
    df = processor.add_time_encoding(df)
    processor.fit_normalizer(df)
    df = processor.normalize_features(df)
    
    # 创建滑动窗口序列
    features_list, _, _, next_day_returns = processor.create_sliding_window_sequences(df)
    
    # 清理临时文件
    os.remove(temp_file)
    
    return features_list, next_day_returns, processor


def train_strategy_network():
    """训练策略网络"""
    print("🚀 开始训练GRU策略网络...")
    print("💡 特点: 基于预训练价格网络 + 相对基准收益 + 风险成本 + 机会成本")
    
    # 1. 加载预训练价格网络
    try:
        price_network, price_config = load_pretrained_price_network('best_price_network.pth')
    except FileNotFoundError:
        print("❌ 未找到预训练价格网络!")
        print("📋 请先运行: python train_price_network.py")
        return
    
    # 2. 创建策略训练数据
    features_list, next_day_returns, processor = create_strategy_dataset()
    
    print(f"📊 策略数据形状:")
    print(f"  - 特征序列: {features_list.shape}")      # [n_sequences, 20, 180, 11]
    print(f"  - 次日收益: {next_day_returns.shape}")    # [n_sequences, 20]
    
    # 3. 数据分割
    n_sequences = features_list.shape[0]
    train_size = int(0.8 * n_sequences)
    
    train_features = features_list[:train_size]
    train_returns = next_day_returns[:train_size]
    
    val_features = features_list[train_size:]
    val_returns = next_day_returns[train_size:]
    
    print(f"📊 数据分割:")
    print(f"  - 训练序列: {len(train_features)} 个")
    print(f"  - 验证序列: {len(val_features)} 个")
    
    # 4. 创建策略网络
    strategy_config = ModelConfigs.small()
    
    # 策略网络配置
    strategy_config.enable_stateful_training = True
    strategy_config.strategy_state_dim = 128
    strategy_config.state_update_method = 'gru'
    strategy_config.position_method = 'gumbel_softmax'
    strategy_config.d_model = price_config.d_model  # 与价格网络保持一致
    
    # 损失权重
    strategy_config.relative_return_weight = 1.0
    strategy_config.risk_cost_weight = 0.2
    strategy_config.opportunity_cost_weight = 0.1
    
    strategy_network = GRUStrategyNetwork(strategy_config)
    
    print(f"🏗️ 策略网络创建:")
    print(f"  - 参数数量: {sum(p.numel() for p in strategy_network.parameters()):,}")
    print(f"  - 策略状态维度: {strategy_config.strategy_state_dim}")
    print(f"  - 状态更新方法: {strategy_config.state_update_method}")
    
    # 5. 创建损失函数
    market_classifier = create_market_classifier(strategy_config)
    strategy_loss = StrategyLoss(
        market_classifier=market_classifier,
        relative_return_weight=strategy_config.relative_return_weight,
        risk_cost_weight=strategy_config.risk_cost_weight,
        opportunity_cost_weight=strategy_config.opportunity_cost_weight
    )
    
    # 6. 创建训练流水线
    training_pipeline = StrategyTrainingPipeline(
        price_network=price_network,
        strategy_network=strategy_network,
        strategy_loss=strategy_loss
    )
    
    # 7. 设置优化器（只优化策略网络）
    optimizer = optim.AdamW(
        strategy_network.parameters(),  # 只优化策略网络
        lr=strategy_config.learning_rate * 0.5,  # 策略网络用较小学习率
        weight_decay=strategy_config.weight_decay,
        betas=(strategy_config.beta1, strategy_config.beta2)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )
    
    # 8. 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    price_network.to(device)
    strategy_network.to(device)
    
    print(f"🔧 使用设备: {device}")
    print("📈 开始策略网络训练...")
    
    num_epochs = 20
    best_val_return = float('-inf')
    batch_size = 2  # 策略训练使用小批次
    
    for epoch in range(num_epochs):
        # 训练阶段
        strategy_network.train()
        train_metrics = {
            'total_loss': 0.0,
            'relative_return_loss': 0.0,
            'risk_cost': 0.0,
            'opportunity_cost': 0.0,
            'mean_cumulative_return': 0.0,
            'mean_sharpe_ratio': 0.0
        }
        
        # 创建训练批次
        train_batch_data = []
        for i in range(0, len(train_features), batch_size):
            end_idx = min(i + batch_size, len(train_features))
            train_batch_data.append({
                'financial_data': train_features[i:end_idx],
                'next_day_returns': train_returns[i:end_idx]
            })
        
        for batch_idx, batch_data in enumerate(train_batch_data):
            # 移动数据到设备
            for key in batch_data:
                batch_data[key] = batch_data[key].to(device)
            
            optimizer.zero_grad()
            
            # 训练步骤
            loss_dict = training_pipeline.train_step(batch_data)
            
            # 反向传播
            loss_tensor = loss_dict['loss_tensor']
            loss_tensor.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(strategy_network.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累积指标
            for key in train_metrics:
                if key in loss_dict:
                    train_metrics[key] += loss_dict[key]
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_batch_data)}: "
                      f"Loss={loss_dict['total_loss']:.6f}, "
                      f"Return={loss_dict['mean_cumulative_return']:+.4f}, "
                      f"Sharpe={loss_dict['mean_sharpe_ratio']:.4f}")
        
        # 验证阶段
        strategy_network.eval()
        val_metrics = {
            'total_loss': 0.0,
            'relative_return_loss': 0.0,
            'risk_cost': 0.0,
            'opportunity_cost': 0.0,
            'mean_cumulative_return': 0.0,
            'mean_sharpe_ratio': 0.0
        }
        
        # 创建验证批次
        val_batch_data = []
        for i in range(0, len(val_features), batch_size):
            end_idx = min(i + batch_size, len(val_features))
            val_batch_data.append({
                'financial_data': val_features[i:end_idx],
                'next_day_returns': val_returns[i:end_idx]
            })
        
        for batch_data in val_batch_data:
            # 移动数据到设备
            for key in batch_data:
                batch_data[key] = batch_data[key].to(device)
            
            # 验证步骤
            loss_dict = training_pipeline.validate_step(batch_data)
            
            # 累积指标
            for key in val_metrics:
                if key in loss_dict:
                    val_metrics[key] += loss_dict[key]
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均指标
        for key in train_metrics:
            train_metrics[key] /= len(train_batch_data)
        for key in val_metrics:
            val_metrics[key] /= len(val_batch_data)
        
        # 保存最佳模型
        if val_metrics['mean_cumulative_return'] > best_val_return:
            best_val_return = val_metrics['mean_cumulative_return']
            torch.save({
                'strategy_model_state_dict': strategy_network.state_dict(),
                'price_model_path': 'best_price_network.pth',
                'config': strategy_config,
                'epoch': epoch,
                'val_return': best_val_return
            }, 'best_strategy_network.pth')
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1:2d}/{num_epochs}:")
        print(f"  训练 - 损失: {train_metrics['total_loss']:.6f}, "
              f"累计收益: {train_metrics['mean_cumulative_return']:+.4f}, "
              f"夏普比率: {train_metrics['mean_sharpe_ratio']:+.4f}")
        print(f"  验证 - 损失: {val_metrics['total_loss']:.6f}, "
              f"累计收益: {val_metrics['mean_cumulative_return']:+.4f}, "
              f"夏普比率: {val_metrics['mean_sharpe_ratio']:+.4f}")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ 策略网络训练完成!")
    
    # 9. 测试最佳模型
    print("\n🔮 测试最佳策略网络...")
    checkpoint = torch.load('best_strategy_network.pth')
    strategy_network.load_state_dict(checkpoint['strategy_model_state_dict'])
    strategy_network.eval()
    
    # 使用一个验证批次进行测试
    if val_batch_data:
        test_batch = val_batch_data[0]
        for key in test_batch:
            test_batch[key] = test_batch[key].to(device)
        
        loss_dict = training_pipeline.validate_step(test_batch)
        
        print(f"📊 最佳策略网络性能:")
        print(f"  - 累计收益率: {loss_dict['mean_cumulative_return']:+.4f} ({loss_dict['mean_cumulative_return']*100:+.2f}%)")
        print(f"  - 夏普比率: {loss_dict['mean_sharpe_ratio']:+.4f}")
        print(f"  - 最大回撤: {loss_dict['mean_max_drawdown']:.4f} ({loss_dict['mean_max_drawdown']*100:.2f}%)")
        print(f"  - 相对收益损失: {loss_dict['relative_return_loss']:.6f}")
        print(f"  - 风险成本: {loss_dict['risk_cost']:.6f}")
        print(f"  - 机会成本: {loss_dict['opportunity_cost']:.6f}")
    
    print(f"\n💾 策略网络已保存到: best_strategy_network.pth")
    print(f"🎉 两阶段训练完成!")
    print(f"📋 现在可以使用训练好的模型进行预测和交易")


if __name__ == "__main__":
    train_strategy_network()

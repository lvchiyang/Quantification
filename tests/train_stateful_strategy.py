#!/usr/bin/env python3
"""
状态化交易策略训练脚本
基于递归状态更新和信息比率损失的训练
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
from transformer import FinancialTransformer
from financial_data import FinancialDataProcessor, create_sample_data
from recurrent_trainer import RecurrentStrategyTrainer, create_sliding_window_batches


def create_stateful_dataset():
    """创建状态化训练数据集"""
    print("🔄 创建状态化训练数据...")
    
    # 生成更长的示例数据
    sample_data = create_sample_data(n_days=600)  # 更多数据用于训练
    
    # 保存到临时文件
    temp_file = "temp_stateful_data.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    # 处理数据
    processor = FinancialDataProcessor(
        sequence_length=180,
        prediction_horizon=7,
        trading_horizon=20,
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
    features_list, price_targets_list, _, next_day_returns = processor.create_sliding_window_sequences(df)
    
    # 清理临时文件
    os.remove(temp_file)
    
    return features_list, price_targets_list, next_day_returns, processor


def train_stateful_model():
    """训练状态化模型"""
    print("🚀 开始训练状态化交易策略模型...")
    
    # 1. 创建数据
    features_list, price_targets_list, next_day_returns, processor = create_stateful_dataset()
    
    print(f"📊 数据形状:")
    print(f"  - 特征序列: {features_list.shape}")      # [n_sequences, 20, 180, 11]
    print(f"  - 价格目标: {price_targets_list.shape}")  # [n_sequences, 20, 7]
    print(f"  - 次日收益: {next_day_returns.shape}")    # [n_sequences, 20]
    
    # 2. 数据分割
    n_sequences = features_list.shape[0]
    train_size = int(0.8 * n_sequences)
    
    train_features = features_list[:train_size]
    train_price_targets = price_targets_list[:train_size]
    train_returns = next_day_returns[:train_size]
    
    val_features = features_list[train_size:]
    val_price_targets = price_targets_list[train_size:]
    val_returns = next_day_returns[train_size:]
    
    print(f"📊 数据分割:")
    print(f"  - 训练序列: {len(train_features)} 个")
    print(f"  - 验证序列: {len(val_features)} 个")
    
    # 3. 创建批次数据
    batch_size = 2  # 由于递归计算，使用较小的批次
    train_batches = create_sliding_window_batches(
        train_features, train_price_targets, train_returns, batch_size
    )
    val_batches = create_sliding_window_batches(
        val_features, val_price_targets, val_returns, batch_size
    )
    
    print(f"📊 批次信息:")
    print(f"  - 训练批次: {len(train_batches)} 个")
    print(f"  - 验证批次: {len(val_batches)} 个")
    print(f"  - 每批次序列数: {batch_size}")
    
    # 4. 创建模型
    config = ModelConfigs.tiny()
    
    # 启用状态化训练
    config.enable_stateful_training = True
    config.strategy_state_dim = 128  # 较小的状态维度
    config.state_update_method = 'gru'
    config.position_method = 'gumbel_softmax'
    
    # 调整损失权重
    config.information_ratio_weight = 1.0
    config.opportunity_cost_weight = 0.1
    config.risk_adjustment_weight = 0.05
    config.state_regularization_weight = 0.001
    
    model = FinancialTransformer(config)
    
    print(f"🏗️ 模型创建完成:")
    print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 状态维度: {config.strategy_state_dim}")
    print(f"  - 状态更新方法: {config.state_update_method}")
    print(f"  - 仓位方法: {config.position_method}")
    
    # 5. 创建训练器
    trainer = RecurrentStrategyTrainer(model, use_information_ratio=True)
    
    # 6. 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate * 0.3,  # 降低学习率
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=25, eta_min=1e-6
    )
    
    # 7. 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"🔧 使用设备: {device}")
    print("📈 开始状态化训练...")
    print("💡 特点: 递归状态更新 + 信息比率损失 + 市场自适应基准")
    
    num_epochs = 25
    best_val_return = float('-inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_metrics = {
            'total_loss': 0.0,
            'price_loss': 0.0,
            'information_ratio_loss': 0.0,
            'information_ratio': 0.0,
            'opportunity_cost': 0.0,
            'risk_penalty': 0.0,
            'state_regularization': 0.0,
            'mean_cumulative_return': 0.0,
            'sharpe_ratio': 0.0
        }
        
        for batch_idx, batch_data in enumerate(train_batches):
            # 移动数据到设备
            for key in batch_data:
                batch_data[key] = batch_data[key].to(device)
            
            optimizer.zero_grad()
            
            # 执行递归训练步骤
            loss_dict = trainer.train_step(batch_data)
            
            # 反向传播
            loss_tensor = loss_dict['loss_tensor']
            loss_tensor.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累积指标
            for key in train_metrics:
                if key in loss_dict:
                    train_metrics[key] += loss_dict[key]
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_batches)}: "
                      f"Loss={loss_dict['total_loss']:.6f}, "
                      f"Return={loss_dict['mean_cumulative_return']:+.4f}, "
                      f"IR={loss_dict['information_ratio']:.4f}")
        
        # 验证阶段
        model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'price_loss': 0.0,
            'information_ratio_loss': 0.0,
            'information_ratio': 0.0,
            'opportunity_cost': 0.0,
            'risk_penalty': 0.0,
            'state_regularization': 0.0,
            'mean_cumulative_return': 0.0,
            'sharpe_ratio': 0.0
        }
        
        for batch_data in val_batches:
            # 移动数据到设备
            for key in batch_data:
                batch_data[key] = batch_data[key].to(device)
            
            # 验证步骤
            loss_dict = trainer.validate_step(batch_data)
            
            # 累积指标
            for key in val_metrics:
                if key in loss_dict:
                    val_metrics[key] += loss_dict[key]
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均指标
        for key in train_metrics:
            train_metrics[key] /= len(train_batches)
        for key in val_metrics:
            val_metrics[key] /= len(val_batches)
        
        # 保存最佳模型
        if val_metrics['mean_cumulative_return'] > best_val_return:
            best_val_return = val_metrics['mean_cumulative_return']
            torch.save(model.state_dict(), 'best_stateful_model.pth')
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1:2d}/{num_epochs}:")
        print(f"  训练 - 损失: {train_metrics['total_loss']:.6f}, "
              f"累计收益: {train_metrics['mean_cumulative_return']:+.4f}, "
              f"信息比率: {train_metrics['information_ratio']:+.4f}")
        print(f"  验证 - 损失: {val_metrics['total_loss']:.6f}, "
              f"累计收益: {val_metrics['mean_cumulative_return']:+.4f}, "
              f"信息比率: {val_metrics['information_ratio']:+.4f}")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ 状态化训练完成!")
    
    # 8. 测试最佳模型
    print("\n🔮 测试最佳模型...")
    model.load_state_dict(torch.load('best_stateful_model.pth'))
    model.eval()
    
    # 使用一个验证批次进行测试
    if val_batches:
        test_batch = val_batches[0]
        for key in test_batch:
            test_batch[key] = test_batch[key].to(device)
        
        loss_dict = trainer.validate_step(test_batch)
        
        print(f"📊 最佳模型性能:")
        print(f"  - 累计收益率: {loss_dict['mean_cumulative_return']:+.4f} ({loss_dict['mean_cumulative_return']*100:+.2f}%)")
        print(f"  - 信息比率: {loss_dict['information_ratio']:+.4f}")
        print(f"  - 夏普比率: {loss_dict['sharpe_ratio']:+.4f}")
        print(f"  - 最大回撤: {loss_dict['max_drawdown']:.4f} ({loss_dict['max_drawdown']*100:.2f}%)")
        print(f"  - 机会成本: {loss_dict['opportunity_cost']:.6f}")
        print(f"  - 风险惩罚: {loss_dict['risk_penalty']:.6f}")
        
        # 演示递归预测过程
        print(f"\n📈 演示20天递归预测过程:")
        features = test_batch['features'][:1]  # 取第一个序列
        
        model.eval()
        strategy_state = None
        positions_over_time = []
        
        with torch.no_grad():
            for slide in range(20):
                slide_features = features[:, slide, :, :]
                outputs = model.forward_single_day(
                    slide_features, strategy_state, return_dict=True
                )
                
                position = outputs['position_predictions'][0, 0].item()
                strategy_state = outputs['strategy_state']
                
                positions_over_time.append(round(position))
        
        print(f"  仓位序列: {positions_over_time}")
        print(f"  平均仓位: {np.mean(positions_over_time):.1f}")
        print(f"  仓位变化: {np.std(positions_over_time):.1f}")


if __name__ == "__main__":
    train_stateful_model()

#!/usr/bin/env python3
"""
价格预测网络训练脚本
第一阶段：训练Transformer价格预测网络
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
from price_prediction.price_transformer import PriceTransformer, PricePredictionLoss
from financial_data import FinancialDataProcessor, create_sample_data


def create_price_dataset():
    """创建价格预测数据集"""
    print("🔄 创建价格预测数据...")
    
    # 生成示例数据
    sample_data = create_sample_data(n_days=800)  # 更多数据用于价格预测训练
    
    # 保存到临时文件
    temp_file = "temp_price_data.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    # 处理数据
    processor = FinancialDataProcessor(
        sequence_length=180,
        prediction_horizon=7,
        trading_horizon=1,  # 价格预测不需要交易序列
        normalize=True,
        enable_trading_strategy=False,  # 只做价格预测
        sliding_window=False
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
    
    # 创建价格预测序列
    features_list, price_targets_list = processor.create_price_prediction_sequences(df)
    
    # 清理临时文件
    os.remove(temp_file)
    
    return features_list, price_targets_list, processor


def create_price_batches(features_list, price_targets_list, batch_size=8):
    """创建价格预测批次"""
    n_samples = len(features_list)
    batches = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        batch_features = torch.stack(features_list[i:end_idx])
        batch_targets = torch.stack(price_targets_list[i:end_idx])
        
        batches.append({
            'features': batch_features,
            'price_targets': batch_targets
        })
    
    return batches


def train_price_network():
    """训练价格预测网络"""
    print("🚀 开始训练价格预测网络...")
    print("💡 目标: 专门优化价格预测能力")
    
    # 1. 创建数据
    features_list, price_targets_list, processor = create_price_dataset()
    
    print(f"📊 数据形状:")
    print(f"  - 特征数量: {len(features_list)}")
    print(f"  - 特征维度: {features_list[0].shape}")      # [180, 11]
    print(f"  - 价格目标: {price_targets_list[0].shape}")  # [7]
    
    # 2. 数据分割
    n_samples = len(features_list)
    train_size = int(0.8 * n_samples)
    
    train_features = features_list[:train_size]
    train_targets = price_targets_list[:train_size]
    
    val_features = features_list[train_size:]
    val_targets = price_targets_list[train_size:]
    
    print(f"📊 数据分割:")
    print(f"  - 训练样本: {len(train_features)} 个")
    print(f"  - 验证样本: {len(val_features)} 个")
    
    # 3. 创建批次
    batch_size = 8
    train_batches = create_price_batches(train_features, train_targets, batch_size)
    val_batches = create_price_batches(val_features, val_targets, batch_size)
    
    print(f"📊 批次信息:")
    print(f"  - 训练批次: {len(train_batches)} 个")
    print(f"  - 验证批次: {len(val_batches)} 个")
    
    # 4. 创建模型
    config = ModelConfigs.small()  # 价格预测可以用稍大的模型
    
    model = PriceTransformer(config)
    loss_fn = PricePredictionLoss(loss_type='mse')
    
    print(f"🏗️ 价格预测模型:")
    print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 模型维度: {config.d_model}")
    print(f"  - 层数: {config.n_layers}")
    print(f"  - 注意力头数: {config.n_heads}")
    
    # 5. 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-6
    )
    
    # 6. 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"🔧 使用设备: {device}")
    print("📈 开始价格预测训练...")
    
    num_epochs = 30
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_direction_acc = 0.0
        
        for batch_idx, batch_data in enumerate(train_batches):
            # 移动数据到设备
            features = batch_data['features'].to(device)
            targets = batch_data['price_targets'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features, return_features=False, return_dict=True)
            predictions = outputs['price_predictions']
            
            # 计算损失
            loss_dict = loss_fn(predictions, targets)
            loss = loss_dict['loss']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累积指标
            train_loss += loss.item()
            train_mae += loss_dict['mae']
            train_direction_acc += loss_dict['direction_accuracy']
            
            # 打印进度
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_batches)}: "
                      f"Loss={loss.item():.6f}, "
                      f"MAE={loss_dict['mae']:.6f}, "
                      f"DirAcc={loss_dict['direction_accuracy']:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_direction_acc = 0.0
        
        with torch.no_grad():
            for batch_data in val_batches:
                features = batch_data['features'].to(device)
                targets = batch_data['price_targets'].to(device)
                
                outputs = model(features, return_features=False, return_dict=True)
                predictions = outputs['price_predictions']
                
                loss_dict = loss_fn(predictions, targets)
                
                val_loss += loss_dict['loss'].item()
                val_mae += loss_dict['mae']
                val_direction_acc += loss_dict['direction_accuracy']
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均指标
        train_loss /= len(train_batches)
        train_mae /= len(train_batches)
        train_direction_acc /= len(train_batches)
        
        val_loss /= len(val_batches)
        val_mae /= len(val_batches)
        val_direction_acc /= len(val_batches)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': val_loss
            }, 'best_price_network.pth')
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1:2d}/{num_epochs}:")
        print(f"  训练 - 损失: {train_loss:.6f}, MAE: {train_mae:.6f}, 方向准确率: {train_direction_acc:.4f}")
        print(f"  验证 - 损失: {val_loss:.6f}, MAE: {val_mae:.6f}, 方向准确率: {val_direction_acc:.4f}")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ 价格预测网络训练完成!")
    
    # 7. 测试最佳模型
    print("\n🔮 测试最佳价格预测模型...")
    checkpoint = torch.load('best_price_network.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 使用一个验证批次进行测试
    if val_batches:
        test_batch = val_batches[0]
        features = test_batch['features'].to(device)
        targets = test_batch['price_targets'].to(device)
        
        with torch.no_grad():
            outputs = model(features, return_features=True, return_dict=True)
            predictions = outputs['price_predictions']
            features_extracted = outputs['strategy_features']
            
            loss_dict = loss_fn(predictions, targets)
            
            print(f"📊 最佳模型性能:")
            print(f"  - 预测损失: {loss_dict['loss']:.6f}")
            print(f"  - 平均绝对误差: {loss_dict['mae']:.6f}")
            print(f"  - 相对误差: {loss_dict['relative_error']:.4f}")
            print(f"  - 方向准确率: {loss_dict['direction_accuracy']:.4f}")
            print(f"  - 提取特征维度: {features_extracted.shape}")
            
            # 显示一个预测示例
            sample_pred = predictions[0].cpu().numpy()
            sample_target = targets[0].cpu().numpy()
            
            print(f"\n📈 预测示例:")
            print(f"  预测价格: {sample_pred}")
            print(f"  真实价格: {sample_target}")
            print(f"  预测误差: {sample_pred - sample_target}")
    
    print(f"\n💾 模型已保存到: best_price_network.pth")
    print(f"📋 下一步: 运行 python train_strategy_network.py 训练策略网络")


if __name__ == "__main__":
    train_price_network()

#!/usr/bin/env python3
"""
滑动窗口金融量化模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os
from typing import Tuple

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ModelArgs, ModelConfigs
from transformer import FinancialTransformer
from financial_data import FinancialDataProcessor, create_sample_data


def create_sliding_window_dataset() -> Tuple[torch.Tensor, ...]:
    """创建滑动窗口数据集"""
    print("🔄 创建滑动窗口金融数据...")
    
    # 生成更长的示例数据（至少207天：180+20+7）
    sample_data = create_sample_data(n_days=500)
    
    # 保存到临时文件
    temp_file = "temp_sliding_window_data.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    # 处理数据
    processor = FinancialDataProcessor(
        sequence_length=180,  # 使用180天历史数据
        prediction_horizon=7,  # 预测7天价格
        trading_horizon=20,    # 20天滑动窗口
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
    
    # 添加时间编码
    df = processor.add_time_encoding(df)
    
    # 计算标准化参数
    processor.fit_normalizer(df)
    
    # 标准化特征
    df = processor.normalize_features(df)
    
    # 创建滑动窗口序列
    data_outputs = processor.create_sliding_window_sequences(df)
    
    # 清理临时文件
    os.remove(temp_file)
    
    return data_outputs + (processor,)


def train_sliding_window_model():
    """训练滑动窗口模型"""
    print("🚀 开始训练滑动窗口金融量化模型...")
    
    # 1. 创建数据
    features_list, price_targets_list, position_targets, next_day_returns, processor = create_sliding_window_dataset()
    
    print(f"📊 数据形状:")
    print(f"  - 特征序列: {features_list.shape}")  # [n_sequences, 20, 180, 11]
    print(f"  - 价格目标: {price_targets_list.shape}")  # [n_sequences, 20, 7]
    print(f"  - 仓位目标: {position_targets.shape}")  # [n_sequences, 20]
    print(f"  - 次日收益: {next_day_returns.shape}")  # [n_sequences, 20]
    
    # 2. 重新组织数据用于训练
    # 将 [n_sequences, 20, ...] 展平为 [n_sequences*20, ...]
    n_sequences, n_slides = features_list.shape[:2]
    
    # 展平特征数据
    features_flat = features_list.view(-1, features_list.shape[2], features_list.shape[3])  # [n_sequences*20, 180, 11]
    price_targets_flat = price_targets_list.view(-1, price_targets_list.shape[2])  # [n_sequences*20, 7]
    next_day_returns_flat = next_day_returns.view(-1)  # [n_sequences*20]
    
    print(f"📊 展平后数据形状:")
    print(f"  - 特征: {features_flat.shape}")
    print(f"  - 价格目标: {price_targets_flat.shape}")
    print(f"  - 次日收益: {next_day_returns_flat.shape}")
    
    # 3. 数据分割
    train_size = int(0.8 * len(features_flat))
    train_features = features_flat[:train_size]
    train_price_targets = price_targets_flat[:train_size]
    train_returns = next_day_returns_flat[:train_size]
    
    val_features = features_flat[train_size:]
    val_price_targets = price_targets_flat[train_size:]
    val_returns = next_day_returns_flat[train_size:]
    
    print(f"📊 数据分割:")
    print(f"  - 训练集: {len(train_features)} 样本")
    print(f"  - 验证集: {len(val_features)} 样本")
    
    # 4. 创建数据加载器
    train_dataset = TensorDataset(train_features, train_price_targets, train_returns)
    val_dataset = TensorDataset(val_features, val_price_targets, val_returns)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 5. 创建模型
    config = ModelConfigs.tiny()  # 使用小模型进行测试
    model = FinancialTransformer(config)
    
    print(f"🏗️ 模型创建完成:")
    print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 输入特征: {config.n_features}")
    print(f"  - 价格预测: {config.n_predictions}")
    print(f"  - 仓位预测: 启用")
    
    # 6. 设置优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    
    # 7. 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"🔧 使用设备: {device}")
    print("📈 开始训练...")
    
    num_epochs = 30
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_price_loss = 0.0
        train_position_loss = 0.0
        train_batches = 0
        
        for batch_features, batch_price_targets, batch_returns in train_loader:
            batch_features = batch_features.to(device)
            batch_price_targets = batch_price_targets.to(device)
            batch_returns = batch_returns.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                financial_data=batch_features,
                target_prices=batch_price_targets,
                next_day_returns=batch_returns,
                return_dict=True
            )
            
            loss = outputs['loss']
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            if outputs.get('price_loss') is not None:
                train_price_loss += outputs['price_loss'].item()
            if outputs.get('position_loss') is not None:
                train_position_loss += outputs['position_loss'].item()
            train_batches += 1
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_price_loss = 0.0
        val_position_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_price_targets, batch_returns in val_loader:
                batch_features = batch_features.to(device)
                batch_price_targets = batch_price_targets.to(device)
                batch_returns = batch_returns.to(device)
                
                outputs = model(
                    financial_data=batch_features,
                    target_prices=batch_price_targets,
                    next_day_returns=batch_returns,
                    return_dict=True
                )
                
                val_loss += outputs['loss'].item()
                if outputs.get('price_loss') is not None:
                    val_price_loss += outputs['price_loss'].item()
                if outputs.get('position_loss') is not None:
                    val_position_loss += outputs['position_loss'].item()
                val_batches += 1
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        avg_train_price_loss = train_price_loss / train_batches if train_batches > 0 else 0
        avg_train_position_loss = train_position_loss / train_batches if train_batches > 0 else 0
        avg_val_price_loss = val_price_loss / val_batches if val_batches > 0 else 0
        avg_val_position_loss = val_position_loss / val_batches if val_batches > 0 else 0
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_sliding_window_model.pth')
        
        # 打印进度
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {avg_train_loss:.6f}/{avg_val_loss:.6f} | "
              f"Price: {avg_train_price_loss:.6f}/{avg_val_price_loss:.6f} | "
              f"Position: {avg_train_position_loss:.6f}/{avg_val_position_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ 训练完成!")
    
    # 8. 测试预测
    print("\n🔮 测试滑动窗口预测功能...")
    model.eval()
    
    # 使用验证集的第一个样本进行预测
    test_features = val_features[:1].to(device)
    test_price_targets = val_price_targets[:1].to(device)
    test_returns = val_returns[:1].to(device)
    
    with torch.no_grad():
        predictions = model.predict(test_features, return_dict=True)
        
        # 价格预测
        predicted_prices = predictions['price_predictions']
        predicted_prices_denorm = processor.denormalize_predictions(predicted_prices)
        actual_prices_denorm = processor.denormalize_predictions(test_price_targets)
        
        print(f"📊 价格预测结果:")
        print(f"  实际价格: {actual_prices_denorm[0].cpu().numpy()}")
        print(f"  预测价格: {predicted_prices_denorm[0].cpu().numpy()}")
        
        # 仓位预测
        if 'position_predictions' in predictions:
            position_pred = predictions['position_predictions']
            print(f"\n📈 仓位预测结果:")
            print(f"  预测仓位: {position_pred[0].item():.2f} (0-10)")
            print(f"  次日收益: {test_returns[0].item():.4f} ({test_returns[0].item()*100:.2f}%)")
            
            # 计算该仓位的收益
            position_return = position_pred[0].item() * test_returns[0].item()
            print(f"  仓位收益: {position_return:.4f} ({position_return*100:.2f}%)")


if __name__ == "__main__":
    train_sliding_window_model()

#!/usr/bin/env python3
"""
金融量化模型训练脚本
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

def create_sample_dataset() -> Tuple[torch.Tensor, torch.Tensor]:
    """创建示例数据集"""
    print("🔄 创建示例金融数据...")
    
    # 生成示例数据
    sample_data = create_sample_data(n_days=300)
    
    # 保存到临时文件
    temp_file = "temp_financial_data.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    # 处理数据
    processor = FinancialDataProcessor(
        sequence_length=60,
        prediction_horizon=7,
        normalize=True
    )
    
    features, targets = processor.process_file(temp_file)
    
    # 清理临时文件
    os.remove(temp_file)
    
    return features, targets, processor

def train_model():
    """训练金融预测模型"""
    print("🚀 开始训练金融量化模型...")
    
    # 1. 创建数据
    features, targets, processor = create_sample_dataset()
    
    # 2. 数据分割
    train_size = int(0.8 * len(features))
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    val_features = features[train_size:]
    val_targets = targets[train_size:]
    
    print(f"📊 数据分割:")
    print(f"  - 训练集: {len(train_features)} 样本")
    print(f"  - 验证集: {len(val_features)} 样本")
    
    # 3. 创建数据加载器
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 4. 创建模型
    config = ModelConfigs.tiny()  # 使用小模型进行测试
    model = FinancialTransformer(config)
    
    print(f"🏗️ 模型创建完成:")
    print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 输入特征: {config.n_features}")
    print(f"  - 预测数量: {config.n_predictions}")
    
    # 5. 设置优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    
    # 6. 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"🔧 使用设备: {device}")
    print("📈 开始训练...")
    
    num_epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                financial_data=batch_features,
                target_prices=batch_targets,
                return_dict=True
            )
            
            loss = outputs['loss']
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(
                    financial_data=batch_features,
                    target_prices=batch_targets,
                    return_dict=True
                )
                
                val_loss += outputs['loss'].item()
                val_batches += 1
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_financial_model.pth')
        
        # 打印进度
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ 训练完成!")
    
    # 7. 测试预测
    print("\n🔮 测试预测功能...")
    model.eval()
    
    # 使用验证集的第一个样本进行预测
    test_features = val_features[:1].to(device)  # 取一个样本
    test_targets = val_targets[:1].to(device)
    
    with torch.no_grad():
        predictions = model.predict(test_features, return_dict=True)
        predicted_prices = predictions['predictions']
        
        # 反标准化预测结果
        predicted_prices = processor.denormalize_predictions(predicted_prices)
        actual_prices = processor.denormalize_predictions(test_targets)
        
        print(f"📊 预测结果对比:")
        print(f"  实际价格: {actual_prices[0].cpu().numpy()}")
        print(f"  预测价格: {predicted_prices[0].cpu().numpy()}")
        
        # 计算误差
        mae = torch.mean(torch.abs(predicted_prices - actual_prices)).item()
        mse = torch.mean((predicted_prices - actual_prices) ** 2).item()
        
        print(f"  平均绝对误差 (MAE): {mae:.4f}")
        print(f"  均方误差 (MSE): {mse:.4f}")

def main():
    """主函数"""
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

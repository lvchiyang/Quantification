"""
价格预测网络独立训练脚本
专门训练Transformer架构的价格预测模型
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from price_prediction.config import PricePredictionConfigs
from price_prediction.data_processor import PricePredictionDataProcessor
from price_prediction.price_transformer import PriceTransformer

def create_dataloader(features, targets, batch_size, shuffle=True):
    """创建数据加载器"""
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_price_prediction_model(config):
    """训练价格预测模型"""
    print("🚀 开始训练价格预测模型")
    print("="*50)
    
    # 1. 创建数据处理器
    print("📊 加载数据...")
    data_processor = PricePredictionDataProcessor(
        data_dir=config.data_dir,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
        large_value_transform=config.large_value_transform
    )
    
    # 2. 加载所有股票数据
    stock_data = data_processor.load_all_stocks_for_price_prediction()
    
    if not stock_data:
        print("❌ 没有加载到任何数据")
        return None
    
    # 3. 合并所有股票数据
    all_features = []
    all_targets = []
    
    for stock_name, (features, targets) in stock_data.items():
        all_features.append(features)
        all_targets.append(targets)
        print(f"  {stock_name}: {features.shape[0]} 个序列")
    
    # 合并数据
    train_features = torch.cat(all_features, dim=0)
    train_targets = torch.cat(all_targets, dim=0)
    
    print(f"\n📈 训练数据统计:")
    print(f"  特征形状: {train_features.shape}")
    print(f"  目标形状: {train_targets.shape}")
    print(f"  总序列数: {len(train_features)}")
    
    # 4. 创建数据加载器
    train_loader = create_dataloader(train_features, train_targets, config.batch_size)
    
    # 5. 创建模型
    print(f"\n🏗️  创建模型...")
    model = PriceTransformer(config)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 6. 设置优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    if config.loss_type == "mse":
        criterion = nn.MSELoss()
    elif config.loss_type == "mae":
        criterion = nn.L1Loss()
    elif config.loss_type == "huber":
        criterion = nn.HuberLoss(delta=config.huber_delta)
    else:
        criterion = nn.MSELoss()
    
    # 7. 训练循环
    print(f"\n🎯 开始训练...")
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.max_epochs):
        model.train()
        epoch_losses = []
        
        # 训练一个epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        for batch_idx, (features, targets) in enumerate(pbar):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features)
            price_predictions = outputs['price_predictions']
            
            # 计算损失
            loss = criterion(price_predictions, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}: 平均损失 = {avg_loss:.6f}")
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # 保存最佳模型
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, os.path.join(config.save_dir, 'best_price_model.pth'))
            
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 定期保存
        if (epoch + 1) % config.save_every_n_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(config.save_dir, f'price_model_epoch_{epoch+1}.pth'))
    
    # 8. 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('价格预测模型训练曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.save_dir, 'training_curve.png'))
    plt.show()
    
    print(f"\n✅ 训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型保存在: {config.save_dir}")
    
    return model, train_losses

def main():
    """主函数"""
    # 选择配置
    print("选择模型配置:")
    print("1. tiny - 轻量级测试")
    print("2. small - 小型模型")
    print("3. base - 基础模型（推荐）")
    print("4. large - 大型模型")
    print("5. long_sequence - 长序列专用")
    
    choice = input("请选择配置 (1-5, 默认3): ").strip() or "3"
    
    config_map = {
        "1": PricePredictionConfigs.tiny(),
        "2": PricePredictionConfigs.small(),
        "3": PricePredictionConfigs.base(),
        "4": PricePredictionConfigs.large(),
        "5": PricePredictionConfigs.for_long_sequence()
    }
    
    config = config_map.get(choice, PricePredictionConfigs.base())
    
    # 开始训练
    try:
        model, losses = train_price_prediction_model(config)
        
        if model is not None:
            print("\n🎉 价格预测模型训练成功!")
            print("现在可以使用训练好的模型进行价格预测了。")
            
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

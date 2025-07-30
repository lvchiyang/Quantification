"""
策略网络独立训练脚本
专门训练GRU架构的交易策略模型
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

from strategy_network.config import StrategyNetworkConfigs
from strategy_network.data_processor import StrategyNetworkDataProcessor
from strategy_network.gru_strategy import GRUStrategyNetwork
from strategy_network.strategy_loss import StrategyLoss

def create_strategy_dataloader(features, positions, returns, batch_size, shuffle=True):
    """创建策略数据加载器"""
    dataset = TensorDataset(features, positions, returns)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_strategy_network_model(config):
    """训练策略网络模型"""
    print("🧠 开始训练策略网络模型")
    print("="*50)
    
    # 1. 创建数据处理器
    print("📊 加载数据...")
    data_processor = StrategyNetworkDataProcessor(
        data_dir=config.data_dir,
        trading_horizon=config.trading_horizon,
        feature_extraction_length=config.feature_extraction_length,
        large_value_transform=config.large_value_transform
    )
    
    # 2. 加载所有股票数据
    stock_data = data_processor.load_all_stocks_for_strategy()
    
    if not stock_data:
        print("❌ 没有加载到任何数据")
        return None
    
    # 3. 合并所有股票数据
    all_features = []
    all_positions = []
    all_returns = []
    
    for stock_name, (features, positions, returns) in stock_data.items():
        all_features.append(features)
        all_positions.append(positions)
        all_returns.append(returns)
        print(f"  {stock_name}: {features.shape[0]} 个序列")
    
    # 合并数据
    train_features = torch.cat(all_features, dim=0)
    train_positions = torch.cat(all_positions, dim=0)
    train_returns = torch.cat(all_returns, dim=0)
    
    print(f"\n📈 训练数据统计:")
    print(f"  策略特征形状: {train_features.shape}")
    print(f"  仓位目标形状: {train_positions.shape}")
    print(f"  收益率形状: {train_returns.shape}")
    print(f"  总序列数: {len(train_features)}")
    
    # 4. 创建数据加载器
    train_loader = create_strategy_dataloader(
        train_features, train_positions, train_returns, config.batch_size
    )
    
    # 5. 创建模型
    print(f"\n🏗️  创建策略网络...")
    model = GRUStrategyNetwork(config)
    
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
    
    # 策略损失函数
    criterion = StrategyLoss(config)
    
    # 7. 训练循环
    print(f"\n🎯 开始训练...")
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.max_epochs):
        model.train()
        epoch_losses = []
        epoch_returns = []
        
        # 训练一个epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        for batch_idx, (features, positions, returns) in enumerate(pbar):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model.forward_sequence(features)
            position_predictions = outputs['position_output']['positions']
            
            # 计算策略损失
            loss_dict = criterion(position_predictions, returns)
            total_loss = loss_dict['loss_tensor']
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失和收益
            epoch_losses.append(total_loss.item())
            if 'cumulative_return' in loss_dict:
                epoch_returns.append(loss_dict['cumulative_return'])
            
            # 更新Gumbel温度
            if hasattr(model, 'update_temperature'):
                model.update_temperature(config.temperature_decay, config.min_temperature)
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.6f}',
                'return': f'{loss_dict.get("cumulative_return", 0):.4f}'
            })
        
        # 计算平均损失和收益
        avg_loss = np.mean(epoch_losses)
        avg_return = np.mean(epoch_returns) if epoch_returns else 0
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}: 损失 = {avg_loss:.6f}, 平均收益 = {avg_return:.4f}")
        
        # 早停检查（基于损失，但策略网络可能需要更复杂的评估）
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
                'avg_return': avg_return,
                'config': config
            }, os.path.join(config.save_dir, 'best_strategy_model.pth'))
            
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
                'avg_return': avg_return,
                'config': config
            }, os.path.join(config.save_dir, f'strategy_model_epoch_{epoch+1}.pth'))
    
    # 8. 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('策略网络训练损失')
    plt.legend()
    plt.grid(True)
    
    if epoch_returns:
        plt.subplot(1, 2, 2)
        plt.plot(epoch_returns, label='平均收益', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Return')
        plt.title('策略网络收益表现')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, 'training_curve.png'))
    plt.show()
    
    print(f"\n✅ 训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型保存在: {config.save_dir}")
    
    return model, train_losses

def main():
    """主函数"""
    # 选择配置
    print("选择策略网络配置:")
    print("1. tiny - 轻量级测试")
    print("2. small - 小型模型")
    print("3. base - 基础模型（推荐）")
    print("4. large - 大型模型")
    print("5. conservative - 保守策略")
    print("6. aggressive - 激进策略")
    print("7. high_frequency - 高频交易")
    print("8. long_term - 长期投资")
    
    choice = input("请选择配置 (1-8, 默认3): ").strip() or "3"
    
    config_map = {
        "1": StrategyNetworkConfigs.tiny(),
        "2": StrategyNetworkConfigs.small(),
        "3": StrategyNetworkConfigs.base(),
        "4": StrategyNetworkConfigs.large(),
        "5": StrategyNetworkConfigs.conservative(),
        "6": StrategyNetworkConfigs.aggressive(),
        "7": StrategyNetworkConfigs.high_frequency(),
        "8": StrategyNetworkConfigs.long_term()
    }
    
    config = config_map.get(choice, StrategyNetworkConfigs.base())
    
    # 开始训练
    try:
        model, losses = train_strategy_network_model(config)
        
        if model is not None:
            print("\n🎉 策略网络训练成功!")
            print("现在可以使用训练好的模型进行交易策略决策了。")
            
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

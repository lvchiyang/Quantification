"""
价格预测网络独立训练脚本
专门训练Transformer架构的价格预测模型
"""

import sys
import os
import glob
from datetime import datetime
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from price_prediction.config import PricePredictionConfigs
from price_prediction.price_transformer import PriceTransformer
from price_prediction.financial_losses import FinancialMultiLoss
# 使用序列处理器
sys.path.append('.')
from sequence_processor import PriceDataset


def find_latest_data_dir():
    """查找最新的数据目录"""
    pattern = "processed_data_*"
    data_dirs = glob.glob(pattern)

    if not data_dirs:
        # 如果没找到，尝试创建今天的目录名
        today = datetime.now().strftime("%Y-%m-%d")
        return f"processed_data_{today}"

    # 返回最新的目录
    return sorted(data_dirs)[-1]



def train_price_prediction_model(config):
    """训练价格预测模型"""
    print("🚀 开始训练价格预测模型")
    print("="*50)

    # 1. 查找数据目录
    print("📊 查找数据目录...")
    data_dir = find_latest_data_dir()
    data_path = os.path.join(data_dir, "股票数据")

    print(f"使用数据目录: {data_dir}")

    if not os.path.exists(data_path):
        print(f"❌ 数据目录不存在: {data_path}")
        print("请确保数据已经处理完成！")
        return None

    # 2. 使用序列处理器创建数据集
    dataset = PriceDataset(data_path, sequence_length=config.sequence_length)

    if len(dataset) == 0:
        print("❌ 没有加载到任何数据")
        return None

    print(f"\n📈 训练数据统计:")
    print(f"  总序列数: {len(dataset)}")

    # 检查数据形状
    sample_input, sample_target = dataset[0]
    print(f"  输入形状: {sample_input.shape}")  # [180, 20]
    print(f"  目标形状: {sample_target.shape}")  # [10]

    # 3. 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
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

    # 根据配置选择损失函数
    if config.use_financial_loss:
        # 使用金融专用多损失函数组合
        print(f"🎯 使用金融专用多损失函数:")
        print(f"  基础损失类型: {config.loss_type}")

        criterion = FinancialMultiLoss(
            base_loss_type=config.loss_type,
            use_direction_loss=config.use_direction_loss,
            use_trend_loss=config.use_trend_loss,
            use_temporal_weighting=config.use_temporal_weighting,
            use_ranking_loss=config.use_ranking_loss,
            use_volatility_loss=config.use_volatility_loss,

            # 损失权重配置
            base_weight=config.base_weight,
            direction_weight=config.direction_weight,
            trend_weight=config.trend_weight,
            ranking_weight=config.ranking_weight,
            volatility_weight=config.volatility_weight
        )

        enabled_components = []
        if config.use_direction_loss:
            enabled_components.append(f"方向损失({config.direction_weight})")
        if config.use_trend_loss:
            enabled_components.append(f"趋势损失({config.trend_weight})")
        if config.use_temporal_weighting:
            enabled_components.append("时间加权")
        if config.use_ranking_loss:
            enabled_components.append(f"排序损失({config.ranking_weight})")
        if config.use_volatility_loss:
            enabled_components.append(f"波动率损失({config.volatility_weight})")

        print(f"  启用组件: {' + '.join(enabled_components)}")
        use_multi_loss = True

    else:
        # 使用传统单一损失函数
        print(f"🎯 使用传统损失函数: {config.loss_type}")

        if config.loss_type == "mse":
            criterion = nn.MSELoss()
        elif config.loss_type == "mae":
            criterion = nn.L1Loss()
        elif config.loss_type == "huber":
            criterion = nn.HuberLoss(delta=config.huber_delta)
        else:
            criterion = nn.MSELoss()

        use_multi_loss = False
    
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
        for features, targets in pbar:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features)
            price_predictions = outputs['price_predictions']
            
            # 计算损失
            if use_multi_loss:
                # 金融专用多损失函数
                loss_dict = criterion(price_predictions, targets)
                total_loss = loss_dict['loss']  # 主损失用于反向传播

                # 显示详细损失信息
                pbar.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'base': f'{loss_dict["base_loss"]:.4f}',
                    'dir': f'{loss_dict["direction_loss"]:.4f}',
                    'trend': f'{loss_dict["trend_loss"]:.4f}',
                    'dir_acc': f'{loss_dict["direction_accuracy"]:.2%}'
                })
            else:
                # 传统单一损失函数
                total_loss = criterion(price_predictions, targets)

                # 显示简单损失信息
                pbar.set_postfix({'loss': f'{total_loss.item():.6f}'})

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 记录损失
            epoch_losses.append(total_loss.item())
        
        # 计算平均损失
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}: 平均总损失 = {avg_loss:.6f}")
        
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
        result = train_price_prediction_model(config)

        if result is not None:
            model, train_losses = result
            print("\n🎉 价格预测模型训练成功!")
            print("现在可以使用训练好的模型进行价格预测了。")
        else:
            print("\n❌ 训练失败，请检查数据和配置")
            
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

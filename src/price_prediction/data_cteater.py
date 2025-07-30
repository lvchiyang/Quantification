# -*- coding: utf-8 -*-
"""
序列处理器 - 最终版
用于创建训练和预测序列，避免数据泄露
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
import os
from typing import Tuple, List, Optional


class SequenceProcessor:
    """
    序列级数据处理器
    核心：每个180天序列独立计算基准，避免数据泄露
    """
    
    def __init__(self, sequence_length: int = 180):
        self.sequence_length = sequence_length
        self.feature_columns = [
            '月', '日', '星期',                           # 时间特征 (3维)
            'open_rel', 'high_rel', 'low_rel', 'close_rel',  # 价格特征 (4维)
            '涨幅', '振幅',                               # 价格变化 (2维)
            'volume_rel', 'volume_change',                # 成交量特征 (2维)
            'amount_rel', 'amount_change',                # 金额特征 (2维)
            '换手%', '成交次数',                          # 市场活跃度 (2维)
            'big_order_activity', 'chip_concentration',   # 金融特征1,2 (2维)
            'market_sentiment', 'price_volume_sync'       # 金融特征3,4 (2维)
        ]  # 总计20维特征
        
    def process_sequence_features(self, sequence_df: pd.DataFrame) -> dict:
        """
        对单个180天序列进行特征工程
        关键：使用序列内数据计算基准，避免数据泄露
        """
        # 1. 序列级价格特征处理
        ohlc_data = sequence_df[['开盘', '最高', '最低', '收盘']].copy()
        for col in ['开盘', '最高', '最低', '收盘']:
            ohlc_data[col] = pd.to_numeric(ohlc_data[col], errors='coerce')
        
        # 计算序列内所有OHLC的中位数作为价格基准
        all_prices = ohlc_data.values.flatten()
        all_prices = all_prices[~pd.isna(all_prices)]
        sequence_price_median = np.median(all_prices)
        
        # OHLC相对于序列基准的比值
        open_rel = (sequence_df['开盘'] / sequence_price_median).fillna(1.0)
        high_rel = (sequence_df['最高'] / sequence_price_median).fillna(1.0)
        low_rel = (sequence_df['最低'] / sequence_price_median).fillna(1.0)
        close_rel = (sequence_df['收盘'] / sequence_price_median).fillna(1.0)
        
        # 2. 序列级成交量/金额处理
        def process_volume_amount(col_name):
            values = pd.to_numeric(sequence_df[col_name], errors='coerce').fillna(0)
            
            # 序列内中位数基准
            sequence_median = values.median()
            if sequence_median == 0:
                sequence_median = 1.0
            relative_values = values / sequence_median
            
            # 相对变化率（20日滚动均值）
            rolling_mean = values.rolling(window=20, min_periods=1).mean()
            rolling_mean = rolling_mean.fillna(method='bfill').fillna(1.0).replace(0, 1.0)
            relative_change = ((values - rolling_mean) / rolling_mean * 100).fillna(0.0)
            
            return relative_values, relative_change
        
        volume_rel, volume_change = process_volume_amount('总手')
        amount_rel, amount_change = process_volume_amount('金额')
        
        # 3. 序列级金融特征处理
        def standardize_without_clipping(series):
            mean = series.mean()
            std = series.std() + 1e-6
            return (series - mean) / std
        
        # 确保数值类型
        total_volume = pd.to_numeric(sequence_df['总手'], errors='coerce').fillna(0)
        trade_count = pd.to_numeric(sequence_df['成交次数'], errors='coerce').fillna(1)
        turnover_rate = pd.to_numeric(sequence_df['换手%'], errors='coerce').fillna(0)
        price_change = pd.to_numeric(sequence_df['涨幅'], errors='coerce').fillna(0)
        amplitude = pd.to_numeric(sequence_df['振幅'], errors='coerce').fillna(0)
        
        # 金融特征计算
        big_order_activity = total_volume / (trade_count + 1e-6)
        
        volume_mean = total_volume.rolling(30, min_periods=1).mean().fillna(method='bfill').fillna(1.0)
        volume_normalized = total_volume / (volume_mean + 1e-6)
        chip_concentration = turnover_rate / (volume_normalized + 1e-6)
        
        market_sentiment = price_change * amplitude / 100
        
        price_direction = np.sign(price_change)
        volume_change_pct = total_volume.pct_change().fillna(0)
        volume_direction = np.sign(volume_change_pct)
        price_volume_sync = price_direction * volume_direction
        
        # 标准化但不裁剪
        big_order_activity = standardize_without_clipping(big_order_activity)
        chip_concentration = standardize_without_clipping(chip_concentration)
        market_sentiment = standardize_without_clipping(market_sentiment)
        
        return {
            'open_rel': open_rel,
            'high_rel': high_rel,
            'low_rel': low_rel,
            'close_rel': close_rel,
            'volume_rel': volume_rel,
            'volume_change': volume_change,
            'amount_rel': amount_rel,
            'amount_change': amount_change,
            'big_order_activity': big_order_activity,
            'chip_concentration': chip_concentration,
            'market_sentiment': market_sentiment,
            'price_volume_sync': price_volume_sync
        }
    
    def build_feature_vector(self, sequence_df: pd.DataFrame, computed_features: dict) -> np.ndarray:
        """
        构建20维特征向量
        """
        # 基础特征（来自原始数据）
        result_df = sequence_df[['月', '日', '星期', '涨幅', '振幅', '换手%', '成交次数']].copy()
        
        # 添加计算特征
        for feature_name, feature_values in computed_features.items():
            result_df[feature_name] = feature_values
        
        # 按照指定顺序选择特征
        return result_df[self.feature_columns].values  # [180, 20]
    
    def create_training_sequences(self, cleaned_data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        从清洗后的数据创建训练序列
        
        Args:
            cleaned_data: 基础清洗后的数据（14列）
            
        Returns:
            List of (input_sequence, target_prices)
            input_sequence: [180, 20] 特征矩阵
            target_prices: [10] 未来10个时间点的close_rel
        """
        sequences = []
        target_days = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]  # 未来第N天
        max_target_day = max(target_days)
        
        for i in range(len(cleaned_data) - self.sequence_length - max_target_day + 1):
            # 提取180天序列
            sequence = cleaned_data.iloc[i:i+self.sequence_length].copy()
            
            # 序列级特征工程（避免数据泄露）
            computed_features = self.process_sequence_features(sequence)
            
            # 构建特征向量
            feature_vector = self.build_feature_vector(sequence, computed_features)
            
            # 提取目标价格（未来指定天数的收盘价相对值）
            target_prices = []
            for day in target_days:
                target_idx = i + self.sequence_length + day - 1
                if target_idx < len(cleaned_data):
                    # 计算目标序列的价格基准（用于计算target的close_rel）
                    target_sequence = cleaned_data.iloc[i+day:i+day+self.sequence_length]
                    if len(target_sequence) >= self.sequence_length:
                        target_features = self.process_sequence_features(target_sequence)
                        target_close_rel = target_features['close_rel'].iloc[-1]  # 最后一天的close_rel
                        target_prices.append(target_close_rel)
                    else:
                        target_prices.append(1.0)  # 默认值
                else:
                    target_prices.append(1.0)  # 默认值
            
            sequences.append((feature_vector, np.array(target_prices)))
        
        return sequences
    
    def create_prediction_sequence(self, recent_data: pd.DataFrame) -> np.ndarray:
        """
        创建预测序列（最后180天）
        
        Args:
            recent_data: 最近的数据（至少180行）
            
        Returns:
            feature_vector: [180, 20] 特征矩阵
        """
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"数据长度不足，需要至少{self.sequence_length}天数据")
        
        # 取最后180天
        sequence = recent_data.iloc[-self.sequence_length:].copy()
        
        # 序列级特征工程
        computed_features = self.process_sequence_features(sequence)
        
        # 构建特征向量
        feature_vector = self.build_feature_vector(sequence, computed_features)
        
        return feature_vector  # [180, 20]


class PriceDataset(Dataset):
    """
    价格预测数据集
    """
    
    def __init__(self, data_dir: str, sequence_length: int = 180):
        self.processor = SequenceProcessor(sequence_length)
        self.data = []
        
        print(f"加载数据目录: {data_dir}")
        
        # 加载所有清洗后的数据文件
        for data_file in glob.glob(f"{data_dir}/**/*.xlsx", recursive=True):
            try:
                df = pd.read_excel(data_file)
                print(f"处理文件: {os.path.basename(data_file)}, 数据长度: {len(df)}")
                
                # 创建训练序列
                sequences = self.processor.create_training_sequences(df)
                self.data.extend(sequences)
                
                print(f"  生成序列数: {len(sequences)}")
                
            except Exception as e:
                print(f"处理文件失败: {data_file}, 错误: {e}")
        
        print(f"总序列数: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_prices = self.data[idx]
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_prices)


def predict_stock_price(model, stock_file: str, processor: SequenceProcessor = None):
    """
    对单只股票进行价格预测
    
    Args:
        model: 训练好的模型
        stock_file: 股票数据文件路径
        processor: 序列处理器
        
    Returns:
        predictions: [10] 未来10个时间点的预测值
    """
    if processor is None:
        processor = SequenceProcessor()
    
    # 加载数据
    df = pd.read_excel(stock_file)
    
    # 创建预测序列
    input_seq = processor.create_prediction_sequence(df)
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)  # [1, 180, 20]
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, dict):
            predictions = outputs['price_predictions']  # [1, 10]
        else:
            predictions = outputs  # [1, 10]
    
    return predictions.squeeze().cpu().numpy()  # [10]


def validate_sequence_processing(cleaned_data: pd.DataFrame, processor: SequenceProcessor = None):
    """
    验证序列处理是否避免了数据泄露
    """
    if processor is None:
        processor = SequenceProcessor()

    print("🔍 验证序列处理...")

    # 检查不同序列的价格基准
    if len(cleaned_data) >= 280:
        sequence1 = cleaned_data.iloc[0:180]
        sequence2 = cleaned_data.iloc[100:280]

        features1 = processor.process_sequence_features(sequence1)
        features2 = processor.process_sequence_features(sequence2)

        # 计算价格基准
        median1 = np.median(sequence1[['开盘', '最高', '最低', '收盘']].values.flatten())
        median2 = np.median(sequence2[['开盘', '最高', '最低', '收盘']].values.flatten())

        print(f"序列1价格基准: {median1:.2f}")
        print(f"序列2价格基准: {median2:.2f}")
        print(f"基准差异: {abs(median1 - median2):.2f}")

        # 检查特征范围（不应该被裁剪）
        print(f"序列1大单活跃度范围: [{features1['big_order_activity'].min():.3f}, {features1['big_order_activity'].max():.3f}]")
        print(f"序列2大单活跃度范围: [{features2['big_order_activity'].min():.3f}, {features2['big_order_activity'].max():.3f}]")

        # 检查是否有NaN值
        feature_vector1 = processor.build_feature_vector(sequence1, features1)
        feature_vector2 = processor.build_feature_vector(sequence2, features2)

        has_nan1 = np.isnan(feature_vector1).any()
        has_nan2 = np.isnan(feature_vector2).any()

        print(f"序列1包含NaN: {has_nan1}")
        print(f"序列2包含NaN: {has_nan2}")

        if not has_nan1 and not has_nan2:
            print("✅ 验证通过：无数据泄露，无NaN值")
        else:
            print("❌ 验证失败：存在NaN值")
    else:
        print("❌ 数据长度不足，无法验证")


def check_data_quality(sequences: List[Tuple[np.ndarray, np.ndarray]]):
    """
    检查处理后数据的质量
    """
    print("📊 数据质量检查...")
    print(f"序列数量: {len(sequences)}")

    if len(sequences) > 0:
        input_seq, target_seq = sequences[0]
        print(f"输入序列形状: {input_seq.shape}")  # 应该是 (180, 20)
        print(f"目标序列形状: {target_seq.shape}")  # 应该是 (10,)

        # 检查是否有NaN值
        all_inputs = np.array([seq[0] for seq in sequences])
        all_targets = np.array([seq[1] for seq in sequences])

        has_nan_inputs = np.isnan(all_inputs).any()
        has_nan_targets = np.isnan(all_targets).any()

        print(f"输入包含NaN值: {has_nan_inputs}")
        print(f"目标包含NaN值: {has_nan_targets}")

        # 检查数值范围
        feature_names = [
            '月', '日', '星期', 'open_rel', 'high_rel', 'low_rel', 'close_rel',
            '涨幅', '振幅', 'volume_rel', 'volume_change', 'amount_rel', 'amount_change',
            '换手%', '成交次数', 'big_order_activity', 'chip_concentration',
            'market_sentiment', 'price_volume_sync'
        ]

        print("\n特征数值范围:")
        for i, feature_name in enumerate(feature_names):
            feature_values = all_inputs[:, :, i].flatten()
            print(f"  {feature_name}: [{feature_values.min():.3f}, {feature_values.max():.3f}]")

        print(f"\n目标值范围: [{all_targets.min():.3f}, {all_targets.max():.3f}]")


# 使用示例和测试
if __name__ == "__main__":
    print("🚀 序列处理器测试")

    # 测试单个文件
    test_file = "processed_data_2025-07-30/股票数据/白酒/茅台.xlsx"
    if os.path.exists(test_file):
        print(f"\n📁 测试文件: {test_file}")

        # 加载数据
        df = pd.read_excel(test_file)
        print(f"数据形状: {df.shape}")

        # 创建处理器
        processor = SequenceProcessor()

        # 验证序列处理
        validate_sequence_processing(df, processor)

        # 创建训练序列
        print(f"\n🔄 创建训练序列...")
        sequences = processor.create_training_sequences(df)
        print(f"生成序列数: {len(sequences)}")

        # 检查数据质量
        if len(sequences) > 0:
            check_data_quality(sequences[:100])  # 检查前100个序列

        # 测试预测序列创建
        print(f"\n🔮 测试预测序列创建...")
        try:
            pred_seq = processor.create_prediction_sequence(df)
            print(f"预测序列形状: {pred_seq.shape}")
            print("✅ 预测序列创建成功")
        except Exception as e:
            print(f"❌ 预测序列创建失败: {e}")

    else:
        print(f"❌ 测试文件不存在: {test_file}")

        # 创建完整数据集测试
        data_dir = "processed_data_2025-07-30/股票数据"
        if os.path.exists(data_dir):
            print(f"\n📂 测试数据目录: {data_dir}")
            dataset = PriceDataset(data_dir)

            if len(dataset) > 0:
                # 创建数据加载器
                from torch.utils.data import DataLoader
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                # 检查数据
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    print(f"Batch {batch_idx}: inputs {inputs.shape}, targets {targets.shape}")
                    print(f"  输入范围: [{inputs.min():.3f}, {inputs.max():.3f}]")
                    print(f"  目标范围: [{targets.min():.3f}, {targets.max():.3f}]")
                    if batch_idx == 0:
                        break
            else:
                print("❌ 数据集为空")
        else:
            print(f"❌ 数据目录不存在: {data_dir}")

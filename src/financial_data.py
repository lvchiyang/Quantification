"""
金融数据预处理模块
处理股票OHLC数据和技术指标
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from datetime import datetime
import warnings

class FinancialDataProcessor:
    """
    金融数据预处理器
    
    处理股票数据格式：
    时间, 开盘, 最高, 最低, 收盘, 涨幅, 振幅, 总手, 金额, 换手%, 成交次数
    """
    
    def __init__(self,
                 sequence_length: int = 180,  # 使用180天历史数据
                 prediction_horizon: int = 7,
                 trading_horizon: int = 20,   # 滑动20次预测
                 normalize: bool = True,
                 enable_trading_strategy: bool = True,
                 sliding_window: bool = True):  # 启用滑动窗口模式
        """
        初始化数据处理器

        Args:
            sequence_length: 输入序列长度（默认180天）
            prediction_horizon: 价格预测时间跨度（7天）
            trading_horizon: 交易策略时间跨度（20天滑动窗口）
            normalize: 是否标准化数据
            enable_trading_strategy: 是否启用交易策略学习
            sliding_window: 是否使用滑动窗口模式
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.trading_horizon = trading_horizon
        self.normalize = normalize
        self.enable_trading_strategy = enable_trading_strategy
        self.sliding_window = sliding_window
        
        # 特征列名
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'change_pct', 
            'amplitude', 'volume', 'amount', 'turnover_rate', 
            'trade_count', 'time_encoding'
        ]
        
        # 标准化参数
        self.feature_stats = {}
        self.price_stats = {}
        
    def parse_data_line(self, line: str) -> Dict:
        """
        解析单行数据
        
        Args:
            line: 数据行，格式如：
                  "2009-10-15,四	16.11	17.51	15.53	17.08	44.99%	16.81%	153,586,470	2,501,742,900	87.27	2867"
        
        Returns:
            解析后的数据字典
        """
        parts = line.strip().split('\t')
        
        # 解析日期
        date_str = parts[0].split(',')[0]  # 去掉星期信息
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # 解析数值数据
        try:
            open_price = float(parts[1])
            high_price = float(parts[2])
            low_price = float(parts[3])
            close_price = float(parts[4])
            
            # 处理百分比
            change_pct = float(parts[5].replace('%', ''))
            amplitude = float(parts[6].replace('%', ''))
            
            # 处理大数值（去掉逗号）
            volume = float(parts[7].replace(',', ''))
            amount = float(parts[8].replace(',', ''))
            
            turnover_rate = float(parts[9])
            trade_count = float(parts[10])
            
            return {
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'change_pct': change_pct,
                'amplitude': amplitude,
                'volume': volume,
                'amount': amount,
                'turnover_rate': turnover_rate,
                'trade_count': trade_count
            }
            
        except (ValueError, IndexError) as e:
            warnings.warn(f"解析数据行失败: {line}, 错误: {e}")
            return None
    
    def add_time_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间编码特征
        
        Args:
            df: 包含日期的DataFrame
            
        Returns:
            添加时间编码后的DataFrame
        """
        df = df.copy()
        
        # 简单的时间编码：使用天数作为序列编码
        df['time_encoding'] = (df['date'] - df['date'].min()).dt.days
        
        # 标准化时间编码
        df['time_encoding'] = df['time_encoding'] / df['time_encoding'].max()
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, ...]:
        """
        创建训练序列

        Args:
            df: 处理后的DataFrame

        Returns:
            如果启用交易策略: (features, price_targets, trading_prices)
            否则: (features, price_targets)
        """
        features = []
        price_targets = []
        trading_prices = []

        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        # 提取特征列
        feature_data = df[self.feature_columns].values
        close_prices = df['close'].values

        # 计算所需的最大长度
        if self.enable_trading_strategy:
            max_horizon = max(self.prediction_horizon, self.trading_horizon)
        else:
            max_horizon = self.prediction_horizon

        # 创建滑动窗口
        for i in range(len(df) - self.sequence_length - max_horizon + 1):
            # 输入特征序列
            seq_features = feature_data[i:i + self.sequence_length]

            # 目标价格（未来7个时间点的收盘价）
            target_prices = close_prices[
                i + self.sequence_length:
                i + self.sequence_length + self.prediction_horizon
            ]

            features.append(seq_features)
            price_targets.append(target_prices)

            # 如果启用交易策略，添加交易时间段的价格
            if self.enable_trading_strategy:
                trading_period_prices = close_prices[
                    i + self.sequence_length:
                    i + self.sequence_length + self.trading_horizon
                ]
                trading_prices.append(trading_period_prices)

        if self.enable_trading_strategy:
            return (
                torch.FloatTensor(features),
                torch.FloatTensor(price_targets),
                torch.FloatTensor(trading_prices)
            )
        else:
            return torch.FloatTensor(features), torch.FloatTensor(price_targets)

    def create_sliding_window_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, ...]:
        """
        创建滑动窗口训练序列

        对于每个200天的序列，创建20个滑动窗口：
        - 第1次：用第1-180天预测第181-187天价格，输出第181天仓位
        - 第2次：用第2-181天预测第182-188天价格，输出第182天仓位
        - ...
        - 第20次：用第20-199天预测第200-206天价格，输出第200天仓位

        Args:
            df: 处理后的DataFrame (至少200+7天数据)

        Returns:
            (features_list, price_targets_list, position_targets, next_day_returns)
        """
        features_list = []
        price_targets_list = []
        position_targets = []  # 每天的仓位决策 (0-10)
        next_day_returns = []  # 次日涨跌幅

        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        # 提取特征列和价格
        feature_data = df[self.feature_columns].values
        close_prices = df['close'].values

        # 计算涨跌幅
        price_changes = np.diff(close_prices) / close_prices[:-1]  # 涨跌幅

        # 需要至少 sequence_length + trading_horizon + prediction_horizon 天的数据
        min_required_days = self.sequence_length + self.trading_horizon + self.prediction_horizon

        # 为每个完整的序列创建滑动窗口
        for start_idx in range(len(df) - min_required_days + 1):
            # 为当前序列创建20个滑动窗口
            sequence_features = []
            sequence_price_targets = []
            sequence_positions = []
            sequence_returns = []

            for slide in range(self.trading_horizon):  # 20次滑动
                # 输入特征：180天历史数据
                input_start = start_idx + slide
                input_end = input_start + self.sequence_length
                seq_features = feature_data[input_start:input_end]

                # 价格预测目标：未来7天价格
                price_start = input_end
                price_end = price_start + self.prediction_horizon
                if price_end <= len(close_prices):
                    target_prices = close_prices[price_start:price_end]
                else:
                    # 如果超出范围，用最后的价格填充
                    available_prices = close_prices[price_start:]
                    target_prices = np.concatenate([
                        available_prices,
                        np.full(price_end - len(close_prices), close_prices[-1])
                    ])

                # 次日涨跌幅（用于计算仓位收益）
                next_day_idx = input_end  # 预测日的第一天
                if next_day_idx < len(price_changes):
                    next_return = price_changes[next_day_idx]
                else:
                    next_return = 0.0  # 如果没有次日数据，收益为0

                sequence_features.append(seq_features)
                sequence_price_targets.append(target_prices)
                sequence_returns.append(next_return)

                # 仓位目标暂时设为0，在训练时会被忽略
                sequence_positions.append(0.0)

            features_list.append(np.array(sequence_features))
            price_targets_list.append(np.array(sequence_price_targets))
            position_targets.append(np.array(sequence_positions))
            next_day_returns.append(np.array(sequence_returns))

        return (
            torch.FloatTensor(features_list),      # [n_sequences, 20, 180, n_features]
            torch.FloatTensor(price_targets_list), # [n_sequences, 20, 7]
            torch.FloatTensor(position_targets),   # [n_sequences, 20]
            torch.FloatTensor(next_day_returns)    # [n_sequences, 20]
        )

    def fit_normalizer(self, df: pd.DataFrame):
        """
        计算标准化参数
        
        Args:
            df: 训练数据DataFrame
        """
        if not self.normalize:
            return
            
        # 计算特征统计
        for col in self.feature_columns:
            if col in df.columns:
                self.feature_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
        
        # 计算价格统计（用于目标标准化）
        self.price_stats = {
            'mean': df['close'].mean(),
            'std': df['close'].std()
        }
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化特征
        
        Args:
            df: 原始DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        if not self.normalize or not self.feature_stats:
            return df
            
        df = df.copy()
        
        for col in self.feature_columns:
            if col in df.columns and col in self.feature_stats:
                stats = self.feature_stats[col]
                df[col] = (df[col] - stats['mean']) / (stats['std'] + 1e-8)
        
        return df
    
    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        反标准化预测结果
        
        Args:
            predictions: 标准化的预测张量
            
        Returns:
            反标准化的预测张量
        """
        if not self.normalize or not self.price_stats:
            return predictions
            
        mean = self.price_stats['mean']
        std = self.price_stats['std']
        
        return predictions * std + mean
    
    def process_file(self, file_path: str) -> Tuple[torch.Tensor, ...]:
        """
        处理整个数据文件

        Args:
            file_path: 数据文件路径

        Returns:
            如果启用交易策略: (features, price_targets, trading_prices)
            否则: (features, price_targets)
        """
        # 读取并解析数据
        data_list = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    parsed = self.parse_data_line(line)
                    if parsed:
                        data_list.append(parsed)
        
        if not data_list:
            raise ValueError("没有成功解析任何数据")
        
        # 转换为DataFrame
        df = pd.DataFrame(data_list)
        
        # 添加时间编码
        df = self.add_time_encoding(df)
        
        # 计算标准化参数
        self.fit_normalizer(df)
        
        # 标准化特征
        df = self.normalize_features(df)
        
        # 创建序列
        sequences = self.create_sequences(df)

        if self.enable_trading_strategy:
            features, price_targets, trading_prices = sequences
            print(f"✅ 数据处理完成:")
            print(f"  - 总样本数: {len(features)}")
            print(f"  - 特征维度: {features.shape}")
            print(f"  - 价格目标维度: {price_targets.shape}")
            print(f"  - 交易价格维度: {trading_prices.shape}")
            return features, price_targets, trading_prices
        else:
            features, price_targets = sequences
            print(f"✅ 数据处理完成:")
            print(f"  - 总样本数: {len(features)}")
            print(f"  - 特征维度: {features.shape}")
            print(f"  - 目标维度: {price_targets.shape}")
            return features, price_targets


def create_sample_data(n_days: int = 100) -> str:
    """
    创建示例金融数据
    
    Args:
        n_days: 生成天数
        
    Returns:
        示例数据字符串
    """
    import random
    from datetime import datetime, timedelta
    
    lines = []
    base_price = 20.0
    
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_days):
        date = start_date + timedelta(days=i)
        weekday = ['一', '二', '三', '四', '五', '六', '日'][date.weekday()]
        
        # 模拟价格波动
        change = random.uniform(-0.05, 0.05)
        base_price *= (1 + change)
        
        open_price = base_price * random.uniform(0.98, 1.02)
        high_price = max(open_price, base_price) * random.uniform(1.0, 1.05)
        low_price = min(open_price, base_price) * random.uniform(0.95, 1.0)
        close_price = base_price
        
        change_pct = change * 100
        amplitude = (high_price - low_price) / base_price * 100
        volume = random.randint(50000000, 200000000)
        amount = volume * base_price
        turnover_rate = random.uniform(1.0, 10.0)
        trade_count = random.randint(1000, 5000)
        
        line = f"{date.strftime('%Y-%m-%d')},{weekday}\t{open_price:.2f}\t{high_price:.2f}\t{low_price:.2f}\t{close_price:.2f}\t{change_pct:.2f}%\t{amplitude:.2f}%\t{volume:,}\t{amount:,.0f}\t{turnover_rate:.2f}\t{trade_count}"
        lines.append(line)
    
    return '\n'.join(lines)

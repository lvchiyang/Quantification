"""
价格预测网络的数据处理器
专门为Transformer架构设计，处理长序列时序数据
"""

import torch
import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Dict
from datetime import datetime
import warnings

class PricePredictionDataProcessor:
    """
    价格预测数据处理器
    
    专门为Transformer架构设计：
    - 输入: 180天历史金融特征 [batch, 180, 13]
    - 输出: 未来7天价格预测 [batch, 7]
    - 关注: 长期时序模式和价格趋势
    """
    
    def __init__(self,
                 data_dir: str = "processed_data_2025-07-29",
                 sequence_length: int = 180,
                 prediction_horizon: int = 7,
                 large_value_transform: str = "relative_change"):
        """
        初始化价格预测数据处理器
        
        Args:
            data_dir: 数据目录
            sequence_length: 输入序列长度（180天）
            prediction_horizon: 预测时间跨度（7天）
            large_value_transform: 大数值处理方法
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.large_value_transform = large_value_transform
        
        # 原始列名（跳过第一列"年"）
        self.raw_columns = [
            '月', '日', '星期', '开盘', '最高', '最低', '收盘', 
            '涨幅', '振幅', '总手', '金额', '换手%', '成交次数'
        ]
        
        # 价格预测特征（专注于价格相关特征）
        self.feature_columns = [
            'month', 'day', 'weekday',           # 时间特征
            'open', 'high', 'low', 'close',      # OHLC价格
            'change_pct', 'amplitude',           # 价格变化特征
            'volume_processed', 'amount_processed', # 成交量特征
            'turnover_rate', 'trade_count'       # 市场活跃度
        ]
        
        # 大数值列
        self.large_value_columns = ['总手', '金额']
        
    def load_stock_data(self, stock_path: str) -> pd.DataFrame:
        """加载单个股票数据"""
        try:
            df = pd.read_excel(stock_path)
            df = df.iloc[:, 1:]  # 跳过第一列"年"
            df.columns = self.raw_columns
            
            # 数据类型转换
            for col in ['月', '日', '星期']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            for col in ['开盘', '最高', '最低', '收盘', '涨幅', '振幅', '换手%']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            for col in ['总手', '金额', '成交次数']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna().reset_index(drop=True)
            return df
            
        except Exception as e:
            warnings.warn(f"加载股票数据失败: {stock_path}, 错误: {e}")
            return None
    
    def process_large_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理大数值列"""
        df_processed = df.copy()
        
        for col in self.large_value_columns:
            if col in df_processed.columns:
                values = df_processed[col].values
                
                if self.large_value_transform == "relative_change":
                    # 相对变化率（推荐）
                    rolling_mean = pd.Series(values).rolling(window=20, min_periods=1).mean()
                    relative_change = (values - rolling_mean) / rolling_mean * 100
                    processed_values = np.nan_to_num(relative_change, nan=0.0)
                else:
                    processed_values = values
                
                new_col_name = col.replace('总手', 'volume_processed').replace('金额', 'amount_processed')
                df_processed[new_col_name] = processed_values
        
        return df_processed
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建价格预测特征"""
        feature_df = pd.DataFrame()
        
        # 时间特征
        feature_df['month'] = df['月']
        feature_df['day'] = df['日']
        feature_df['weekday'] = df['星期']
        
        # OHLC价格特征
        feature_df['open'] = df['开盘']
        feature_df['high'] = df['最高']
        feature_df['low'] = df['最低']
        feature_df['close'] = df['收盘']
        
        # 价格变化特征
        feature_df['change_pct'] = df['涨幅']
        feature_df['amplitude'] = df['振幅']
        
        # 成交量特征
        if 'volume_processed' in df.columns:
            feature_df['volume_processed'] = df['volume_processed']
        if 'amount_processed' in df.columns:
            feature_df['amount_processed'] = df['amount_processed']
        
        # 市场活跃度
        feature_df['turnover_rate'] = df['换手%']
        feature_df['trade_count'] = df['成交次数']
        
        return feature_df
    
    def create_price_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建价格预测序列
        
        Args:
            df: 特征DataFrame
            
        Returns:
            (input_features, price_targets)
            - input_features: [n_samples, 180, 13] 输入特征序列
            - price_targets: [n_samples, 7] 价格预测目标
        """
        if len(df) < self.sequence_length + self.prediction_horizon:
            warnings.warn(f"数据长度不足: {len(df)} < {self.sequence_length + self.prediction_horizon}")
            return None, None
        
        features_list = []
        targets_list = []
        
        # 滑动窗口创建序列
        for i in range(len(df) - self.sequence_length - self.prediction_horizon + 1):
            # 输入特征：180天历史数据
            feature_seq = df.iloc[i:i+self.sequence_length][self.feature_columns].values
            features_list.append(feature_seq)
            
            # 预测目标：未来7天收盘价
            target_start = i + self.sequence_length
            target_end = target_start + self.prediction_horizon
            price_targets = df.iloc[target_start:target_end]['close'].values
            targets_list.append(price_targets)
        
        features = torch.FloatTensor(np.array(features_list))
        targets = torch.FloatTensor(np.array(targets_list))
        
        return features, targets
    
    def load_all_stocks_for_price_prediction(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        加载所有股票的价格预测数据
        
        Returns:
            股票名称到(features, targets)的映射
        """
        stock_data = {}
        stock_dir = os.path.join(self.data_dir, "股票数据")
        
        if not os.path.exists(stock_dir):
            raise FileNotFoundError(f"股票数据目录不存在: {stock_dir}")
        
        for industry_dir in os.listdir(stock_dir):
            industry_path = os.path.join(stock_dir, industry_dir)
            if os.path.isdir(industry_path):
                for stock_file in os.listdir(industry_path):
                    if stock_file.endswith('.xlsx'):
                        stock_name = stock_file.replace('.xlsx', '')
                        stock_path = os.path.join(industry_path, stock_file)
                        
                        # 处理单只股票
                        df = self.load_stock_data(stock_path)
                        if df is not None:
                            df = self.process_large_values(df)
                            feature_df = self.create_features(df)
                            
                            # 创建价格预测序列
                            features, targets = self.create_price_sequences(feature_df)
                            if features is not None:
                                stock_key = f"{industry_dir}_{stock_name}"
                                stock_data[stock_key] = (features, targets)
                                print(f"价格预测数据: {stock_key}, 序列数: {len(features)}")
        
        print(f"价格预测模块加载了 {len(stock_data)} 只股票")
        return stock_data
    
    def get_feature_info(self) -> Dict:
        """获取特征信息"""
        return {
            'n_features': len(self.feature_columns),
            'feature_names': self.feature_columns,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'model_type': 'transformer_price_prediction'
        }

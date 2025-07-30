"""
策略网络的数据处理器
专门为GRU架构设计，处理交易决策序列数据
"""

import torch
import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Dict
from datetime import datetime
import warnings

class StrategyNetworkDataProcessor:
    """
    策略网络数据处理器
    
    专门为GRU架构设计：
    - 输入: 20天策略特征 [batch, 20, feature_dim]
    - 输出: 20天仓位决策 [batch, 20, 1] + 收益率 [batch, 20]
    - 关注: 短期决策序列和收益优化
    """
    
    def __init__(self,
                 data_dir: str = "processed_data_2025-07-29",
                 trading_horizon: int = 20,
                 feature_extraction_length: int = 180,
                 large_value_transform: str = "relative_change"):
        """
        初始化策略网络数据处理器
        
        Args:
            data_dir: 数据目录
            trading_horizon: 交易决策时间跨度（20天）
            feature_extraction_length: 特征提取的历史长度（180天）
            large_value_transform: 大数值处理方法
        """
        self.data_dir = data_dir
        self.trading_horizon = trading_horizon
        self.feature_extraction_length = feature_extraction_length
        self.large_value_transform = large_value_transform
        
        # 原始列名
        self.raw_columns = [
            '月', '日', '星期', '开盘', '最高', '最低', '收盘', 
            '涨幅', '振幅', '总手', '金额', '换手%', '成交次数'
        ]
        
        # 策略特征（专注于交易决策相关特征）
        self.strategy_features = [
            'price_trend',           # 价格趋势特征
            'volatility',           # 波动率特征
            'volume_trend',         # 成交量趋势
            'market_sentiment',     # 市场情绪
            'technical_indicators', # 技术指标
            'risk_metrics'          # 风险指标
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
    
    def create_strategy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建策略决策特征
        
        专门为交易策略设计的特征工程：
        - 价格趋势和动量
        - 波动率和风险指标
        - 成交量和市场情绪
        - 技术指标
        """
        feature_df = pd.DataFrame()
        
        # 1. 价格趋势特征
        feature_df['price_trend'] = self._calculate_price_trend(df)
        feature_df['price_momentum'] = self._calculate_momentum(df['收盘'])
        feature_df['price_acceleration'] = self._calculate_acceleration(df['收盘'])
        
        # 2. 波动率特征
        feature_df['volatility'] = self._calculate_volatility(df)
        feature_df['volatility_trend'] = self._calculate_volatility_trend(df)
        
        # 3. 成交量特征
        feature_df['volume_trend'] = self._calculate_volume_trend(df)
        feature_df['volume_price_correlation'] = self._calculate_volume_price_corr(df)
        
        # 4. 市场情绪特征
        feature_df['market_sentiment'] = self._calculate_market_sentiment(df)
        feature_df['sentiment_momentum'] = self._calculate_sentiment_momentum(df)
        
        # 5. 技术指标
        feature_df['rsi'] = self._calculate_rsi(df['收盘'])
        feature_df['macd_signal'] = self._calculate_macd_signal(df['收盘'])
        
        # 6. 风险指标
        feature_df['risk_metrics'] = self._calculate_risk_metrics(df)
        feature_df['drawdown_risk'] = self._calculate_drawdown_risk(df['收盘'])
        
        return feature_df
    
    def _calculate_price_trend(self, df: pd.DataFrame, window: int = 10) -> np.ndarray:
        """计算价格趋势"""
        close_prices = df['收盘']
        ma_short = close_prices.rolling(window=window//2).mean()
        ma_long = close_prices.rolling(window=window).mean()
        trend = (ma_short - ma_long) / ma_long * 100
        return trend.fillna(0).values
    
    def _calculate_momentum(self, prices: pd.Series, window: int = 5) -> np.ndarray:
        """计算价格动量"""
        momentum = prices.pct_change(window) * 100
        return momentum.fillna(0).values
    
    def _calculate_acceleration(self, prices: pd.Series) -> np.ndarray:
        """计算价格加速度"""
        returns = prices.pct_change()
        acceleration = returns.diff()
        return acceleration.fillna(0).values * 10000  # 放大以便观察
    
    def _calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> np.ndarray:
        """计算波动率"""
        returns = df['收盘'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100
        return volatility.fillna(volatility.mean()).values
    
    def _calculate_volatility_trend(self, df: pd.DataFrame) -> np.ndarray:
        """计算波动率趋势"""
        volatility = self._calculate_volatility(df)
        vol_series = pd.Series(volatility)
        vol_trend = vol_series.pct_change(5) * 100
        return vol_trend.fillna(0).values
    
    def _calculate_volume_trend(self, df: pd.DataFrame, window: int = 10) -> np.ndarray:
        """计算成交量趋势"""
        volume = df['总手']
        vol_ma = volume.rolling(window=window).mean()
        vol_trend = (volume - vol_ma) / vol_ma * 100
        return vol_trend.fillna(0).values
    
    def _calculate_volume_price_corr(self, df: pd.DataFrame, window: int = 20) -> np.ndarray:
        """计算量价相关性"""
        price_change = df['收盘'].pct_change()
        volume_change = df['总手'].pct_change()
        correlation = price_change.rolling(window=window).corr(volume_change)
        return correlation.fillna(0).values
    
    def _calculate_market_sentiment(self, df: pd.DataFrame) -> np.ndarray:
        """计算市场情绪"""
        # 基于涨幅和振幅的综合情绪指标
        change_pct = df['涨幅']
        amplitude = df['振幅']
        sentiment = (change_pct * 0.7 + amplitude * 0.3) / 2
        return sentiment.values
    
    def _calculate_sentiment_momentum(self, df: pd.DataFrame) -> np.ndarray:
        """计算情绪动量"""
        sentiment = self._calculate_market_sentiment(df)
        sentiment_series = pd.Series(sentiment)
        momentum = sentiment_series.rolling(window=5).mean()
        return momentum.fillna(0).values
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> np.ndarray:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values  # 中性值50
    
    def _calculate_macd_signal(self, prices: pd.Series) -> np.ndarray:
        """计算MACD信号"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_signal = macd - signal
        return macd_signal.fillna(0).values
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> np.ndarray:
        """计算风险指标"""
        returns = df['收盘'].pct_change()
        # 简单的风险指标：负收益的概率
        risk_window = 10
        negative_return_ratio = (returns < 0).rolling(window=risk_window).mean()
        return negative_return_ratio.fillna(0.5).values * 100
    
    def _calculate_drawdown_risk(self, prices: pd.Series, window: int = 20) -> np.ndarray:
        """计算回撤风险"""
        rolling_max = prices.rolling(window=window).max()
        drawdown = (prices - rolling_max) / rolling_max * 100
        return drawdown.fillna(0).values
    
    def create_strategy_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建策略决策序列
        
        Args:
            df: 原始DataFrame
            
        Returns:
            (strategy_features, position_targets, next_day_returns)
            - strategy_features: [n_samples, 20, n_features] 策略特征序列
            - position_targets: [n_samples, 20, 1] 仓位目标（需要学习）
            - next_day_returns: [n_samples, 20] 下一日收益率
        """
        min_length = self.feature_extraction_length + self.trading_horizon
        if len(df) < min_length:
            warnings.warn(f"数据长度不足: {len(df)} < {min_length}")
            return None, None, None
        
        # 创建策略特征
        strategy_df = self.create_strategy_features(df)
        
        features_list = []
        returns_list = []
        
        # 滑动窗口创建序列
        for i in range(len(df) - min_length + 1):
            # 策略特征：20天决策序列
            feature_start = i + self.feature_extraction_length - self.trading_horizon
            feature_end = i + self.feature_extraction_length
            feature_seq = strategy_df.iloc[feature_start:feature_end].values
            features_list.append(feature_seq)
            
            # 下一日收益率：用于计算策略收益
            returns = []
            for j in range(self.trading_horizon):
                day_idx = i + self.feature_extraction_length + j
                if day_idx + 1 < len(df):
                    today_price = df.iloc[day_idx]['收盘']
                    tomorrow_price = df.iloc[day_idx + 1]['收盘']
                    daily_return = (tomorrow_price - today_price) / today_price
                    returns.append(daily_return)
                else:
                    returns.append(0.0)
            returns_list.append(returns)
        
        features = torch.FloatTensor(np.array(features_list))
        returns = torch.FloatTensor(np.array(returns_list))
        
        # 仓位目标初始化为零（需要通过训练学习）
        position_targets = torch.zeros(features.shape[0], self.trading_horizon, 1)
        
        return features, position_targets, returns
    
    def load_all_stocks_for_strategy(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        加载所有股票的策略数据
        
        Returns:
            股票名称到(features, positions, returns)的映射
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
                            # 创建策略序列
                            features, positions, returns = self.create_strategy_sequences(df)
                            if features is not None:
                                stock_key = f"{industry_dir}_{stock_name}"
                                stock_data[stock_key] = (features, positions, returns)
                                print(f"策略数据: {stock_key}, 序列数: {len(features)}")
        
        print(f"策略网络模块加载了 {len(stock_data)} 只股票")
        return stock_data
    
    def get_feature_info(self) -> Dict:
        """获取特征信息"""
        return {
            'n_features': len(self.strategy_features) * 2,  # 大约12-14个特征
            'feature_names': self.strategy_features,
            'trading_horizon': self.trading_horizon,
            'model_type': 'gru_strategy_network'
        }

"""
滑动窗口预测器
实现200天数据的20次滑动窗口预测和累计收益计算
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any


class SlidingWindowPredictor:
    """
    滑动窗口预测器
    
    实现对200天金融数据进行20次滑动窗口预测：
    - 每次使用180天历史数据
    - 预测未来7天价格
    - 输出当天仓位（0-10）
    - 根据次日涨跌幅累计收益
    """
    
    def __init__(self, model, processor):
        """
        初始化预测器
        
        Args:
            model: 训练好的FinancialTransformer模型
            processor: 数据处理器
        """
        self.model = model
        self.processor = processor
        self.model.eval()
    
    def predict_sequence(
        self, 
        financial_data: np.ndarray, 
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        对一个200天的金融序列进行滑动窗口预测
        
        Args:
            financial_data: 金融数据 [200+, n_features]
            return_details: 是否返回详细信息
            
        Returns:
            预测结果字典
        """
        if len(financial_data) < 200:
            raise ValueError(f"需要至少200天数据，实际得到{len(financial_data)}天")
        
        # 存储预测结果
        price_predictions_list = []
        position_predictions_list = []
        actual_returns_list = []
        
        # 计算实际涨跌幅
        close_prices = financial_data[:, 3]  # 假设收盘价在第4列
        actual_returns = np.diff(close_prices) / close_prices[:-1]
        
        # 进行20次滑动窗口预测
        for slide in range(20):
            # 获取180天历史数据
            start_idx = slide
            end_idx = start_idx + 180
            history_data = financial_data[start_idx:end_idx]
            
            # 标准化数据
            history_df = pd.DataFrame(history_data, columns=[
                'open', 'high', 'low', 'close', 'change_pct', 
                'amplitude', 'volume', 'amount', 'turnover_rate', 
                'trade_count', 'time_encoding'
            ])
            
            # 应用标准化
            if self.processor.normalize:
                for col in self.processor.feature_columns:
                    if col in history_df.columns and col in self.processor.feature_stats:
                        stats = self.processor.feature_stats[col]
                        history_df[col] = (history_df[col] - stats['mean']) / (stats['std'] + 1e-8)
            
            # 转换为tensor
            input_tensor = torch.FloatTensor(history_df.values).unsqueeze(0)  # [1, 180, 11]
            
            # 模型预测
            with torch.no_grad():
                outputs = self.model.predict(input_tensor, return_dict=True)
                
                # 价格预测
                price_pred = outputs['price_predictions']  # [1, 7]
                price_pred_denorm = self.processor.denormalize_predictions(price_pred)
                
                # 仓位预测
                position_pred = outputs.get('position_predictions', torch.zeros(1, 1))  # [1, 1]
                position_output = outputs.get('position_output', {})

                # 获取离散仓位（如果可用）
                if 'discrete_positions' in position_output:
                    discrete_pos = position_output['discrete_positions'][0, 0].cpu().item()
                    position_value = discrete_pos
                else:
                    # 四舍五入到最近的整数
                    position_value = round(position_pred[0, 0].cpu().item())

                # 确保仓位在有效范围内
                position_value = max(0, min(10, int(position_value)))

                price_predictions_list.append(price_pred_denorm[0].cpu().numpy())
                position_predictions_list.append(position_value)
            
            # 获取次日实际涨跌幅
            next_day_idx = end_idx  # 预测日的第一天
            if next_day_idx < len(actual_returns):
                actual_return = actual_returns[next_day_idx]
            else:
                actual_return = 0.0
            
            actual_returns_list.append(actual_return)
        
        # 计算累计收益
        cumulative_return = self._calculate_cumulative_return(
            position_predictions_list, actual_returns_list
        )
        
        result = {
            'price_predictions': np.array(price_predictions_list),  # [20, 7]
            'position_predictions': np.array(position_predictions_list),  # [20]
            'actual_returns': np.array(actual_returns_list),  # [20]
            'cumulative_return': cumulative_return,
            'final_portfolio_value': 1.0 + cumulative_return
        }
        
        if return_details:
            result['details'] = self._calculate_detailed_returns(
                position_predictions_list, actual_returns_list
            )
        
        return result
    
    def _calculate_cumulative_return(
        self, 
        positions: List[float], 
        returns: List[float]
    ) -> float:
        """
        计算累计收益率
        
        Args:
            positions: 每天的仓位 [20]
            returns: 每天的次日涨跌幅 [20]
            
        Returns:
            累计收益率
        """
        portfolio_value = 1.0  # 初始份额为1
        
        for position, next_return in zip(positions, returns):
            # 仓位收益 = 仓位 * 次日涨跌幅
            position_return = (position / 10.0) * next_return  # 将仓位标准化到[0,1]
            portfolio_value *= (1.0 + position_return)
        
        return portfolio_value - 1.0  # 返回收益率
    
    def _calculate_detailed_returns(
        self, 
        positions: List[float], 
        returns: List[float]
    ) -> Dict[str, Any]:
        """
        计算详细的收益信息
        
        Args:
            positions: 每天的仓位 [20]
            returns: 每天的次日涨跌幅 [20]
            
        Returns:
            详细收益信息
        """
        portfolio_values = [1.0]  # 初始份额为1
        daily_returns = []
        
        for i, (position, next_return) in enumerate(zip(positions, returns)):
            # 计算当日收益
            position_normalized = position / 10.0  # 标准化仓位到[0,1]
            daily_return = position_normalized * next_return
            daily_returns.append(daily_return)
            
            # 更新组合价值
            new_value = portfolio_values[-1] * (1.0 + daily_return)
            portfolio_values.append(new_value)
        
        return {
            'portfolio_values': portfolio_values,  # [21] 包含初始值
            'daily_returns': daily_returns,  # [20]
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'sharpe_ratio': self._calculate_sharpe_ratio(daily_returns),
            'win_rate': sum(1 for r in daily_returns if r > 0) / len(daily_returns)
        }
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        """计算夏普比率"""
        if len(daily_returns) == 0:
            return 0.0
        
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return == 0:
            return 0.0
        
        # 假设无风险利率为0
        return mean_return / std_return * np.sqrt(252)  # 年化夏普比率
    
    def batch_predict(
        self, 
        financial_data_list: List[np.ndarray],
        return_details: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量预测多个序列
        
        Args:
            financial_data_list: 多个金融数据序列
            return_details: 是否返回详细信息
            
        Returns:
            预测结果列表
        """
        results = []
        
        for i, data in enumerate(financial_data_list):
            print(f"预测序列 {i+1}/{len(financial_data_list)}...")
            result = self.predict_sequence(data, return_details)
            results.append(result)
        
        return results
    
    def evaluate_strategy(
        self, 
        financial_data_list: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        评估策略性能
        
        Args:
            financial_data_list: 多个金融数据序列
            
        Returns:
            策略性能指标
        """
        results = self.batch_predict(financial_data_list, return_details=True)
        
        # 汇总统计
        cumulative_returns = [r['cumulative_return'] for r in results]
        final_values = [r['final_portfolio_value'] for r in results]
        
        all_daily_returns = []
        all_max_drawdowns = []
        all_win_rates = []
        
        for r in results:
            if 'details' in r:
                all_daily_returns.extend(r['details']['daily_returns'])
                all_max_drawdowns.append(r['details']['max_drawdown'])
                all_win_rates.append(r['details']['win_rate'])
        
        return {
            'mean_cumulative_return': np.mean(cumulative_returns),
            'std_cumulative_return': np.std(cumulative_returns),
            'mean_final_value': np.mean(final_values),
            'win_rate_sequences': sum(1 for r in cumulative_returns if r > 0) / len(cumulative_returns),
            'mean_daily_return': np.mean(all_daily_returns) if all_daily_returns else 0.0,
            'overall_sharpe_ratio': self._calculate_sharpe_ratio(all_daily_returns),
            'mean_max_drawdown': np.mean(all_max_drawdowns) if all_max_drawdowns else 0.0,
            'mean_win_rate': np.mean(all_win_rates) if all_win_rates else 0.0
        }

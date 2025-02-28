import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from datetime import datetime, timedelta
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import os

class GoldTradingSystem:
    def __init__(self, data_path=None, timeframe="1min"):
        self.data_path = data_path
        self.timeframe = timeframe
        self.data = None
        self.signals = None
        self.positions = None
        self.risk_per_trade = 0.02  # 每笔交易风险2%
        
    def load_local_data(self, file_path=None, time_column='t'):
        """从本地文件加载分钟级数据，适配自定义列名"""
        if file_path is None:
            file_path = self.data_path
            
        if file_path is None:
            raise ValueError("请提供数据文件路径")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # 根据文件扩展名选择加载方法
            if file_ext == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            elif file_ext == '.parquet':
                self.data = pd.read_parquet(file_path)
            elif file_ext == '.pickle' or file_ext == '.pkl':
                self.data = pd.read_pickle(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
                
            # 确保时间列被正确解析为datetime
            if time_column in self.data.columns:
                # 检查时间列的格式，如果是Unix时间戳则转换
                if pd.api.types.is_numeric_dtype(self.data[time_column]):
                    # 假设时间戳是毫秒级
                    if self.data[time_column].iloc[0] > 1e11:
                        self.data[time_column] = pd.to_datetime(self.data[time_column], unit='ms')
                    # 假设时间戳是秒级
                    else:
                        self.data[time_column] = pd.to_datetime(self.data[time_column], unit='s')
                else:
                    self.data[time_column] = pd.to_datetime(self.data[time_column])
                
                self.data.set_index(time_column, inplace=True)
            
            # 确保列名映射到标准名称
            column_mapping = {
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                'vw': 'VWAP'
            }
            
            # 重命名列
            self.data.rename(columns=column_mapping, inplace=True)
            
            # 确保数据按时间排序
            self.data.sort_index(inplace=True)
            
            print(f"成功加载数据，共 {len(self.data)} 行")
            print(f"数据时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
            print(f"数据列: {self.data.columns.tolist()}")
            
            # 检查是否有缺失值
            missing_values = self.data.isnull().sum()
            if missing_values.sum() > 0:
                print("警告: 数据中存在缺失值:")
                print(missing_values[missing_values > 0])
                
            return self.data
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def resample_data(self, timeframe=None):
        """重采样数据到指定的时间框架"""
        if self.data is None:
            print("没有数据可重采样")
            return None
            
        if timeframe is None:
            timeframe = self.timeframe
            
        # 如果已经是目标时间框架，则不需要重采样
        if timeframe == '1min' or timeframe.lower() == 'original':
            return self.data
            
        # 定义重采样规则
        timeframe_map = {
            '1min': '1T', '3min': '3T', '5min': '5T', '15min': '15T',
            '30min': '30T', '1h': 'H', '4h': '4H', 'D': 'D'
        }
        
        if timeframe not in timeframe_map:
            print(f"不支持的时间框架: {timeframe}，使用原始数据")
            return self.data
            
        rule = timeframe_map[timeframe]
        
        # 执行重采样
        resampled = self.data.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'VWAP': 'last'  # 使用最后一个VWAP值
        })
        
        # 删除缺失值
        resampled.dropna(inplace=True)
        
        print(f"数据已重采样至 {timeframe}，共 {len(resampled)} 行")
        self.data = resampled
        return self.data
    
    def add_technical_indicators(self):
        """添加技术指标"""
        if self.data is None or len(self.data) == 0:
            print("没有数据，无法计算指标")
            return
            
        # 添加移动平均线 - 使用较短的周期适应分钟级数据
        self.data['SMA10'] = ta.sma(self.data['Close'], length=10)
        self.data['SMA30'] = ta.sma(self.data['Close'], length=30)
        
        # 添加MACD - 调整参数适应分钟级数据
        macd = ta.macd(self.data['Close'], fast=12, slow=26, signal=9)
        self.data = pd.concat([self.data, macd], axis=1)
        
        # 添加RSI
        self.data['RSI'] = ta.rsi(self.data['Close'], length=14)
        
        # 添加布林带
        bbands = ta.bbands(self.data['Close'], length=20)
        self.data = pd.concat([self.data, bbands], axis=1)
        
        # 添加ATR (波动率)
        self.data['ATR'] = ta.atr(self.data['High'], self.data['Low'], 
                                 self.data['Close'], length=14)
        
        # 添加ADX (趋势强度)
        adx = ta.adx(self.data['High'], self.data['Low'], self.data['Close'], length=14)
        self.data = pd.concat([self.data, adx], axis=1)
        
        # 计算成交量变化率
        self.data['Volume_Change'] = self.data['Volume'].pct_change()
        
        # 添加价格动量
        self.data['Momentum'] = self.data['Close'].diff(5)
        
        # 添加波动率指标
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std()
        
        # 删除NaN值
        self.data.dropna(inplace=True)
        return self.data
    
    def test_normality(self):
        """测试价格变化是否符合正态分布"""
        returns = self.data['Close'].pct_change().dropna()
        
        # 对于大数据集，采样以提高效率
        sample_size = min(5000, len(returns))
        sample_returns = returns.sample(sample_size) if len(returns) > sample_size else returns
        
        # Shapiro-Wilk测试
        stat, p_value = stats.shapiro(sample_returns)
        print(f"Shapiro-Wilk测试: 统计量={stat:.4f}, p值={p_value:.4f}")
        if p_value < 0.05:
            print("拒绝原假设：数据不符合正态分布")
        else:
            print("接受原假设：数据符合正态分布")
            
        # QQ图
        plt.figure(figsize=(10, 6))
        stats.probplot(sample_returns, dist="norm", plot=plt)
        plt.title("价格变化的QQ图")
        plt.show()
        
        # 计算峰度和偏度
        kurtosis = stats.kurtosis(returns)
        skewness = stats.skew(returns)
        print(f"峰度: {kurtosis:.4f} (正态分布为0)")
        print(f"偏度: {skewness:.4f} (正态分布为0)")
        
        return {"kurtosis": kurtosis, "skewness": skewness, "p_value": p_value}
    
    def generate_signals(self):
        """生成交易信号"""
        if self.data is None:
            return None
            
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['Price'] = self.data['Close']
        self.signals['Signal'] = 0  # 0:无信号, 1:买入, -1:卖出
        
        # 策略1: 均线交叉 + ADX过滤
        self.signals.loc[(self.data['SMA10'] > self.data['SMA30']) & 
                         (self.data['ADX_14'] > 20), 'Signal_MA_Cross'] = 1
        self.signals.loc[(self.data['SMA10'] < self.data['SMA30']) & 
                         (self.data['ADX_14'] > 20), 'Signal_MA_Cross'] = -1
        
        # 策略2: RSI超买超卖 + 成交量确认
        self.signals.loc[(self.data['RSI'] < 30) & 
                         (self.data['Volume_Change'] > 0), 'Signal_RSI'] = 1
        self.signals.loc[(self.data['RSI'] > 70) & 
                         (self.data['Volume_Change'] > 0), 'Signal_RSI'] = -1
        
        # 策略3: 布林带突破 + ATR止损
        self.signals.loc[self.data['Close'] > self.data['BBU_20_2.0'], 'Signal_BB'] = 1
        self.signals.loc[self.data['Close'] < self.data['BBL_20_2.0'], 'Signal_BB'] = -1
        
        # 策略4: MACD交叉
        self.signals.loc[(self.data['MACDh_12_26_9'] > 0) & 
                         (self.data['MACDh_12_26_9'].shift(1) < 0), 'Signal_MACD'] = 1
        self.signals.loc[(self.data['MACDh_12_26_9'] < 0) & 
                         (self.data['MACDh_12_26_9'].shift(1) > 0), 'Signal_MACD'] = -1
        
        # 策略5: 价格动量 + 波动率过滤
        self.signals.loc[(self.data['Momentum'] > 0) & 
                         (self.data['Volatility'] < self.data['Volatility'].rolling(50).mean()), 'Signal_Momentum'] = 1
        self.signals.loc[(self.data['Momentum'] < 0) & 
                         (self.data['Volatility'] < self.data['Volatility'].rolling(50).mean()), 'Signal_Momentum'] = -1
        
        # 策略6: VWAP交叉
        self.signals.loc[self.data['Close'] > self.data['VWAP'], 'Signal_VWAP'] = 1
        self.signals.loc[self.data['Close'] < self.data['VWAP'], 'Signal_VWAP'] = -1
        
        # 综合信号 (加权投票)
        signal_cols = [col for col in self.signals.columns if col.startswith('Signal_')]
        weights = {
            'Signal_MA_Cross': 0.25,
            'Signal_RSI': 0.15,
            'Signal_BB': 0.15,
            'Signal_MACD': 0.20,
            'Signal_Momentum': 0.10,
            'Signal_VWAP': 0.15
        }
        
        self.signals['Signal'] = 0
        for col, weight in weights.items():
            if col in self.signals.columns:
                self.signals['Signal'] += self.signals[col].fillna(0) * weight
        
        # 转换为离散信号
        self.signals['Signal'] = np.where(self.signals['Signal'] > 0.2, 1, 
                                         np.where(self.signals['Signal'] < -0.2, -1, 0))
        
        # 信号平滑 (避免频繁交易)
        min_holding_periods = 5  # 最小持仓周期
        for i in range(1, len(self.signals)):
            if (i < min_holding_periods or 
                (self.signals['Signal'].iloc[i-1] != 0 and 
                 self.signals['Signal'].iloc[i-1] != self.signals['Signal'].iloc[i])):
                last_non_zero = self.signals['Signal'].iloc[i-1]
                if last_non_zero != 0:
                    self.signals['Signal'].iloc[i] = 0  # 不急于反转信号
        
        return self.signals
    
    def calculate_position_size(self, price, stop_loss):
        """计算仓位大小"""
        account_size = 10000  # 假设账户大小
        risk_amount = account_size * self.risk_per_trade
        position_size = risk_amount / abs(price - stop_loss)
        return position_size
    
    def backtest(self, commission=0.0001, slippage=0.0001):
        """回测策略，包含交易成本"""
        if self.signals is None:
            self.generate_signals()
            
        self.positions = self.signals.copy()
        self.positions['Position'] = 0
        
        # 根据信号生成持仓
        for i in range(1, len(self.positions)):
            if self.positions['Signal'].iloc[i] == 1 and self.positions['Position'].iloc[i-1] <= 0:
                # 买入信号
                self.positions['Position'].iloc[i] = 1
            elif self.positions['Signal'].iloc[i] == -1 and self.positions['Position'].iloc[i-1] >= 0:
                # 卖出信号
                self.positions['Position'].iloc[i] = -1
            else:
                # 保持原有仓位
                self.positions['Position'].iloc[i] = self.positions['Position'].iloc[i-1]
        
        # 计算收益 (考虑交易成本)
        self.positions['Returns'] = 0.0
        
        for i in range(1, len(self.positions)):
            # 如果仓位发生变化，计入交易成本
            if self.positions['Position'].iloc[i] != self.positions['Position'].iloc[i-1]:
                # 交易成本 = 佣金 + 滑点
                cost = commission + slippage
                
                # 如果是从多仓到空仓或从空仓到多仓，成本翻倍(双向交易)
                if (self.positions['Position'].iloc[i-1] != 0 and 
                    self.positions['Position'].iloc[i] != 0 and 
                    self.positions['Position'].iloc[i] != self.positions['Position'].iloc[i-1]):
                    cost *= 2
                
                # 应用交易成本
                self.positions['Returns'].iloc[i] = (self.positions['Price'].iloc[i] / 
                                                   self.positions['Price'].iloc[i-1] - 1) * \
                                                   self.positions['Position'].iloc[i-1] - cost
            else:
                # 无交易，正常计算收益
                self.positions['Returns'].iloc[i] = (self.positions['Price'].iloc[i] / 
                                                   self.positions['Price'].iloc[i-1] - 1) * \
                                                   self.positions['Position'].iloc[i-1]
        
        # 计算累计收益
        self.positions['Cumulative_Returns'] = (1 + self.positions['Returns']).cumprod()
        
        # 计算绩效指标
        total_return = self.positions['Cumulative_Returns'].iloc[-1] - 1
        annual_return = total_return * (252 * 390 / len(self.positions))  # 假设一年252个交易日，每天390分钟
        daily_returns = self.positions['Returns']
        sharpe_ratio = np.sqrt(252 * 390) * daily_returns.mean() / daily_returns.std()
        max_drawdown = (self.positions['Cumulative_Returns'] / self.positions['Cumulative_Returns'].cummax() - 1).min()
        
        # 计算交易统计
        trades = self.positions['Position'].diff().fillna(0) != 0
        num_trades = trades.sum()
        win_trades = self.positions.loc[trades, 'Returns'] > 0
        win_rate = win_trades.sum() / num_trades if num_trades > 0 else 0
        
        print(f"总收益率: {total_return:.2%}")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"交易次数: {num_trades}")
        print(f"胜率: {win_rate:.2%}")
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "win_rate": win_rate
        }
    
    def plot_results(self):
        """可视化回测结果"""
        if self.positions is None:
            self.backtest()
            
        # 绘制价格和信号
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.positions.index, self.positions['Price'])
        plt.plot(self.positions.index, self.data['SMA10'], 'r--', alpha=0.5)
        plt.plot(self.positions.index, self.data['SMA30'], 'g--', alpha=0.5)
        plt.plot(self.positions.index, self.data['VWAP'], 'y--', alpha=0.5)
        
        # 标记买入点
        buy_signals = self.positions[self.positions['Signal'] == 1]
        plt.scatter(buy_signals.index, buy_signals['Price'], marker='^', color='g', s=100)
        
        # 标记卖出点
        sell_signals = self.positions[self.positions['Signal'] == -1]
        plt.scatter(sell_signals.index, sell_signals['Price'], marker='v', color='r', s=100)
        
        plt.title('黄金价格和交易信号')
        plt.ylabel('价格')
        plt.grid(True)
        
        # 绘制累计收益
        plt.subplot(2, 1, 2)
        plt.plot(self.positions.index, self.positions['Cumulative_Returns'])
        plt.title('策略累计收益')
        plt.ylabel('累计收益')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 使用Plotly绘制交互式图表
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, 
                               subplot_titles=('黄金价格和交易信号', '策略累计收益'))
            
            # 添加价格线
            fig.add_trace(go.Scatter(x=self.positions.index, y=self.positions['Price'],
                                    mode='lines', name='黄金价格'), row=1, col=1)
            
            # 添加均线
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['SMA10'],
                                    mode='lines', name='SMA10', line=dict(dash='dash', color='red')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['SMA30'],
                                    mode='lines', name='SMA30', line=dict(dash='dash', color='green')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['VWAP'],
                                    mode='lines', name='VWAP', line=dict(dash='dash', color='orange')), row=1, col=1)
            
            # 添加买入点
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Price'],
                                    mode='markers', name='买入信号',
                                    marker=dict(symbol='triangle-up', size=15, color='green')), row=1, col=1)
            
            # 添加卖出点
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Price'],
                                    mode='markers', name='卖出信号',
                                    marker=dict(symbol='triangle-down', size=15, color='red')), row=1, col=1)
            
            # 添加累计收益
            fig.add_trace(go.Scatter(x=self.positions.index, y=self.positions['Cumulative_Returns'],
                                    mode='lines', name='累计收益'), row=2, col=1)
            
            fig.update_layout(height=800, title_text="黄金交易策略回测结果",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            
            fig.show()
        except ImportError:
            print("Plotly未安装，跳过交互式图表")
    
    def analyze_time_patterns(self):
        """分析不同时间段的交易特征"""
        if self.data is None:
            return None
            
        # 添加时间特征
        self.data['Hour'] = self.data.index.hour
        self.data['Minute'] = self.data.index.minute
        self.data['Day'] = self.data.index.day_name()
        
        # 按小时分析波动率
        hourly_volatility = self.data.groupby('Hour')['Close'].pct_change().std()
        
        # 按小时分析交易量
        hourly_volume = self.data.groupby('Hour')['Volume'].mean()
        
        # 按星期几分析收益率
        daily_returns = self.data.groupby('Day')['Close'].pct_change().mean()
        
        # 可视化
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        hourly_volatility.plot(kind='bar', ax=ax1)
        ax1.set_title('不同小时的价格波动率')
        ax1.set_ylabel('波动率')
        ax1.grid(True, alpha=0.3)
        
        hourly_volume.plot(kind='bar', ax=ax2)
        ax2.set_title('不同小时的平均交易量')
        ax2.set_ylabel('交易量')
        ax2.grid(True, alpha=0.3)
        
        daily_returns.plot(kind='bar', ax=ax3)
        ax3.set_title('不同星期的平均收益率')
        ax3.set_ylabel('收益率')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            "hourly_volatility": hourly_volatility,
            "hourly_volume": hourly_volume,
            "daily_returns": daily_returns
        }
    
    def risk_management(self):
        """风险管理模块"""
        if self.data is None:
            return None
            
        # 计算每日波动率
        minute_volatility = self.data['Close'].pct_change().std()
        
        # 计算VaR (95%置信度)
        var_95 = minute_volatility * 1.65 * 10000  # 假设10000美元投资
        
        # 计算CVaR (条件风险价值)
        returns = self.data['Close'].pct_change().dropna()
        cvar_95 = returns[returns < -minute_volatility * 1.65].mean() * 10000
        
        # 基于ATR的止损计算
        avg_atr = self.data['ATR'].mean()
        suggested_stop_loss = 2 * avg_atr  # 2倍ATR作为止损
        
        # 计算最优仓位大小 (Kelly准则)
        if self.positions is not None:
            win_rate = len(self.positions[self.positions['Returns'] > 0]) / len(self.positions)
            avg_win = self.positions[self.positions['Returns'] > 0]['Returns'].mean()
            avg_loss = abs(self.positions[self.positions['Returns'] < 0]['Returns'].mean())
            kelly_fraction = (win_rate / avg_loss) - ((1 - win_rate) / avg_win) if avg_win > 0 and avg_loss > 0 else 0.1
            kelly_fraction = max(0, min(kelly_fraction, 0.5))  # 限制在0-50%之间
        else:
            kelly_fraction = 0.1  # 默认值
        
        print(f"分钟波动率: {minute_volatility:.4%}")
        print(f"95% VaR: ${var_95:.2f}")
        print(f"95% CVaR: ${cvar_95:.2f}")
        print(f"建议止损点: {suggested_stop_loss:.4f}")
        print(f"Kelly仓位比例: {kelly_fraction:.2%}")
        
        return {
            "minute_volatility": minute_volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "suggested_stop_loss": suggested_stop_loss,
            "kelly_fraction": kelly_fraction
        }
if __name__ == "__main__":
    gs = GoldTradingSystem(data_path="./data/combined.csv")

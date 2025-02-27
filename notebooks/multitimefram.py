import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

class MultiTimeframeVP:
    def __init__(self, price_precision=0.5):
        """
        初始化多时间框架成交量分布分析器
        
        参数:
            price_precision: 价格区间精度
        """
        self.price_precision = price_precision
        self.minute_data = []
        
        # 不同时间框架的数据存储
        self.timeframe_data = {
            '4H': defaultdict(list),
            'D': defaultdict(list),
            'W': defaultdict(list)
        }
        
        # VP计算结果
        self.vp_results = {
            '4H': {},
            'D': {},
            'W': {}
        }
    
    def add_minute_data(self, timestamp, vw, volume, open_price=None, close_price=None, high_price=None, low_price=None):
        """
        添加单条分钟K线数据
        
        参数:
            timestamp: 时间戳
            vw: 成交量加权平均价
            volume: 成交量
            open_price, close_price, high_price, low_price: OHLC数据(可选)
        """
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # 添加到原始数据列表
        self.minute_data.append({
            'timestamp': timestamp,
            'vw': vw,
            'volume': volume,
            'open': open_price,
            'close': close_price,
            'high': high_price,
            'low': low_price
        })
        
        # 直接分配到相应的时间框架
        self._assign_to_timeframes(timestamp, vw, volume)
    
    def _assign_to_timeframes(self, timestamp, vw, volume):
        """将单条数据分配到各个时间框架"""
        # 4小时框架
        h4_key = timestamp.floor('4H')
        self.timeframe_data['4H'][h4_key].append([vw, volume])
        
        # 日框架
        day_key = timestamp.floor('D')
        self.timeframe_data['D'][day_key].append([vw, volume])
        
        # 周框架 (以周一为起始)
        week_key = timestamp.to_period('W-MON').start_time
        self.timeframe_data['W'][week_key].append([vw, volume])
    
    def load_from_dataframe(self, df, timestamp_col='t', vw_col='vw', volume_col='v', 
                           open_col='o', close_col='c', high_col='h', low_col='l'):
        """
        从DataFrame批量加载K线数据
        
        参数:
            df: 包含分钟K线数据的DataFrame
            timestamp_col: 时间戳列名
            vw_col: 成交量加权平均价列名
            volume_col: 成交量列名
            open_col, close_col, high_col, low_col: OHLC列名
        """
        df = df.copy()
        if not isinstance(df[timestamp_col].iloc[0], pd.Timestamp):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # 确保数据按时间排序
        df = df.sort_values(timestamp_col)
        
        for _, row in df.iterrows():
            self.add_minute_data(
                row[timestamp_col],
                row[vw_col],
                row[volume_col],
                row.get(open_col) if open_col in df.columns else None,
                row.get(close_col) if close_col in df.columns else None,
                row.get(high_col) if high_col in df.columns else None,
                row.get(low_col) if low_col in df.columns else None
            )
    
    def calculate_volume_profile(self, timeframe='D', value_area_pct=0.7, recalculate_all=False):
        """
        计算指定时间框架的成交量分布
        
        参数:
            timeframe: 时间框架 ('4H', 'D', 'W')
            value_area_pct: 价值区域包含的成交量百分比
            recalculate_all: 是否重新计算所有周期
        """
        if timeframe not in self.timeframe_data:
            raise ValueError(f"不支持的时间框架: {timeframe}")
        
        # 确定需要计算的时间键
        time_keys = list(self.timeframe_data[timeframe].keys())
        if not recalculate_all and self.vp_results[timeframe]:
            # 只计算新的时间键
            time_keys = [k for k in time_keys if k not in self.vp_results[timeframe]]
        
        for time_key in time_keys:
            data = self.timeframe_data[timeframe][time_key]
            if not data:
                continue
            
            # 提取价格(vw)和成交量
            prices = [item[0] for item in data]
            volumes = [item[1] for item in data]
            
            # 确定价格范围
            min_price = min(prices)
            max_price = max(prices)
            
            # 创建价格区间
            price_range = np.arange(
                np.floor(min_price / self.price_precision) * self.price_precision,
                np.ceil(max_price / self.price_precision) * self.price_precision + self.price_precision,
                self.price_precision
            )
            
            # 计算每个价格区间的成交量
            volume_by_price = defaultdict(float)
            for price, volume in zip(prices, volumes):
                bin_idx = int((price - price_range[0]) / self.price_precision)
                if 0 <= bin_idx < len(price_range) - 1:
                    bin_low = price_range[bin_idx]
                    volume_by_price[bin_low] += volume
            
            # 转换为列表
            price_bins = sorted(volume_by_price.keys())
            volumes_list = [volume_by_price[price] for price in price_bins]
            
            # 计算POC (Point of Control)
            poc = None
            if price_bins:
                poc_idx = np.argmax(volumes_list)
                poc = price_bins[poc_idx]
            
            # 计算价值区域 (Value Area)
            value_area_low, value_area_high = None, None
            if price_bins and sum(volumes_list) > 0:
                # 按成交量排序
                sorted_idx = np.argsort(volumes_list)[::-1]
                total_volume = sum(volumes_list)
                cumulative_volume = 0
                value_area_bins = []
                
                for idx in sorted_idx:
                    value_area_bins.append(price_bins[idx])
                    cumulative_volume += volumes_list[idx]
                    if cumulative_volume >= total_volume * value_area_pct:
                        break
                
                if value_area_bins:
                    value_area_low = min(value_area_bins)
                    value_area_high = max(value_area_bins) + self.price_precision
            
            # 保存结果
            self.vp_results[timeframe][time_key] = {
                'price_bins': price_bins,
                'volumes': volumes_list,
                'poc': poc,
                'value_area': (value_area_low, value_area_high),
                'total_volume': sum(volumes_list) if volumes_list else 0
            }
    
    def calculate_all_profiles(self, recalculate_all=False):
        """计算所有时间框架的成交量分布"""
        for timeframe in self.timeframe_data.keys():
            self.calculate_volume_profile(timeframe, recalculate_all=recalculate_all)
    
    def plot_volume_profile(self, timeframe='D', n_periods=5, figsize=(15, 10)):
        """
        绘制指定时间框架的成交量分布
        
        参数:
            timeframe: 时间框架 ('4H', 'D', 'W')
            n_periods: 显示最近的几个周期
            figsize: 图表大小
        """
        # 确保VP已计算
        self.calculate_volume_profile(timeframe)
        
        # 获取最近的n个周期
        time_keys = sorted(self.vp_results[timeframe].keys())[-n_periods:]
        
        if not time_keys:
            print(f"没有{timeframe}时间框架的数据")
            return None
        
        # 创建子图
        fig, axes = plt.subplots(n_periods, 1, figsize=figsize, sharex=True)
        if n_periods == 1:
            axes = [axes]
        
        # 设置时间框架显示名称
        timeframe_names = {'4H': '4小时', 'D': '日', 'W': '周'}
        
        # 找出所有图表的价格范围
        all_prices = []
        for key in time_keys:
            if self.vp_results[timeframe][key]['price_bins']:
                all_prices.extend(self.vp_results[timeframe][key]['price_bins'])
        
        min_price = min(all_prices) if all_prices else 0
        max_price = max(all_prices) + self.price_precision if all_prices else 100
        
        for i, time_key in enumerate(time_keys):
            ax = axes[i]
            vp_data = self.vp_results[timeframe][time_key]
            
            if vp_data['price_bins']:
                # 绘制水平条形图
                bars = ax.barh(vp_data['price_bins'], vp_data['volumes'], 
                              height=self.price_precision*0.9, color='skyblue', alpha=0.7)
                
                # 标记POC
                if vp_data['poc'] is not None:
                    ax.axhline(y=vp_data['poc'], color='red', linestyle='-', linewidth=1,
                              label=f"POC: {vp_data['poc']:.2f}")
                
                # 标记价值区域
                if vp_data['value_area'][0] is not None and vp_data['value_area'][1] is not None:
                    va_low, va_high = vp_data['value_area']
                    ax.axhspan(va_low, va_high, alpha=0.2, color='green',
                              label=f"Value Area: {va_low:.2f} - {va_high:.2f}")
                
                # 统一Y轴范围
                ax.set_ylim(min_price, max_price)
                
                # 格式化时间显示
                if isinstance(time_key, pd.Timestamp):
                    time_str = time_key.strftime('%Y-%m-%d %H:%M')
                else:
                    time_str = str(time_key)
                
                ax.set_title(f"{timeframe_names.get(timeframe, timeframe)} VP - {time_str}")
                ax.set_xlabel('成交量')
                ax.set_ylabel('价格')
                ax.legend()
            else:
                ax.text(0.5, 0.5, '没有足够的数据', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_combined_profiles(self, figsize=(18, 12)):
        """绘制所有时间框架的成交量分布对比图"""
        # 确保所有VP已计算
        self.calculate_all_profiles()
        
        # 创建3行1列的子图
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 时间框架和对应的标题
        timeframes = [('4H', '4小时成交量分布'), ('D', '日成交量分布'), ('W', '周成交量分布')]
        
        # 找出所有图表的价格范围
        all_prices = []
        for timeframe in ['4H', 'D', 'W']:
            for vp_data in self.vp_results[timeframe].values():
                if vp_data['price_bins']:
                    all_prices.extend(vp_data['price_bins'])
        
        min_price = min(all_prices) if all_prices else 0
        max_price = max(all_prices) + self.price_precision if all_prices else 100
        
        for i, (timeframe, title) in enumerate(timeframes):
            ax = axes[i]
            
            # 获取最新的时间周期
            time_keys = sorted(self.vp_results[timeframe].keys())
            if not time_keys:
                ax.text(0.5, 0.5, f'没有{timeframe}时间框架的数据', 
                    ha='center', va='center', transform=ax.transAxes)
                continue
            
            latest_key = time_keys[-1]
            vp_data = self.vp_results[timeframe][latest_key]
            
            if vp_data['price_bins']:
                # 绘制水平条形图
                bars = ax.barh(vp_data['price_bins'], vp_data['volumes'], 
                            height=self.price_precision*0.9, color='skyblue', alpha=0.7)
                
                # 标记POC
                if vp_data['poc'] is not None:
                    ax.axhline(y=vp_data['poc'], color='red', linestyle='-', linewidth=1,
                            label=f"POC: {vp_data['poc']:.2f}")
                
                # 标记价值区域
                if vp_data['value_area'][0] is not None and vp_data['value_area'][1] is not None:
                    va_low, va_high = vp_data['value_area']
                    ax.axhspan(va_low, va_high, alpha=0.2, color='green',
                            label=f"Value Area: {va_low:.2f} - {va_high:.2f}")
                
                # 统一Y轴范围
                ax.set_ylim(min_price, max_price)
                
                # 格式化时间显示
                if isinstance(latest_key, pd.Timestamp):
                    time_str = latest_key.strftime('%Y-%m-%d %H:%M')
                else:
                    time_str = str(latest_key)
                
                ax.set_title(f"{title} - {time_str}")
                ax.set_xlabel('成交量')
                ax.set_ylabel('价格')
                ax.legend()
            else:
                ax.text(0.5, 0.5, '没有足够的数据', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    def test(self):
        pass



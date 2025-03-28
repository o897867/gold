import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

# 确保matplotlib不使用交互模式
plt.switch_backend('agg')

def identify_trend_direction(df, timeframe="1H"):
    """
    识别单一时间框架的趋势方向
    """
    # 计算必要的指标
    df = df.copy()
    
    # 1. 计算均线
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 2. 计算MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_line'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
    
    # 过滤掉NaN值
    df = df.dropna()
    
    if len(df) < 60:  # 确保有足够的数据来判断趋势
        return {
            "trend": "未知",
            "score": 0,
            "confidence": 0,
            "reasons": [f"{timeframe}数据不足，无法判断趋势"]
        }
    
    # 获取最新值
    latest = df.iloc[-1]
    
    # 判断趋势的理由
    reasons = []
    bull_points = 0
    bear_points = 0
    
    # 均线发散判断
    if latest['MA20'] > latest['MA60']:
        reasons.append(f"{timeframe} 20均线在60均线上方")
        bull_points += 1
    else:
        reasons.append(f"{timeframe} 20均线在60均线下方")
        bear_points += 1
    
    # 均线向上/向下发散
    ma20_slope = (df['MA20'].iloc[-1] - df['MA20'].iloc[-20]) / df['MA20'].iloc[-20] * 100
    ma60_slope = (df['MA60'].iloc[-1] - df['MA60'].iloc[-20]) / df['MA60'].iloc[-20] * 100
    
    if ma20_slope > 0 and ma60_slope > 0:
        reasons.append(f"{timeframe} 均线组向上运行 (MA20斜率: {ma20_slope:.2f}%, MA60斜率: {ma60_slope:.2f}%)")
        bull_points += 1
    elif ma20_slope < 0 and ma60_slope < 0:
        reasons.append(f"{timeframe} 均线组向下运行 (MA20斜率: {ma20_slope:.2f}%, MA60斜率: {ma60_slope:.2f}%)")
        bear_points += 1
    else:
        reasons.append(f"{timeframe} 均线组方向不一致，可能处于转折点")
    
    # MACD判断
    if latest['MACD_line'] > 0:
        reasons.append(f"{timeframe} MACD位于零轴上方，偏多头")
        bull_points += 1
    else:
        reasons.append(f"{timeframe} MACD位于零轴下方，偏空头")
        bear_points += 1
    
    # MACD金叉/死叉判断
    if df['MACD_line'].iloc[-2] < df['MACD_signal'].iloc[-2] and df['MACD_line'].iloc[-1] > df['MACD_signal'].iloc[-1]:
        reasons.append(f"{timeframe} MACD金叉，看涨信号")
        bull_points += 1.5
    elif df['MACD_line'].iloc[-2] > df['MACD_signal'].iloc[-2] and df['MACD_line'].iloc[-1] < df['MACD_signal'].iloc[-1]:
        reasons.append(f"{timeframe} MACD死叉，看跌信号")
        bear_points += 1.5
    
    # 价格高低点结构判断
    # 取最近的高点和低点
    recent_highs = []
    recent_lows = []
    
    for i in range(2, min(30, len(df)-2)):
        # 简单的局部高点判断
        if df['High'].iloc[-i] > df['High'].iloc[-i-1] and df['High'].iloc[-i] > df['High'].iloc[-i+1]:
            recent_highs.append((len(df)-i, df['High'].iloc[-i]))
            if len(recent_highs) >= 3:
                break
    
    for i in range(2, min(30, len(df)-2)):
        # 简单的局部低点判断
        if df['Low'].iloc[-i] < df['Low'].iloc[-i-1] and df['Low'].iloc[-i] < df['Low'].iloc[-i+1]:
            recent_lows.append((len(df)-i, df['Low'].iloc[-i]))
            if len(recent_lows) >= 3:
                break
    
    # 判断高点结构
    if len(recent_highs) >= 2:
        # 按时间排序
        recent_highs.sort(key=lambda x: x[0])
        if recent_highs[-1][1] > recent_highs[0][1]:
            reasons.append(f"{timeframe} 最近高点走高 ({recent_highs[0][1]:.2f} -> {recent_highs[-1][1]:.2f})")
            bull_points += 1
        elif recent_highs[-1][1] < recent_highs[0][1]:
            reasons.append(f"{timeframe} 最近高点走低 ({recent_highs[0][1]:.2f} -> {recent_highs[-1][1]:.2f})")
            bear_points += 1
    
    # 判断低点结构
    if len(recent_lows) >= 2:
        # 按时间排序
        recent_lows.sort(key=lambda x: x[0])
        if recent_lows[-1][1] > recent_lows[0][1]:
            reasons.append(f"{timeframe} 最近低点走高 ({recent_lows[0][1]:.2f} -> {recent_lows[-1][1]:.2f})")
            bull_points += 1
        elif recent_lows[-1][1] < recent_lows[0][1]:
            reasons.append(f"{timeframe} 最近低点走低 ({recent_lows[0][1]:.2f} -> {recent_lows[-1][1]:.2f})")
            bear_points += 1
    
    # 量价分析
    if 'Volume' in df.columns:
        # 最近两波上涨的量能比较
        up_waves = []
        for i in range(1, min(60, len(df)-1)):
            if df['Close'].iloc[-i] > df['Close'].iloc[-i-1]:
                # 收集上涨的波动
                wave_start = -i-1
                wave_end = -i
                curr_price = df['Close'].iloc[-i]
                prev_price = df['Close'].iloc[-i-1]
                price_change = (curr_price - prev_price) / prev_price * 100
                volume = df['Volume'].iloc[-i]
                up_waves.append((price_change, volume))
        
        if len(up_waves) >= 4:
            recent_up_vol = sum([wave[1] for wave in up_waves[:2]])
            prev_up_vol = sum([wave[1] for wave in up_waves[2:4]])
            
            if recent_up_vol > prev_up_vol * 1.2:
                reasons.append(f"{timeframe} 上涨量能增加，多头趋势增强")
                bull_points += 0.5
            elif recent_up_vol < prev_up_vol * 0.8:
                reasons.append(f"{timeframe} 上涨量能减少，多头趋势减弱")
                bear_points += 0.5
    
    # 总结判断
    total_points = bull_points + bear_points
    confidence = total_points / 7 * 100  # 假设最多7分，转换为百分比
    
    if bull_points > bear_points * 1.5:
        trend = "多头"
        score = bull_points - bear_points
    elif bear_points > bull_points * 1.5:
        trend = "空头"
        score = bear_points - bull_points
    else:
        trend = "震荡"
        score = 0
    
    return {
        "trend": trend,
        "score": score,
        "confidence": min(confidence, 100),
        "bull_points": bull_points,
        "bear_points": bear_points,
        "reasons": reasons
    }

def analyze_multi_timeframe_trend(df_dict):
    """
    分析多个时间框架的趋势，实现自上而下的趋势判断
    """
    # 确保时间框架从大到小排序
    timeframes = ["4H", "1H", "15T"]
    trend_results = {}
    
    # 分析每个时间框架
    for tf in timeframes:
        if tf in df_dict and len(df_dict[tf]) > 0:
            trend_results[tf] = identify_trend_direction(df_dict[tf], timeframe=tf)
        else:
            trend_results[tf] = {
                "trend": "未知",
                "score": 0,
                "confidence": 0,
                "reasons": [f"无{tf}数据"]
            }
    
    # 多周期趋势一致性判断
    trends_list = [result.get("trend") for result in trend_results.values()]
    
    # 如果所有时间框架都是多头趋势
    if all(trend == "多头" for trend in trends_list):
        overall_trend = "强烈多头趋势"
        trend_consistency = "高"
        recommendation = "可考虑顺势做多"
    # 如果所有时间框架都是空头趋势
    elif all(trend == "空头" for trend in trends_list):
        overall_trend = "强烈空头趋势"
        trend_consistency = "高"
        recommendation = "可考虑顺势做空"
    # 如果主要时间框架(4H)是多头，其他大部分也是多头
    elif trend_results.get("4H", {}).get("trend") == "多头" and trends_list.count("多头") >= 2:
        overall_trend = "偏多头趋势"
        trend_consistency = "中"
        recommendation = "偏向做多，注意设置止损"
    # 如果主要时间框架(4H)是空头，其他大部分也是空头
    elif trend_results.get("4H", {}).get("trend") == "空头" and trends_list.count("空头") >= 2:
        overall_trend = "偏空头趋势"
        trend_consistency = "中"
        recommendation = "偏向做空，注意设置止损"
    # 如果主要时间框架是震荡或时间框架之间不一致
    else:
        overall_trend = "不明确趋势/震荡市场"
        trend_consistency = "低"
        recommendation = "建议观望或轻仓操作"
    
    # 计算一个综合信心得分 (基于各时间框架的一致性和信心水平)
    weights = {"4H": 0.5, "1H": 0.3, "15T": 0.2}  # 权重分配，更重视大周期
    confidence_score = 0
    
    for tf, result in trend_results.items():
        if tf in weights:
            confidence_score += result.get("confidence", 0) * weights.get(tf, 0)
    
    # 提高多周期一致性时的信心
    if trends_list.count(trends_list[0]) == len(trends_list) and trends_list[0] != "震荡":
        confidence_score *= 1.2
    
    return {
        "overall_trend": overall_trend,
        "trend_consistency": trend_consistency,
        "confidence_score": min(confidence_score, 100),
        "recommendation": recommendation,
        "timeframe_analysis": trend_results
    }

def resample_to_multiple_timeframes(df):
    """
    将原始数据重采样为多个时间框架
    """
    # 确保索引是DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 创建不同时间框架的数据
    print(f"Original data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    result = {}
    
    # 将原始数据作为最小时间框架
    result["原始"] = df.copy()
    
    # 重采样为15分钟
    try:
        result["15T"] = df.resample('15T').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    except Exception as e:
        print(f"15分钟重采样错误: {e}")
        result["15T"] = pd.DataFrame()
    
    # 重采样为1小时
    try:
        result["1H"] = df.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    except Exception as e:
        print(f"1小时重采样错误: {e}")
        result["1H"] = pd.DataFrame()
    
    # 重采样为4小时
    try:
        result["4H"] = df.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    except Exception as e:
        print(f"4小时重采样错误: {e}")
        result["4H"] = pd.DataFrame()
    for tf, resampled_df in result.items():
        if not resampled_df.empty:
            print(f"Resampled {tf} shape: {resampled_df.shape}")
            print(f"Resampled {tf} date range: {resampled_df.index.min()} to {resampled_df.index.max()}")

    
    return result

def create_trend_visualization(df_dict, trend_analysis):
    """
    创建趋势可视化图表
    """
    # 创建多子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    timeframes = ["4H", "1H", "15T"]
    
    for i, tf in enumerate(timeframes):
        if tf in df_dict and len(df_dict[tf]) > 0:
            df = df_dict[tf]
            ax = axes[i]
            
            # 计算均线
            if 'MA20' not in df.columns:
                df['MA20'] = df['Close'].rolling(window=20).mean()
            if 'MA60' not in df.columns:
                df['MA60'] = df['Close'].rolling(window=60).mean()
            
            # 获取最近100个数据点用于可视化
            recent_df = df.iloc[-100:] if len(df) > 100 else df
            
            # 绘制蜡烛图（简化版）
            ax.plot(recent_df.index, recent_df['Close'], color='black', label='Close')
            
            # 绘制均线
            ax.plot(recent_df.index, recent_df['MA20'], color='blue', label='MA20')
            ax.plot(recent_df.index, recent_df['MA60'], color='red', label='MA60')
            
            # 设置标题和标签
            trend_result = trend_analysis["timeframe_analysis"].get(tf, {})
            trend = trend_result.get("trend", "未知")
            confidence = trend_result.get("confidence", 0)
            
            ax.set_title(f"{tf} 时间框架 - 趋势: {trend} (信心: {confidence:.1f}%)")
            ax.set_xlabel("日期")
            ax.set_ylabel("价格")
            ax.legend()
            ax.grid(True)
    
    # 添加总体趋势分析
    plt.figtext(0.5, 0.01, 
                f"总体趋势: {trend_analysis['overall_trend']} | 一致性: {trend_analysis['trend_consistency']} | 信心分数: {trend_analysis['confidence_score']:.1f}%\n建议: {trend_analysis['recommendation']}", 
                ha="center", 
                fontsize=14, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # 将图表转换为base64字符串
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def generate_trend_report(df):
    """
    生成完整的趋势分析报告
    """
    # 重采样数据到多个时间框架
    df_dict = resample_to_multiple_timeframes(df)
    
    # 分析多时间框架趋势
    trend_analysis = analyze_multi_timeframe_trend(df_dict)
    
    # 创建可视化
    visualization = create_trend_visualization(df_dict, trend_analysis)
    
    return {
        "trend_analysis": trend_analysis,
        "visualization": visualization
    }
def calculate_stop_loss(entry_price, direction, risk_percentage, account_size, leverage=500, position_percentage=None):
    """
    计算止损价格和仓位大小
    
    参数:
    entry_price (float): 入场价格
    direction (str): 交易方向 ('long' 或 'short')
    risk_percentage (float): 账户风险百分比 (例如 1 表示 1%)
    account_size (float): 账户总资金
    leverage (int): 杠杆倍数，默认500倍
    position_percentage (float, optional): 仓位比例百分比 (例如 10 表示 10%)，为None时将根据风险计算
    
    返回:
    dict: 包含止损价格、仓位大小和潜在损失的字典
    """
    max_risk_amount = account_size * (risk_percentage / 100)
    
    if position_percentage is not None:
        # 使用固定比例计算仓位大小
        position_size = account_size * (position_percentage / 100)
        
        # 反向计算止损价格
        if direction.lower() == 'long':
            # 多头: 止损价格 = 入场价格 - (入场价格 * 可亏损比例)
            # 可亏损比例 = 最大风险金额 / (仓位大小 * 杠杆)
            max_loss_percentage = max_risk_amount / (position_size * leverage)
            stop_loss_price = entry_price * (1 - max_loss_percentage)
        else:
            # 空头: 止损价格 = 入场价格 + (入场价格 * 可亏损比例)
            max_loss_percentage = max_risk_amount / (position_size * leverage)
            stop_loss_price = entry_price * (1 + max_loss_percentage)
    else:
        # 根据风险计算止损距离和仓位大小
        # 固定止损比例 (可根据波动性调整)
        if direction.lower() == 'long':
            # 多头: 止损在入场价下方
            stop_loss_percentage = 0.5  # 0.5% 止损距离，可根据实际情况调整
            stop_loss_price = entry_price * (1 - stop_loss_percentage / 100)
            
            # 计算仓位大小
            price_diff_percentage = (entry_price - stop_loss_price) / entry_price * 100
            # 最大亏损金额 / (价格变动百分比 * 杠杆 / 100)
            position_size = max_risk_amount / (price_diff_percentage * leverage / 100)
        else:
            # 空头: 止损在入场价上方
            stop_loss_percentage = 0.5  # 0.5% 止损距离，可根据实际情况调整
            stop_loss_price = entry_price * (1 + stop_loss_percentage / 100)
            
            # 计算仓位大小
            price_diff_percentage = (stop_loss_price - entry_price) / entry_price * 100
            position_size = max_risk_amount / (price_diff_percentage * leverage / 100)
    
    # 计算潜在损失金额
    if direction.lower() == 'long':
        potential_loss = position_size * leverage * ((entry_price - stop_loss_price) / entry_price)
    else:
        potential_loss = position_size * leverage * ((stop_loss_price - entry_price) / entry_price)
    
    # 计算杠杆下的合约数量
    contract_price = entry_price
    contract_size = position_size * leverage / contract_price
    
    return {
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "position_size": position_size,
        "contract_size": contract_size,
        "potential_loss": potential_loss,
        "max_risk_percentage": risk_percentage,
        "max_risk_amount": max_risk_amount,
        "leverage": leverage,
        "direction": direction
    }

def calculate_dynamic_position_sizing(df, entry_price, direction, risk_percentage, account_size, atr_periods=14, atr_multiplier=1.5, leverage=500):
    """
    基于ATR (平均真实波幅) 计算动态仓位和止损
    
    参数:
    df (DataFrame): 包含价格数据的DataFrame
    entry_price (float): 入场价格
    direction (str): 交易方向 ('long' 或 'short')
    risk_percentage (float): 账户风险百分比 (例如 1 表示 1%)
    account_size (float): 账户总资金
    atr_periods (int): 计算ATR的周期数
    atr_multiplier (float): ATR乘数用于确定止损距离
    leverage (int): 杠杆倍数
    
    返回:
    dict: 包含止损价格、仓位大小和风险信息的字典
    """
    # 计算ATR
    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_periods).mean().iloc[-1]
    
    # 计算止损距离 (ATR的倍数)
    stop_distance = atr * atr_multiplier
    
    # 根据方向设置止损价格
    if direction.lower() == 'long':
        stop_loss_price = entry_price - stop_distance
    else:
        stop_loss_price = entry_price + stop_distance
    
    # 计算风险金额
    max_risk_amount = account_size * (risk_percentage / 100)
    
    # 计算价格变动百分比
    if direction.lower() == 'long':
        price_diff_percentage = (entry_price - stop_loss_price) / entry_price * 100
    else:
        price_diff_percentage = (stop_loss_price - entry_price) / entry_price * 100
    
    # 计算仓位大小
    position_size = max_risk_amount / (price_diff_percentage * leverage / 100)
    
    # 计算潜在损失
    if direction.lower() == 'long':
        potential_loss = position_size * leverage * ((entry_price - stop_loss_price) / entry_price)
    else:
        potential_loss = position_size * leverage * ((stop_loss_price - entry_price) / entry_price)
    
    # 计算杠杆下的合约数量
    contract_price = entry_price
    contract_size = position_size * leverage / contract_price
    
    return {
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "position_size": position_size,
        "contract_size": contract_size,
        "potential_loss": potential_loss,
        "max_risk_percentage": risk_percentage,
        "max_risk_amount": max_risk_amount,
        "atr_value": atr,
        "leverage": leverage,
        "direction": direction
    }


# 添加到Flas
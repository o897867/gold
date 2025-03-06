# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64
# import scipy.stats as st

# def calc_atr(df, window=14):
#     """
#     简易ATR
#     """
#     df = df.copy()
#     df["H-L"] = df["High"] - df["Low"]
#     df["H-C"] = (df["High"] - df["Close"].shift(1)).abs()
#     df["L-C"] = (df["Low"] - df["Close"].shift(1)).abs()
#     df["TR"] = df[["H-L","H-C","L-C"]].max(axis=1)
#     df["ATR"] = df["TR"].rolling(window=window).mean()
#     return df["ATR"]

# def calc_pivots(df):
#     """
#     简单示例：按日计算Pivot、R1、S1
#     """
#     daily = df.resample('1D').agg({"High": "max", "Low": "min", "Close":"last"})
#     daily["Pivot"] = (daily["High"] + daily["Low"] + daily["Close"])/3
#     daily["R1"] = 2*daily["Pivot"] - daily["Low"]
#     daily["S1"] = 2*daily["Pivot"] - daily["High"]
#     return daily

# def volume_profile(df, bins=50):
#     """
#     生成简单Volume Profile
#     """
#     price_min = df["Low"].min()
#     price_max = df["High"].max()
#     prices = df["Close"].values
#     vols = df["Volume"].values
#     hist, bin_edges = np.histogram(prices, bins=bins, range=(price_min, price_max), weights=vols)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     return bin_centers, hist

# def calc_normality_test(log_returns):
#     """
#     使用 Jarque-Bera 检验对 log_returns 进行正态性检测
#     """
#     jb_stat, jb_pvalue = st.jarque_bera(log_returns.dropna())
#     is_normal = jb_pvalue > 0.05  # 以 5% 显著性水平判断
#     return {
#         "jb_stat": jb_stat,
#         "jb_pvalue": jb_pvalue,
#         "is_normal": is_normal
#     }

# def normality_test_for_day(df, day):
#     """
#     对指定日期的数据进行对数收益率正态性检测
    
#     参数:
#       df: DataFrame，其中索引必须是 datetime 格式，并且包含 "Close" 列
#       day: 日期字符串，例如 '2023-10-10' 或者 datetime 对象
      
#     返回:
#       正态性检测的结果字典
#     """
#     # 筛选指定日期的数据
#     df_day = df.loc[str(day)]
    
#     # 计算对数收益率
#     df_day = df_day.copy()  # 防止修改原始 DataFrame
#     df_day['log_ret'] = np.log(df_day['Close'] / df_day['Close'].shift(1))
    
#     # 进行正态性检测
#     result = calc_normality_test(df_day['log_ret'])
#     return result
# def normality_test_vp(df, day, bins=50):
#     """
#     对指定日期内的 VP（成交量分布）数据做正态性检测
    
#     参数:
#       df: DataFrame，其中索引为 datetime 格式，包含 'Close', 'Low', 'High', 'Volume' 字段
#       day: 指定日期，格式如 '2023-10-10'
#       bins: 分箱数量
      
#     返回:
#       正态性检测结果字典
#     """
#     # 筛选出指定日期的数据
#     df_day = df.loc[str(day)]
    
#     # 计算 Volume Profile
#     bin_centers, vol_hist = volume_profile(df_day, bins=bins)
    
#     # 将成交量直方图数据转换为 Series 进行检验
#     vol_hist_series = pd.Series(vol_hist)
    
#     # 调用已有的正态性检测函数（这里使用 Jarque-Bera 检验）
#     result = calc_normality_test(vol_hist_series)
#     return result

# # 使用示例：
# # normality_vp = normality_test_vp(df, '2023-10-10', bins=50)
# # print("VP 正态性检测结果：", normality_vp)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import scipy.stats as st
from scipy.signal import find_peaks
from arch import arch_model



########################################
#               原有函数
########################################

def calc_atr(df, window=14):
    """
    计算 ATR（Average True Range）。
    
    优化思路：
      1. 先复制 DataFrame，防止修改原对象。
      2. 提供可选加权/指数平滑计算方式（此处维持简单移动平均）。
      3. 返回整个 df 以便保留 ATR 列做后续分析。
    """
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-C"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L","H-C","L-C"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=window).mean()
    return df

def calc_pivots(df):
    """
    简单示例：按日计算Pivot、R1、S1。
    
    优化思路：
      1. 返回新的 DataFrame，并保留原信息（可用 merge 方式）。
      2. 提供更多支撑阻力（R2、S2 等）。此处不做过多展开。
    """
    daily = df.resample('1D').agg({"High": "max", "Low": "min", "Close":"last"})
    daily["Pivot"] = (daily["High"] + daily["Low"] + daily["Close"])/3
    daily["R1"] = 2 * daily["Pivot"] - daily["Low"]
    daily["S1"] = 2 * daily["Pivot"] - daily["High"]
    return daily

def volume_profile(df, bins=50):
    """
    生成简单 Volume Profile。
    
    优化思路：
      1. 可以改用价格区间等宽或不等宽分箱。
      2. 同时返回对应的价格区间与成交量分布，用于绘图或进一步统计。
    """
    price_min = df["Low"].min()
    price_max = df["High"].max()
    prices = df["Close"].values
    vols = df["Volume"].values
    hist, bin_edges = np.histogram(prices, bins=bins, range=(price_min, price_max), weights=vols)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

def calc_normality_test(log_returns):
    """
    使用 Jarque-Bera 检验对 log_returns 进行正态性检测。
    
    返回:
      {
        "jb_stat": float,
        "jb_pvalue": float,
        "is_normal": bool
      }
    """
    jb_stat, jb_pvalue = st.jarque_bera(log_returns.dropna())
    is_normal = jb_pvalue > 0.05  # 以 5% 显著性水平判断
    return {
        "jb_stat": jb_stat,
        "jb_pvalue": jb_pvalue,
        "is_normal": is_normal
    }

def normality_test_for_day(df, day):
    """
    对指定日期的数据进行对数收益率正态性检测。
    
    参数:
      df: DataFrame，其中索引必须是 datetime 格式，并且包含 "Close" 列
      day: 日期字符串，例如 '2023-10-10' 或者 datetime 对象
    """
    df_day = df.loc[str(day)].copy()
    df_day['log_ret'] = np.log(df_day['Close'] / df_day['Close'].shift(1))
    result = calc_normality_test(df_day['log_ret'])
    return result

def normality_test_vp(df, day, bins=50):
    """
    对指定日期内的 VP（成交量分布）数据做正态性检测。
    
    参数:
      df: DataFrame，其中索引为 datetime 格式，包含 'Close', 'Low', 'High', 'Volume' 字段
      day: 指定日期，格式如 '2023-10-10'
      bins: 分箱数量
    """
    df_day = df.loc[str(day)]
    bin_centers, vol_hist = volume_profile(df_day, bins=bins)
    vol_hist_series = pd.Series(vol_hist)
    result = calc_normality_test(vol_hist_series)
    return result


########################################
#               新增函数
########################################

def calc_ma(df, periods=[20, 50], price_col='Close'):
    """
    计算多条移动平均线(MA)并将其合并到原 DataFrame。
    
    参数:
      df: 带有 price_col 列的 DataFrame
      periods: 列表，包含所需的 MA 周期
      price_col: 用于计算MA的价格列
    
    返回:
      df: 包含新列 MA_{period} 的 DataFrame
    """
    df = df.copy()
    for p in periods:
        df[f"MA_{p}"] = df[price_col].rolling(window=p).mean()
    return df

def calc_macd(df, fastperiod=12, slowperiod=26, signalperiod=9, price_col='Close'):
    """
    计算 MACD 指标并返回包含 MACD 主线、信号线、柱状图的 DataFrame。
    
    参数:
      df: 包含 price_col 的 DataFrame
      fastperiod: MACD 快线周期，默认12
      slowperiod: MACD 慢线周期，默认26
      signalperiod: 信号线周期，默认9
      price_col: 用于计算MACD的价格列
    
    返回:
      df: 新增 [MACD_line, MACD_signal, MACD_hist] 列
    """
    df = df.copy()
    # EMA 计算
    df['EMA_fast'] = df[price_col].ewm(span=fastperiod, adjust=False).mean()
    df['EMA_slow'] = df[price_col].ewm(span=slowperiod, adjust=False).mean()
    # MACD 主线
    df['MACD_line'] = df['EMA_fast'] - df['EMA_slow']
    # 信号线
    df['MACD_signal'] = df['MACD_line'].ewm(span=signalperiod, adjust=False).mean()
    # 柱状图
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
    # 清理临时列
    df.drop(['EMA_fast','EMA_slow'], axis=1, inplace=True)
    return df

def calc_rsi(df, period=14, price_col='Close'):
    """
    计算 RSI 指标（相对强弱指数）。
    
    参数:
      df: 包含 price_col 的 DataFrame
      period: RSI周期
      price_col: 用于计算RSI的价格列
    
    返回:
      df: 新增 RSI 列
    """
    df = df.copy()
    change = df[price_col].diff()
    gain = change.where(change > 0, 0.0)
    loss = -change.where(change < 0, 0.0)
    
    # EMA 方式平滑
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calc_bollinger_bands(df, window=20, num_std=2, price_col='Close'):
    """
    计算布林带（Bollinger Bands）。
    
    参数:
      df: 包含 price_col 的 DataFrame
      window: 计算布林带的时间窗口
      num_std: 标准差倍数
      price_col: 用于计算布林带的价格列
    
    返回:
      df: 新增 Bollinger_mid, Bollinger_up, Bollinger_down 列
    """
    df = df.copy()
    df['Bollinger_mid'] = df[price_col].rolling(window=window).mean()
    rolling_std = df[price_col].rolling(window=window).std()
    df['Bollinger_up'] = df['Bollinger_mid'] + num_std * rolling_std
    df['Bollinger_down'] = df['Bollinger_mid'] - num_std * rolling_std
    return df

def calc_zscore(df, window=20, price_col='Close'):
    """
    计算价格相对于移动均值的Z-score，用于衡量价格偏离程度。
    Z-score = (price - MA) / Std
    
    参数:
      df: 包含 price_col 的 DataFrame
      window: 计算均值和标准差的窗口
      price_col: 用于计算Z-score的价格列
    
    返回:
      df: 新增 Zscore 列
    """
    df = df.copy()
    mean = df[price_col].rolling(window=window).mean()
    std = df[price_col].rolling(window=window).std()
    df['Zscore'] = (df[price_col] - mean) / (std + 1e-8)  # 避免除以0
    return df

def calc_vol_delta_naive(df):
    """
    基于收盘价涨跌的极简 Volume Delta。
    说明：真正的 Volume Delta 通常需要逐笔交易或更细粒度数据。
    这里仅做示例：若本K线收盘大于开盘，则把该K线全部成交量算作买入量；反之算作卖出量。
    
    返回:
      df: 新增 BuyVol、SellVol、VolumeDelta 列
    """
    df = df.copy()
    df['BuyVol'] = np.where(df['Close'] > df['Open'], df['Volume'], 0)
    df['SellVol'] = np.where(df['Close'] <= df['Open'], df['Volume'], 0)
    df['VolumeDelta'] = df['BuyVol'] - df['SellVol']
    return df

def calc_atr_optimized(df, window=14, method='sma'):
    """
    进一步优化的 ATR 计算，可选择不同平滑方式（sma 或 ewm）。
    
    参数:
      df: 包含 High/Low/Close 列的 DataFrame
      window: ATR 计算周期
      method: 平滑方式: 'sma' 或 'ewm'
    
    返回:
      df: 新增 ATR_optimized 列
    """
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-C"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L","H-C","L-C"]].max(axis=1)
    
    if method == 'sma':
        df["ATR_optimized"] = df["TR"].rolling(window=window).mean()
    elif method == 'ewm':
        df["ATR_optimized"] = df["TR"].ewm(span=window, adjust=False).mean()
    else:
        raise ValueError("method 参数必须为 'sma' 或 'ewm'")
    
    return df
def calc_garch_volatility(df, p=1, q=1):
    """
    使用 GARCH(p, q) 模型计算波动率。
    
    参数:
      df: DataFrame，必须包含 'Close' 列，索引为 datetime 格式。
      p: ARCH 模型的滞后阶数，默认 1
      q: GARCH 模型的滞后阶数，默认 1
      
    返回:
      df: 原始 DataFrame，新增 'log_ret' 和 'GARCH_vol' 列，
          'log_ret' 为对数收益率，
          'GARCH_vol' 为基于 GARCH 模型计算的条件波动率（标准差）。
      res: GARCH 模型的拟合结果对象，便于进一步分析。
    """
    df = df.copy()
    # 计算对数收益率（可以乘以100以便于模型拟合，但这里保持原尺度）
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()

    # 定义 GARCH 模型：采用常数均值模型，残差服从正态分布
    # 这里将对数收益率乘以 100，提高模型的数值稳定性（可选）
    am = arch_model(df["log_ret"] * 100, vol="Garch", p=p, q=q, mean="Constant", dist="normal")
    res = am.fit(disp="off")
    
    # 模型拟合后，计算条件波动率，并转换回原尺度
    df["GARCH_vol"] = res.conditional_volatility / 100.0
    return df, res


########################################
#    将各指标整合到 DataFrame 的示例
########################################

def calculate_all_indicators(df):
    """
    统一调用上面定义的指标计算函数，整合到同一个 DataFrame 中。
    根据你的需求，可选不同的参数。
    
    返回:
      df: 包含多种技术指标的新 DataFrame
    """
    # 先将 df 复制，避免修改原始数据
    df = df.copy()
    # Garch波动率
    df,garch_result = calc_garch_volatility(df, p=1, q=1)
    
    # 1) ATR（简单） + 优化版ATR（二选一或者都留着做对比）
    df = calc_atr(df, window=14)  # 简易 ATR
    df = calc_atr_optimized(df, window=14, method='ewm')  # 优化版 ATR
    
    # 2) 移动平均线 MA
    df = calc_ma(df, periods=[20, 50], price_col='Close')
    
    # 3) MACD
    df = calc_macd(df, fastperiod=12, slowperiod=26, signalperiod=9, price_col='Close')
    
    # 4) RSI
    df = calc_rsi(df, period=14, price_col='Close')
    
    # 5) 布林带
    df = calc_bollinger_bands(df, window=20, num_std=2, price_col='Close')
    
    # 6) Z-score（正态分布偏离度）
    df = calc_zscore(df, window=20, price_col='Close')
    
    # 7) Volume Delta（naive版本，仅示例）
    df = calc_vol_delta_naive(df)
    
    return df




########################################
#         如何使用/测试
########################################

def calc_multi_tf_support_resistance_with_volume(df, timeframes=["15T", "1H", "4H", "1D"]):
    """
    针对15分钟、1小时、4小时和日线周期，计算支撑阻力位，同时结合成交量计算 VWAP。
    
    对于每个周期：
      - 重采样计算周期内的最高价、最低价、最后收盘价及总成交量
      - 计算经典 Pivot Point 公式：
          Pivot = (High + Low + Close) / 3
          R1 = 2 * Pivot - Low
          S1 = 2 * Pivot - High
          R2 = Pivot + (High - Low)
          S2 = Pivot - (High - Low)
      - 计算 VWAP (Volume Weighted Average Price)：
          VWAP = (sum(Close * Volume) / sum(Volume)) 对于该周期
      
    参数:
      df: DataFrame，必须包含 'High', 'Low', 'Close', 'Volume' 列，且索引为 pd.DatetimeIndex
      timeframes: list，周期字符串，默认为 ["15T", "1H", "4H", "1D"]
      
    返回:
      results: dict，键为周期，值为对应周期的支撑阻力和 VWAP 数据（DataFrame）
    """
    results = {}
    for tf in timeframes:
        # 重采样：计算每个周期内的最高价、最低价、最后的收盘价及总成交量
        resampled = df.resample(tf).agg({
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()
        
        # 计算经典Pivot及支撑阻力位
        resampled["Pivot"] = (resampled["High"] + resampled["Low"] + resampled["Close"]) / 3
        resampled["R1"] = 2 * resampled["Pivot"] - resampled["Low"]
        resampled["S1"] = 2 * resampled["Pivot"] - resampled["High"]
        resampled["R2"] = resampled["Pivot"] + (resampled["High"] - resampled["Low"])
        resampled["S2"] = resampled["Pivot"] - (resampled["High"] - resampled["Low"])
        
        # 计算VWAP：利用周期内所有K线的成交量加权平均价格
        # 注意，这里用到整个周期内 Close * Volume 的和除以总成交量
        vwap = (df["Close"] * df["Volume"]).resample(tf).sum() / resampled["Volume"]
        resampled["VWAP"] = vwap
        
        results[tf] = resampled
    return results

# 示例用法：
# 假设 df 为包含 OHLCV 数据的 DataFrame，且索引为 pd.DatetimeIndex
# multi_tf_results = calc_multi_tf_support_resistance_with_volume(df)
# 打印日线周期的支撑阻力及 VWAP 数据：
# print(multi_tf_results["1D"].tail())


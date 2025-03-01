import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import scipy.stats as st

def calc_atr(df, window=14):
    """
    简易ATR
    """
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-C"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L","H-C","L-C"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=window).mean()
    return df["ATR"]

def calc_pivots(df):
    """
    简单示例：按日计算Pivot、R1、S1
    """
    daily = df.resample('1D').agg({"High": "max", "Low": "min", "Close":"last"})
    daily["Pivot"] = (daily["High"] + daily["Low"] + daily["Close"])/3
    daily["R1"] = 2*daily["Pivot"] - daily["Low"]
    daily["S1"] = 2*daily["Pivot"] - daily["High"]
    return daily

def volume_profile(df, bins=50):
    """
    生成简单Volume Profile
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
    使用 Jarque-Bera 检验对 log_returns 进行正态性检测
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
    对指定日期的数据进行对数收益率正态性检测
    
    参数:
      df: DataFrame，其中索引必须是 datetime 格式，并且包含 "Close" 列
      day: 日期字符串，例如 '2023-10-10' 或者 datetime 对象
      
    返回:
      正态性检测的结果字典
    """
    # 筛选指定日期的数据
    df_day = df.loc[str(day)]
    
    # 计算对数收益率
    df_day = df_day.copy()  # 防止修改原始 DataFrame
    df_day['log_ret'] = np.log(df_day['Close'] / df_day['Close'].shift(1))
    
    # 进行正态性检测
    result = calc_normality_test(df_day['log_ret'])
    return result
def normality_test_vp(df, day, bins=50):
    """
    对指定日期内的 VP（成交量分布）数据做正态性检测
    
    参数:
      df: DataFrame，其中索引为 datetime 格式，包含 'Close', 'Low', 'High', 'Volume' 字段
      day: 指定日期，格式如 '2023-10-10'
      bins: 分箱数量
      
    返回:
      正态性检测结果字典
    """
    # 筛选出指定日期的数据
    df_day = df.loc[str(day)]
    
    # 计算 Volume Profile
    bin_centers, vol_hist = volume_profile(df_day, bins=bins)
    
    # 将成交量直方图数据转换为 Series 进行检验
    vol_hist_series = pd.Series(vol_hist)
    
    # 调用已有的正态性检测函数（这里使用 Jarque-Bera 检验）
    result = calc_normality_test(vol_hist_series)
    return result

# 使用示例：
# normality_vp = normality_test_vp(df, '2023-10-10', bins=50)
# print("VP 正态性检测结果：", normality_vp)

import pandas as pd
import numpy as np
import scipy.stats as st
from arch import arch_model

def volume_profile(df, bins=50):
    """
    生成简单 Volume Profile。
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
    """
    jb_stat, jb_pvalue = st.jarque_bera(log_returns.dropna())
    is_normal = jb_pvalue > 0.05  # 以 5% 显著性水平判断
    return {
        "jb_stat": jb_stat,
        "jb_pvalue": jb_pvalue,
        "is_normal": is_normal
    }

def calc_pivots(df):
    """
    简单示例：按日计算Pivot、R1、S1。
    """
    daily = df.resample('1D').agg({"High": "max", "Low": "min", "Close":"last"})
    daily["Pivot"] = (daily["High"] + daily["Low"] + daily["Close"])/3
    daily["R1"] = 2 * daily["Pivot"] - daily["Low"]
    daily["S1"] = 2 * daily["Pivot"] - daily["High"]
    return daily

def normality_test_vp(df, day, bins=50):
    """
    对指定日期内的 VP（成交量分布）数据做正态性检测。
    """
    df_day = df.loc[str(day)]
    bin_centers, vol_hist = volume_profile(df_day, bins=bins)
    vol_hist_series = pd.Series(vol_hist)
    result = calc_normality_test(vol_hist_series)
    return result

def calc_ma(df, periods=[20, 50], price_col='Close'):
    """
    计算多条移动平均线(MA)。
    """
    df = df.copy()
    for p in periods:
        df[f"MA_{p}"] = df[price_col].rolling(window=p).mean()
    return df

def calc_macd(df, fastperiod=12, slowperiod=26, signalperiod=9, price_col='Close'):
    """
    计算 MACD 指标。
    """
    df = df.copy()
    df['EMA_fast'] = df[price_col].ewm(span=fastperiod, adjust=False).mean()
    df['EMA_slow'] = df[price_col].ewm(span=slowperiod, adjust=False).mean()
    df['MACD_line'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD_line'].ewm(span=signalperiod, adjust=False).mean()
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
    df.drop(['EMA_fast','EMA_slow'], axis=1, inplace=True)
    return df

def calc_rsi(df, period=14, price_col='Close'):
    """
    计算 RSI 指标。
    """
    df = df.copy()
    change = df[price_col].diff()
    gain = change.where(change > 0, 0.0)
    loss = -change.where(change < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calc_bollinger_bands(df, window=20, num_std=2, price_col='Close'):
    """
    计算布林带。
    """
    df = df.copy()
    df['Bollinger_mid'] = df[price_col].rolling(window=window).mean()
    rolling_std = df[price_col].rolling(window=window).std()
    df['Bollinger_up'] = df['Bollinger_mid'] + num_std * rolling_std
    df['Bollinger_down'] = df['Bollinger_mid'] - num_std * rolling_std
    return df

def calc_zscore(df, window=20, price_col='Close'):
    """
    计算价格Z-score。
    """
    df = df.copy()
    mean = df[price_col].rolling(window=window).mean()
    std = df[price_col].rolling(window=window).std()
    df['Zscore'] = (df[price_col] - mean) / (std + 1e-8)
    return df

def calc_vol_delta_naive(df):
    """
    基于收盘价涨跌的极简 Volume Delta。
    """
    df = df.copy()
    df['BuyVol'] = np.where(df['Close'] > df['Open'], df['Volume'], 0)
    df['SellVol'] = np.where(df['Close'] <= df['Open'], df['Volume'], 0)
    df['VolumeDelta'] = df['BuyVol'] - df['SellVol']
    return df

# def calc_garch_volatility(df, p=1, q=1):
#     """
#     使用 GARCH 模型计算波动率。
#     """
#     df = df.copy()
#     df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
#     df = df.dropna()
#     am = arch_model(df["log_ret"] * 100, vol="Garch", p=p, q=q, mean="Constant", dist="normal")
#     res = am.fit(disp="off")
#     df["GARCH_vol"] = res.conditional_volatility / 100.0
#     return df, res

def calc_atr_optimized(df, window=14, method='sma'):
    """
    优化版 ATR 计算。
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

def calculate_all_indicators(df):
    """
    统一调用指标计算函数。
    """
    df = df.copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Replace GARCH with simple volatility measure
    df["volatility"] = df["log_ret"].rolling(window=35).std()
    df = calc_atr_optimized(df, window=14, method='ewm')
    df = calc_ma(df, periods=[20, 50], price_col='Close')
    df = calc_macd(df, fastperiod=12, slowperiod=26, signalperiod=9, price_col='Close')
    df = calc_rsi(df, period=14, price_col='Close')
    df = calc_bollinger_bands(df, window=20, num_std=2, price_col='Close')
    df = calc_zscore(df, window=20, price_col='Close')
    df = calc_vol_delta_naive(df)
    return df

def calc_multi_tf_support_resistance_with_volume(df, timeframes=["15T", "1H", "4H", "1D"]):
    """
    计算多时间框架支撑阻力位和VWAP。
    """
    results = {}
    for tf in timeframes:
        try:
            resampled = df.resample(tf).agg({
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()
            
            if len(resampled) == 0:
                print(f"No data after resampling for timeframe {tf}")
                results[tf] = {}
                continue
            
            resampled["Pivot"] = (resampled["High"] + resampled["Low"] + resampled["Close"]) / 3
            resampled["R1"] = 2 * resampled["Pivot"] - resampled["Low"]
            resampled["S1"] = 2 * resampled["Pivot"] - resampled["High"]
            resampled["R2"] = resampled["Pivot"] + (resampled["High"] - resampled["Low"])
            resampled["S2"] = resampled["Pivot"] - (resampled["High"] - resampled["Low"])
            
            try:
                vwap = (df["Close"] * df["Volume"]).resample(tf).sum() / resampled["Volume"]
                resampled["VWAP"] = vwap
            except:
                resampled["VWAP"] = resampled["Close"].rolling(3).mean()
            
            if len(resampled) > 0:
                results[tf] = resampled.iloc[-1].to_dict()
            else:
                results[tf] = {}
                
        except Exception as e:
            print(f"Error calculating support/resistance for timeframe {tf}: {e}")
            results[tf] = {}
    
    if all(not data for data in results.values()):
        print("No valid data for any timeframe, creating fallback values")
        current_price = df["Close"].iloc[-1] if len(df) > 0 else 0
        results["fallback"] = {
            "Pivot": current_price,
            "R1": current_price * 1.01,
            "S1": current_price * 0.99,
            "R2": current_price * 1.02,
            "S2": current_price * 0.98,
            "VWAP": current_price
        }
    
    return results

def identify_support_resistance_zones(multi_tf_sr, price_tolerance=0.5):
    """
    识别多时间框架汇聚的支撑阻力区域。
    """
    all_resistance = []
    all_support = []
    
    print(f"Processing timeframes: {list(multi_tf_sr.keys())}")
    
    for tf, levels in multi_tf_sr.items():
        if levels is None or not isinstance(levels, dict) or len(levels) == 0:
            print(f"Skipping empty timeframe: {tf}")
            continue
            
        if not all(key in levels for key in ["R1", "R2", "S1", "S2"]):
            print(f"Missing required keys in timeframe {tf}: {levels.keys()}")
            continue
            
        try:
            all_resistance.extend([
                {"price": float(levels["R1"]), "level": "R1", "timeframe": tf},
                {"price": float(levels["R2"]), "level": "R2", "timeframe": tf}
            ])
            
            all_support.extend([
                {"price": float(levels["S1"]), "level": "S1", "timeframe": tf},
                {"price": float(levels["S2"]), "level": "S2", "timeframe": tf}
            ])
            print(f"Added levels from {tf}: R1={levels['R1']}, R2={levels['R2']}, S1={levels['S1']}, S2={levels['S2']}")
        except (KeyError, TypeError, ValueError) as e:
            print(f"Error processing timeframe {tf}: {e}")
            continue
    
    if not all_resistance and not all_support:
        print("No valid support/resistance levels found in any timeframe")
        return {
            "resistance_zones": [],
            "support_zones": []
        }
    
    resistance_zones = []
    support_zones = []
    
    # 处理阻力位
    if all_resistance:
        all_resistance.sort(key=lambda x: x["price"])
        current_zone = []
        
        for level in all_resistance:
            if not current_zone or abs(level["price"] - current_zone[0]["price"]) <= price_tolerance:
                current_zone.append(level)
            else:
                zone_price = sum(x["price"] for x in current_zone) / len(current_zone)
                resistance_zones.append({
                    "price": zone_price,
                    "strength": len(current_zone),
                    "timeframes": [x["timeframe"] for x in current_zone],
                    "levels": [x["level"] for x in current_zone]
                })
                current_zone = [level]
        
        if current_zone:
            zone_price = sum(x["price"] for x in current_zone) / len(current_zone)
            resistance_zones.append({
                "price": zone_price,
                "strength": len(current_zone),
                "timeframes": [x["timeframe"] for x in current_zone],
                "levels": [x["level"] for x in current_zone]
            })
    
    # 处理支撑位
    if all_support:
        all_support.sort(key=lambda x: x["price"])
        current_zone = []
        
        for level in all_support:
            if not current_zone or abs(level["price"] - current_zone[0]["price"]) <= price_tolerance:
                current_zone.append(level)
            else:
                zone_price = sum(x["price"] for x in current_zone) / len(current_zone)
                support_zones.append({
                    "price": zone_price,
                    "strength": len(current_zone),
                    "timeframes": [x["timeframe"] for x in current_zone],
                    "levels": [x["level"] for x in current_zone]
                })
                current_zone = [level]
        
        if current_zone:
            zone_price = sum(x["price"] for x in current_zone) / len(current_zone)
            support_zones.append({
                "price": zone_price,
                "strength": len(current_zone),
                "timeframes": [x["timeframe"] for x in current_zone],
                "levels": [x["level"] for x in current_zone]
            })
    
    print(f"Identified {len(resistance_zones)} resistance zones and {len(support_zones)} support zones")
    return {
        "resistance_zones": resistance_zones,
        "support_zones": support_zones
    }

def validate_sr_with_historical_data(df, sr_levels, price_tolerance=0.3, reaction_threshold=0.1):
    """
    使用历史价格反应验证支撑/阻力位。
    """
    try:
        if not isinstance(sr_levels, dict):
            print(f"Invalid sr_levels type: {type(sr_levels)}")
            return {"resistance": [], "support": []}
        
        if len(sr_levels.get("resistance_zones", [])) == 0 and len(sr_levels.get("support_zones", [])) == 0:
            print("No zones to validate")
            current_price = df["Close"].iloc[-1]
            
            return {
                "resistance": [
                    {"price": current_price * 1.01, "strength": 0.6, "tests": 0, "reaction_rate": 0},
                    {"price": current_price * 1.02, "strength": 0.7, "tests": 0, "reaction_rate": 0}
                ],
                "support": [
                    {"price": current_price * 0.99, "strength": 0.6, "tests": 0, "reaction_rate": 0},
                    {"price": current_price * 0.98, "strength": 0.7, "tests": 0, "reaction_rate": 0}
                ]
            }
            
        validated_levels = {"resistance": [], "support": []}
        
        # 处理阻力位
        for r_zone in sr_levels.get("resistance_zones", []):
            if "price" not in r_zone:
                continue
                
            price = r_zone["price"]
            reactions = 0
            total_tests = 0
            
            for i in range(1, len(df) - 1):
                if (df["High"].iloc[i] >= price - price_tolerance and 
                    df["High"].iloc[i] <= price + price_tolerance):
                    total_tests += 1
                    if (df["Close"].iloc[i+1] < df["Close"].iloc[i] * (1 - reaction_threshold/100)):
                        reactions += 1
            
            reaction_rate = reactions / max(total_tests, 1)
            validated_levels["resistance"].append({
                "price": price,
                "tests": total_tests,
                "reaction_rate": reaction_rate,
                "strength": r_zone.get("strength", 1) * (0.5 + reaction_rate/2)
            })
        
        # 处理支撑位
        for s_zone in sr_levels.get("support_zones", []):
            if "price" not in s_zone:
                continue
                
            price = s_zone["price"]
            reactions = 0
            total_tests = 0
            
            for i in range(1, len(df) - 1):
                if (df["Low"].iloc[i] <= price + price_tolerance and 
                    df["Low"].iloc[i] >= price - price_tolerance):
                    total_tests += 1
                    if (df["Close"].iloc[i+1] > df["Close"].iloc[i] * (1 + reaction_threshold/100)):
                        reactions += 1
            
            reaction_rate = reactions / max(total_tests, 1)
            validated_levels["support"].append({
                "price": price,
                "tests": total_tests,
                "reaction_rate": reaction_rate,
                "strength": s_zone.get("strength", 1) * (0.5 + reaction_rate/2)
            })
        
        # 检查并提供默认值
        if not validated_levels["resistance"] or not validated_levels["support"]:
            current_price = df["Close"].iloc[-1]
            if not validated_levels["resistance"]:
                validated_levels["resistance"] = [
                    {"price": current_price * 1.01, "strength": 0.6, "tests": 0, "reaction_rate": 0},
                    {"price": current_price * 1.02, "strength": 0.7, "tests": 0, "reaction_rate": 0}
                ]
            if not validated_levels["support"]:
                validated_levels["support"] = [
                    {"price": current_price * 0.99, "strength": 0.6, "tests": 0, "reaction_rate": 0},
                    {"price": current_price * 0.98, "strength": 0.7, "tests": 0, "reaction_rate": 0}
                ]
        
        return validated_levels
        
    except Exception as e:
        print(f"Error in validate_sr_with_historical_data: {e}")
        return {"resistance": [], "support": []}

def enhance_sr_with_volume_profile(df, sr_levels, num_bins=50):
    """
    使用成交量分布增强支撑阻力位。
    """
    try:
        if not isinstance(sr_levels, dict):
            print(f"Invalid sr_levels type: {type(sr_levels)}")
            sr_levels = {"resistance": [], "support": []}
            
        if len(sr_levels.get("resistance", [])) == 0 and len(sr_levels.get("support", [])) == 0:
            print("No zones to enhance")
            return {"resistance": [], "support": []}
            
        # 计算成交量分布
        price_min = df["Low"].min()
        price_max = df["High"].max()
        prices = df["Close"].values
        vols = df["Volume"].values
        
        hist, bin_edges = np.histogram(prices, bins=num_bins, range=(price_min, price_max), weights=vols)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 找出高成交量节点
        mean_vol = np.mean(hist)
        std_vol = np.std(hist)
        high_vol_threshold = mean_vol + 1.0 * std_vol
        
        high_vol_nodes = []
        for i, vol in enumerate(hist):
            if vol > high_vol_threshold:
                high_vol_nodes.append({
                    "price": bin_centers[i],
                    "volume": vol,
                    "vol_strength": (vol - mean_vol) / (std_vol + 1e-8)
                })
        
        # 增强现有支撑/阻力位
        enhanced_levels = {"resistance": [], "support": []}
        
        # 处理阻力位
        for r_level in sr_levels.get("resistance", []):
            if "price" not in r_level:
                continue
                
            matching_nodes = [node for node in high_vol_nodes 
                              if abs(node["price"] - r_level["price"]) <= (price_max - price_min) / num_bins * 3]
            
            if matching_nodes:
                avg_node_price = sum(node["price"] for node in matching_nodes) / len(matching_nodes)
                avg_vol_strength = sum(node["vol_strength"] for node in matching_nodes) / len(matching_nodes)
                
                enhanced_price = (r_level["price"] * r_level.get("strength", 1) + 
                                 avg_node_price * avg_vol_strength) / (r_level.get("strength", 1) + avg_vol_strength)
                
                enhanced_levels["resistance"].append({
                    "price": enhanced_price,
                    "original_price": r_level["price"],
                    "strength": r_level.get("strength", 1) * (1 + 0.3 * avg_vol_strength),
                    "volume_confirmed": True
                })
            else:
                enhanced_levels["resistance"].append({
                    "price": r_level["price"],
                    "strength": r_level.get("strength", 1),
                    "volume_confirmed": False
                })
        
        # 处理支撑位
        for s_level in sr_levels.get("support", []):
            if "price" not in s_level:
                continue
                
            matching_nodes = [node for node in high_vol_nodes 
                              if abs(node["price"] - s_level["price"]) <= (price_max - price_min) / num_bins * 3]
            
            if matching_nodes:
                avg_node_price = sum(node["price"] for node in matching_nodes) / len(matching_nodes)
                avg_vol_strength = sum(node["vol_strength"] for node in matching_nodes) / len(matching_nodes)
                
                enhanced_price = (s_level["price"] * s_level.get("strength", 1) + 
                                 avg_node_price * avg_vol_strength) / (s_level.get("strength", 1) + avg_vol_strength)
                
                enhanced_levels["support"].append({
                    "price": enhanced_price,
                    "original_price": s_level["price"],
                    "strength": s_level.get("strength", 1) * (1 + 0.3 * avg_vol_strength),
                    "volume_confirmed": True
                })
            else:
                enhanced_levels["support"].append({
                    "price": s_level["price"],
                    "strength": s_level.get("strength", 1),
                    "volume_confirmed": False
                })
        
        # 添加基于高成交量的额外支撑阻力位（如果找不到现有的）
        if not enhanced_levels["resistance"] and high_vol_nodes:
            current_price = df["Close"].iloc[-1]
            potential_resistance = [node for node in high_vol_nodes if node["price"] > current_price]
            
            if potential_resistance:
                potential_resistance.sort(key=lambda x: abs(x["price"] - current_price))
                for node in potential_resistance[:3]:
                    enhanced_levels["resistance"].append({
                        "price": node["price"],
                        "strength": 1.0 * node["vol_strength"] / 3.0,
                        "volume_confirmed": True,
                        "auto_generated": True
                    })
        
        if not enhanced_levels["support"] and high_vol_nodes:
            current_price = df["Close"].iloc[-1]
            potential_support = [node for node in high_vol_nodes if node["price"] < current_price]
            
            if potential_support:
                potential_support.sort(key=lambda x: abs(x["price"] - current_price))
                for node in potential_support[:3]:
                    enhanced_levels["support"].append({
                        "price": node["price"],
                        "strength": 1.0 * node["vol_strength"] / 3.0,
                        "volume_confirmed": True,
                        "auto_generated": True
                    })
        
        # 最后保底方案
        if not enhanced_levels["resistance"] or not enhanced_levels["support"]:
            current_price = df["Close"].iloc[-1]
            price_range = price_max - price_min
            
            if not enhanced_levels["resistance"]:
                enhanced_levels["resistance"] = [
                    {"price": current_price + price_range * 0.02, "strength": 0.5, "volume_confirmed": False, "auto_generated": True},
                    {"price": current_price + price_range * 0.05, "strength": 0.7, "volume_confirmed": False, "auto_generated": True}
                ]
            
            if not enhanced_levels["support"]:
                enhanced_levels["support"] = [
                    {"price": current_price - price_range * 0.02, "strength": 0.5, "volume_confirmed": False, "auto_generated": True},
                    {"price": current_price - price_range * 0.05, "strength": 0.7, "volume_confirmed": False, "auto_generated": True}
                ]
            
        return enhanced_levels
        
    except Exception as e:
        print(f"Error in enhance_sr_with_volume_profile: {e}")
        return {"resistance": [], "support": []}
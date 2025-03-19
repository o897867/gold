import pandas as pd
import numpy as np
import datetime
from indicator import (
    calc_atr_optimized,
    calc_pivots,
    calc_normality_test,      
    normality_test_vp,        
    calculate_all_indicators,
    calc_multi_tf_support_resistance_with_volume,
    identify_support_resistance_zones,
    validate_sr_with_historical_data,
    enhance_sr_with_volume_profile
)

def strategy_decision(df):
    """
    接受 DataFrame（含 Open、High、Low、Close、Volume 等必要列），
    计算各项指标，并输出包含 ATR、Pivot、VP正态性检测（基于成交量分布）、Z-Score、
    布林带、MACD、Volume Delta、GARCH 波动率以及多周期支撑阻力（Pivot, R1, S1, R2, S2, VWAP）
    的决策建议。
    
    返回:
      results: dict，包含各项指标分析及交易策略建议。
    """
    results = {}

    # 1) 计算所有指标（包括 GARCH 波动率等），返回带有各指标的 DataFrame
    df_ind = calculate_all_indicators(df)
    if len(df_ind) == 0:
        results["error"] = "数据不足，无法计算指标"
        return results
    last_row = df_ind.iloc[-1]

    # 2) ATR 分析（基于优化版 ATR）
    latest_atr = last_row.get("ATR_optimized", np.nan)
    atr_threshold = 0.8  # 从2.0降低到0.8，适合黄金市场
    if not np.isnan(latest_atr):
        if latest_atr > atr_threshold:
            results["volatility_comment"] = (
                f"当前ATR={latest_atr:.2f} 高于阈值{atr_threshold}，说明波动较大，可能处于趋势/突破行情。"
            )
            results["market_condition"] = "高波动趋势或突破行情"
        else:
            results["volatility_comment"] = (
                f"当前ATR={latest_atr:.2f} 低于阈值{atr_threshold}，说明波动较小，行情偏震荡。"
            )
            results["market_condition"] = "低波动区间行情"
    else:
        results["volatility_comment"] = "无法计算ATR"
        results["market_condition"] = "未知"

    # 3) Pivot 计算（按日聚合）
    pivot_df = calc_pivots(df)
    if len(pivot_df) > 0:
        pivot_today = pivot_df.iloc[-1]
        results["pivots"] = {
            "pivot": pivot_today["Pivot"],
            "R1": pivot_today["R1"],
            "S1": pivot_today["S1"]
        }
    else:
        results["pivots"] = {}

    # 4) VP 正态性检测：基于指定日期的 Volume Profile 数据进行 Jarque-Bera 检验
    # 这里选择最后一天的数据进行检测
    last_day_str = df.index[-1].strftime('%Y-%m-%d')
    vp_normal_test_res = normality_test_vp(df, last_day_str, bins=50)
    if vp_normal_test_res["is_normal"]:
        results["vp_normality_comment"] = (
            f"VP JB p-value={vp_normal_test_res['jb_pvalue']:.4f}，成交量分布近似正态。"
        )
    else:
        results["vp_normality_comment"] = (
            f"VP JB p-value={vp_normal_test_res['jb_pvalue']:.4f}，成交量分布存在肥尾风险。"
        )

    # 5) Z-Score 分析
    zscore = last_row.get("Zscore", np.nan)
    z_threshold = 1.5  # 从2.0降低到1.5，适合黄金市场
    if not np.isnan(zscore):
        if abs(zscore) > z_threshold:
            results["zscore_comment"] = (
                f"当前Zscore={zscore:.2f}，偏离显著，可能预示价格反转或突破。"
            )
        else:
            results["zscore_comment"] = (
                f"当前Zscore={zscore:.2f}，价格偏离正常。"
            )
    else:
        results["zscore_comment"] = "无法计算Zscore"

    # 6) 布林带分析
    current_price = last_row.get("Close", np.nan)
    boll_mid = last_row.get("Bollinger_mid", np.nan)
    boll_up = last_row.get("Bollinger_up", np.nan)
    boll_down = last_row.get("Bollinger_down", np.nan)
    if not (np.isnan(current_price) or np.isnan(boll_mid) or np.isnan(boll_up) or np.isnan(boll_down)):
        if current_price >= boll_up:
            results["bollinger_comment"] = (
                f"当前价格{current_price:.2f}接近或突破布林上轨（{boll_up:.2f}），可能超买，需谨慎。"
            )
        elif current_price <= boll_down:
            results["bollinger_comment"] = (
                f"当前价格{current_price:.2f}接近或突破布林下轨（{boll_down:.2f}），可能超卖，待反弹。"
            )
        else:
            results["bollinger_comment"] = (
                f"当前价格处于布林中轨附近（{boll_mid:.2f}），波动正常。"
            )
    else:
        results["bollinger_comment"] = "无法计算布林带信息"

    # 7) MACD 分析
    macd_line = last_row.get("MACD_line", np.nan)
    macd_signal = last_row.get("MACD_signal", np.nan)
    macd_hist = last_row.get("MACD_hist", np.nan)
    if not (np.isnan(macd_line) or np.isnan(macd_signal) or np.isnan(macd_hist)):
        if macd_line > macd_signal:
            results["macd_comment"] = (
                f"MACD显示多头优势（MACD_line={macd_line:.2f} > MACD_signal={macd_signal:.2f}）。"
            )
        else:
            results["macd_comment"] = (
                f"MACD显示空头优势（MACD_line={macd_line:.2f} < MACD_signal={macd_signal:.2f}）。"
            )
        results["macd_hist"] = macd_hist
    else:
        results["macd_comment"] = "无法计算MACD指标"

    # 8) Volume Delta 分析
    volume_delta = last_row.get("VolumeDelta", np.nan)
    if not np.isnan(volume_delta):
        if volume_delta > 0:
            results["vol_delta_comment"] = (
                f"Volume Delta为正（{volume_delta:.0f}），买盘较强。"
            )
        else:
            results["vol_delta_comment"] = (
                f"Volume Delta为负（{volume_delta:.0f}），卖盘较强。"
            )
    else:
        results["vol_delta_comment"] = "无法计算Volume Delta"

    # 9) GARCH 波动率分析
    garch_vol = last_row.get("GARCH_vol", np.nan)
    if not np.isnan(garch_vol):
        results["garch_vol_comment"] = f"当前GARCH波动率为 {garch_vol:.4f}"
    else:
        results["garch_vol_comment"] = "无法计算GARCH波动率"

    # 10) 综合决策建议
    if results["market_condition"] == "高波动趋势或突破行情":
        strategy = "趋势/突破交易"
        reason = (
            "当前波动较大，若MACD呈多头且Volume Delta为正，说明买盘强劲，适合顺势做多；"
            "若相反，则可考虑做空。"
        )
    elif results["market_condition"] == "低波动区间行情":
        strategy = "区间震荡交易"
        reason = "当前波动较低，价格可能在布林中轨附近波动，适合区间震荡交易。"
    else:
        strategy = "观望"
        reason = "ATR或其他指标信号不明朗，建议暂时观望。"
    results["strategy_suggestion"] = strategy
    results["strategy_reason"] = reason

    # 11) 输出最新指标值
    indicator_keys = [
        "MA_20", "MA_50", "MACD_line", "MACD_signal", "MACD_hist",
        "RSI", "Bollinger_up", "Bollinger_mid", "Bollinger_down",
        "Zscore", "VolumeDelta", "ATR_optimized", "GARCH_vol"
    ]
    latest_indicators = {k: last_row[k] for k in indicator_keys if k in last_row}
    results["latest_indicators"] = latest_indicators

    # 12) 多周期支撑阻力与 VWAP（15T, 1H, 4H, 1D）
    multi_tf_sr = calc_multi_tf_support_resistance_with_volume(df, timeframes=["15T", "1H", "4H", "1D"])
    support_resistance = {}
    for tf, df_sr in multi_tf_sr.items():
        if isinstance(df_sr, pd.DataFrame) and len(df_sr) > 0:
            last_sr = df_sr.iloc[-1]
            support_resistance[tf] = {
                "Pivot": last_sr["Pivot"],
                "R1": last_sr["R1"],
                "S1": last_sr["S1"],
                "R2": last_sr["R2"],
                "S2": last_sr["S2"],
                "VWAP": last_sr["VWAP"]
            }
        elif isinstance(df_sr, dict) and df_sr:
            # 直接使用字典形式的结果
            support_resistance[tf] = df_sr
        else:
            support_resistance[tf] = {}
    results["support_resistance"] = support_resistance

    return results

def analyze_market_context(df, lookback_periods=20):
    """
    Analyze overall market context to determine trend strength and volatility regime.
    
    Parameters:
      df: DataFrame with OHLC data
      lookback_periods: Number of periods to analyze
      
    Returns:
      Dict with market context analysis
    """
    # 确保有足够的数据
    if len(df) < lookback_periods:
        lookback_periods = len(df)
        
    df_context = df.copy().iloc[-lookback_periods:]
    
    # Determine overall trend
    start_price = df_context["Close"].iloc[0]
    end_price = df_context["Close"].iloc[-1]
    price_change_pct = (end_price - start_price) / start_price * 100
    
    # Calculate ADX for trend strength if available
    trend_strength = None
    if "ADX" in df_context.columns:
        trend_strength = df_context["ADX"].iloc[-1]
        strong_trend = trend_strength > 25
    else:
        # Simple trend strength metric as fallback
        up_days = sum(1 for i in range(1, len(df_context)) if df_context["Close"].iloc[i] > df_context["Close"].iloc[i-1])
        down_days = sum(1 for i in range(1, len(df_context)) if df_context["Close"].iloc[i] < df_context["Close"].iloc[i-1])
        strong_trend = abs(up_days - down_days) > len(df_context) * 0.3
    
    # Determine volatility regime
    recent_atr = df_context["ATR_optimized"].iloc[-5:].mean() if "ATR_optimized" in df_context.columns else None
    historical_atr = df_context["ATR_optimized"].mean() if "ATR_optimized" in df_context.columns else None
    
    if recent_atr is not None and historical_atr is not None:
        volatility_regime = "high" if recent_atr > historical_atr * 1.2 else "low" if recent_atr < historical_atr * 0.8 else "normal"
    else:
        # Simple volatility calculation as fallback
        close_returns = df_context["Close"].pct_change().dropna()
        current_vol = close_returns.std()
        volatility_regime = "high" if current_vol > 0.015 else "low" if current_vol < 0.005 else "normal"
    
    # Determine market regime
    if price_change_pct > 3 and strong_trend:
        if price_change_pct > 0:
            market_regime = "strong_uptrend"
        else:
            market_regime = "strong_downtrend"
    elif abs(price_change_pct) < 1 and not strong_trend:
        market_regime = "ranging"
    else:
        if price_change_pct > 0:
            market_regime = "weak_uptrend"
        else:
            market_regime = "weak_downtrend"
    
    return {
        "price_change_pct": price_change_pct,
        "trend_strength": trend_strength,
        "volatility_regime": volatility_regime,
        "market_regime": market_regime,
    }

def backtest_strategy_signals(df, lookback_days=30):
    """
    Backtest strategy signals over recent historical data to determine success probabilities.
    
    Parameters:
      df: DataFrame with OHLC and indicator data
      lookback_days: Number of days to backtest
      
    Returns:
      Dict with success rates for different signal types
    """
    # Ensure we have enough data
    if len(df) < lookback_days:
        lookback_days = len(df)
        if lookback_days < 10:  # 如果数据太少，无法进行有效回测
            return {"error": "Not enough historical data for backtesting"}
    
    backtest_df = df.copy().iloc[-lookback_days:]
    results = {
        "macd_cross_up": {"wins": 0, "losses": 0, "total": 0},
        "macd_cross_down": {"wins": 0, "losses": 0, "total": 0},
        "bollinger_breakout_up": {"wins": 0, "losses": 0, "total": 0},
        "bollinger_breakout_down": {"wins": 0, "losses": 0, "total": 0},
        "rsi_oversold": {"wins": 0, "losses": 0, "total": 0},
        "rsi_overbought": {"wins": 0, "losses": 0, "total": 0}
    }
    
    # 确保有足够的数据进行前向测试
    max_forward_bars = 5
    if len(backtest_df) <= 10 + max_forward_bars:
        return results
    
    for i in range(10, len(backtest_df) - max_forward_bars):  # Leave 5 days for outcome evaluation
        # Check for MACD crossover signals
        if "MACD_line" in backtest_df.columns and "MACD_signal" in backtest_df.columns:
            if (backtest_df["MACD_line"].iloc[i-1] < backtest_df["MACD_signal"].iloc[i-1] and
                backtest_df["MACD_line"].iloc[i] > backtest_df["MACD_signal"].iloc[i]):
                # MACD bullish crossover
                results["macd_cross_up"]["total"] += 1
                # Check outcome: win if price goes up at least 0.5% in next 5 days (降低阈值更适合黄金)
                future_return = (backtest_df["Close"].iloc[i+5] / backtest_df["Close"].iloc[i] - 1) * 100
                if future_return > 0.5:
                    results["macd_cross_up"]["wins"] += 1
                else:
                    results["macd_cross_up"]["losses"] += 1
            
            elif (backtest_df["MACD_line"].iloc[i-1] > backtest_df["MACD_signal"].iloc[i-1] and
                  backtest_df["MACD_line"].iloc[i] < backtest_df["MACD_signal"].iloc[i]):
                # MACD bearish crossover
                results["macd_cross_down"]["total"] += 1
                # Check outcome: win if price goes down at least 0.5% in next 5 days
                future_return = (backtest_df["Close"].iloc[i+5] / backtest_df["Close"].iloc[i] - 1) * 100
                if future_return < -0.5:
                    results["macd_cross_down"]["wins"] += 1
                else:
                    results["macd_cross_down"]["losses"] += 1
        
        # Check for Bollinger Band breakouts
        if all(col in backtest_df.columns for col in ["Close", "Bollinger_up", "Bollinger_down"]):
            if (backtest_df["Close"].iloc[i-1] <= backtest_df["Bollinger_up"].iloc[i-1] and
                backtest_df["Close"].iloc[i] > backtest_df["Bollinger_up"].iloc[i]):
                # Upward Bollinger breakout
                results["bollinger_breakout_up"]["total"] += 1
                future_return = (backtest_df["Close"].iloc[i+5] / backtest_df["Close"].iloc[i] - 1) * 100
                if future_return > 0.8:  # Higher threshold for breakouts, but lower than before
                    results["bollinger_breakout_up"]["wins"] += 1
                else:
                    results["bollinger_breakout_up"]["losses"] += 1
            
            elif (backtest_df["Close"].iloc[i-1] >= backtest_df["Bollinger_down"].iloc[i-1] and
                  backtest_df["Close"].iloc[i] < backtest_df["Bollinger_down"].iloc[i]):
                # Downward Bollinger breakout
                results["bollinger_breakout_down"]["total"] += 1
                future_return = (backtest_df["Close"].iloc[i+5] / backtest_df["Close"].iloc[i] - 1) * 100
                if future_return < -0.8:
                    results["bollinger_breakout_down"]["wins"] += 1
                else:
                    results["bollinger_breakout_down"]["losses"] += 1
        
        # Check for RSI signals
        if "RSI" in backtest_df.columns:
            if backtest_df["RSI"].iloc[i] < 30:
                # RSI oversold
                results["rsi_oversold"]["total"] += 1
                future_return = (backtest_df["Close"].iloc[i+5] / backtest_df["Close"].iloc[i] - 1) * 100
                if future_return > 0.5:
                    results["rsi_oversold"]["wins"] += 1
                else:
                    results["rsi_oversold"]["losses"] += 1
            
            elif backtest_df["RSI"].iloc[i] > 70:
                # RSI overbought
                results["rsi_overbought"]["total"] += 1
                future_return = (backtest_df["Close"].iloc[i+5] / backtest_df["Close"].iloc[i] - 1) * 100
                if future_return < -0.5:
                    results["rsi_overbought"]["wins"] += 1
                else:
                    results["rsi_overbought"]["losses"] += 1
    
    # Calculate win rates
    for signal_type, data in results.items():
        if data["total"] > 0:
            data["win_rate"] = data["wins"] / data["total"] * 100
        else:
            data["win_rate"] = 0
    
    return results

def calculate_leveraged_risk(price_movement, leverage=100, position_size=1):
    """
    计算杠杆交易下的实际风险
    
    参数:
      price_movement: 价格波动幅度(美金)
      leverage: 杠杆倍数
      position_size: 仓位大小(手)
      
    返回:
      account_impact: 对账户的影响百分比
    """
    # 黄金一手标准为100盎司
    standard_lot = 100
    
    # 价格变动的实际影响
    actual_impact = price_movement * standard_lot * position_size
    
    # 杠杆下的影响(假设保证金为合约价值/杠杆)
    leveraged_impact = actual_impact * leverage
    
    # 假设初始保证金为合约价值/杠杆
    contract_value = 2000 * standard_lot * position_size  # 假设黄金价格约为2000美元/盎司
    initial_margin = contract_value / leverage
    
    # 对账户的影响百分比
    account_impact = (leveraged_impact / initial_margin) * 100
    
    return account_impact

def assess_leverage_risk(current_price, nearest_support, nearest_resistance, leverage=100):
    """
    评估杠杆下的风险水平
    """
    if not nearest_support or not nearest_resistance:
        return "高风险 - 无法确定明确的支撑阻力位"
    
    # 计算到最近支撑位的距离
    support_distance = current_price - nearest_support["price"]
    
    # 计算到最近阻力位的距离
    resistance_distance = nearest_resistance["price"] - current_price
    
    # 计算最小距离
    min_distance = min(support_distance, resistance_distance)
    
    # 计算杠杆下的账户影响
    impact = calculate_leveraged_risk(min_distance, leverage)
    
    if impact > 20:
        return f"极高风险 - 价格波动{min_distance:.2f}美金可能导致账户亏损{impact:.1f}%"
    elif impact > 10:
        return f"高风险 - 价格波动{min_distance:.2f}美金可能导致账户亏损{impact:.1f}%"
    elif impact > 5:
        return f"中等风险 - 价格波动{min_distance:.2f}美金可能导致账户亏损{impact:.1f}%"
    else:
        return f"可控风险 - 价格波动{min_distance:.2f}美金可能导致账户亏损{impact:.1f}%"

def suggest_position_size(account_size, risk_percent, stop_loss_dollars):
    """
    根据账户大小和风险比例建议合适的仓位
    
    参数:
      account_size: 账户总资金
      risk_percent: 愿意冒险的账户百分比
      stop_loss_dollars: 止损距离(美金)
      
    返回:
      建议仓位大小(手数)
    """
    # 计算可承受的最大亏损金额
    max_loss = account_size * (risk_percent / 100)
    
    # 黄金一手标准为100盎司
    standard_lot = 100
    
    # 每点波动的美金价值
    pip_value = 0.1 * standard_lot  # 黄金0.1美金通常算作1pip
    
    # 止损距离(点数)
    stop_loss_pips = stop_loss_dollars * 10
    
    # 计算可开的最大仓位
    max_position = max_loss / (stop_loss_pips * pip_value)
    
    # 向下取整到最接近的0.01手
    suggested_position = np.floor(max_position * 100) / 100
    
    return max(suggested_position, 0.01)  # 最小仓位0.01手

def is_good_trading_time():
    """
    判断当前是否是黄金交易的良好时段
    """
    current_time = datetime.datetime.now().time()
    
    # 欧美交易时段重叠(通常流动性最好)
    if (datetime.time(13, 0) <= current_time <= datetime.time(17, 0)):
        return True, "欧美交易时段重叠，流动性好"
    
    # 亚洲时段(通常波动较小)
    elif (datetime.time(1, 0) <= current_time <= datetime.time(8, 0)):
        return False, "亚洲交易时段，建议观望或减小仓位"
    
    # 其他时段
    else:
        return None, "正常交易时段，注意市场波动"

def enhanced_strategy_decision(df, leverage=100, account_size=10000, risk_percent=2, language="cn"):
    """
    Enhanced trading strategy decision that combines technical indicators,
    market context, backtesting results, and leverage risk assessment.
    
    Parameters:
      df: DataFrame with OHLC and technical indicator data
      leverage: 杠杆倍数，默认100倍
      account_size: 账户总资金，默认10000美金
      risk_percent: 单笔交易风险百分比，默认2%
      language: Output language, "en" for English, "cn" for Chinese
      
    Returns:
      Dict with comprehensive decision analysis and recommendation
    """
    # Run the original strategy decision
    try:
        base_decision = strategy_decision(df)
    except Exception as e:
        print(f"Error in base strategy decision: {e}")
        base_decision = {"error": f"基础策略分析错误: {e}"}
    
    # Add market context analysis
    try:
        market_context = analyze_market_context(df)
        base_decision["market_context"] = market_context
    except Exception as e:
        print(f"Error in market context analysis: {e}")
        base_decision["market_context"] = {"error": str(e), "market_regime": "未知", "volatility_regime": "未知"}
    
    # Calculate support/resistance with improved methods
    try:
        multi_tf_sr = calc_multi_tf_support_resistance_with_volume(df)
        sr_zones = identify_support_resistance_zones(multi_tf_sr, price_tolerance=0.5)  # 降低容忍度到0.5美金
        validated_sr = validate_sr_with_historical_data(df, sr_zones, price_tolerance=0.3, reaction_threshold=0.1)  # 调整参数
        enhanced_sr = enhance_sr_with_volume_profile(df, validated_sr)
        base_decision["enhanced_sr"] = enhanced_sr
    except Exception as e:
        print(f"Error in support/resistance analysis: {e}")
        base_decision["enhanced_sr"] = {"support": [], "resistance": []}
    
    # Add backtesting-based probabilities
    try:
        backtest_results = backtest_strategy_signals(df)
        base_decision["signal_probabilities"] = backtest_results
    except Exception as e:
        print(f"Error in backtesting signals: {e}")
        base_decision["signal_probabilities"] = {}
    
    # Check if it's a good trading time
    good_time, time_comment = is_good_trading_time()
    base_decision["trading_time"] = {
        "is_good_time": good_time,
        "comment": time_comment
    }
    
    # Enhanced decision logic based on all factors
    try:
        current_price = df["Close"].iloc[-1]
        latest_atr = df["ATR_optimized"].iloc[-1] if "ATR_optimized" in df.columns else None
        
        # Identify nearest support and resistance levels
        nearest_resistance = None
        nearest_support = None
        
        resistance_levels = base_decision["enhanced_sr"].get("resistance", [])
        support_levels = base_decision["enhanced_sr"].get("support", [])
        
        if resistance_levels:
            resistance_levels = sorted(resistance_levels, key=lambda x: x["price"])
            nearest_resistance = next((r for r in resistance_levels if r["price"] > current_price), None)
        
        if support_levels:
            support_levels = sorted(support_levels, key=lambda x: -x["price"])
            nearest_support = next((s for s in support_levels if s["price"] < current_price), None)
        
        # Leverage risk assessment
        if nearest_support and nearest_resistance:
            leverage_risk = assess_leverage_risk(current_price, nearest_support, nearest_resistance, leverage)
            base_decision["leverage_risk"] = leverage_risk
        else:
            base_decision["leverage_risk"] = "无法评估风险 - 缺少支撑或阻力位数据"
        
        # Calculate risk-reward ratio if both support and resistance are available
        risk_reward_ratio = None
        if nearest_resistance and nearest_support:
            potential_reward = nearest_resistance["price"] - current_price
            potential_risk = current_price - nearest_support["price"]
            
            if potential_risk > 0:
                risk_reward_ratio = potential_reward / potential_risk
        
        # Final decision with confidence level
        decision = {
            "action": None,
            "confidence": 0,
            "primary_reason": None,
            "risk_reward_ratio": risk_reward_ratio
        }
        
        # Get market regime from context, with fallback
        market_regime = base_decision.get("market_context", {}).get("market_regime", "unknown")
        
        # 黄金交易中更小的价格变动百分比
        price_proximity_threshold = 0.001  # 将0.01 (1%)降低到0.001 (0.1%)
        
        if market_regime == "strong_uptrend" or market_regime == "强势上升趋势":
            # In strong uptrend, look for pullbacks to support for long entries
            if nearest_support and abs(current_price - nearest_support["price"]) / current_price < price_proximity_threshold:
                decision["action"] = "BUY" if language == "en" else "买入"
                decision["primary_reason"] = "Price at support in strong uptrend" if language == "en" else "价格处于强势上升趋势中的支撑位"
                decision["confidence"] = 0.8
        
        elif market_regime == "strong_downtrend" or market_regime == "强势下降趋势":
            # In strong downtrend, look for bounces to resistance for short entries
            if nearest_resistance and abs(current_price - nearest_resistance["price"]) / current_price < price_proximity_threshold:
                decision["action"] = "SELL" if language == "en" else "卖出"
                decision["primary_reason"] = "Price at resistance in strong downtrend" if language == "en" else "价格处于强势下降趋势中的阻力位"
                decision["confidence"] = 0.8
        
        elif market_regime in ["weak_uptrend", "weak_downtrend", "ranging", "弱势上升趋势", "弱势下降趋势", "区间震荡"]:
            # In ranging markets, look for confirmed S/R bounces
            
            # Check for bounces off support
            if nearest_support and abs(current_price - nearest_support["price"]) / current_price < price_proximity_threshold:
                decision["action"] = "BUY" if language == "en" else "买入"
                decision["primary_reason"] = "Support bounce in ranging market" if language == "en" else "区间市场中的支撑位反弹"
                decision["confidence"] = 0.6
            
            # Check for bounces off resistance
            elif nearest_resistance and abs(current_price - nearest_resistance["price"]) / current_price < price_proximity_threshold:
                decision["action"] = "SELL" if language == "en" else "卖出"
                decision["primary_reason"] = "Resistance bounce in ranging market" if language == "en" else "区间市场中的阻力位反弹"
                decision["confidence"] = 0.6
        
        # 根据交易时段调整信心度
        trading_time_info = base_decision.get("trading_time", {})
        if trading_time_info.get("is_good_time") is False:
            # 非良好交易时间，降低信心
            decision["confidence"] *= 0.7
            if decision["primary_reason"]:
                decision["primary_reason"] += f"，但{trading_time_info.get('comment', '当前非最佳交易时段')}"
        elif trading_time_info.get("is_good_time") is True:
            # 良好交易时间，适当提升信心
            decision["confidence"] *= 1.1
        
        # 考虑杠杆风险，如果风险太高则降低信心
        leverage_risk = base_decision.get("leverage_risk", "")
        if "极高风险" in leverage_risk:
            decision["confidence"] *= 0.6
        elif "高风险" in leverage_risk:
            decision["confidence"] *= 0.8
        
        # 计算合适的止损位
        if decision["action"] in ["BUY", "买入"] and nearest_support and latest_atr:
            stop_loss = nearest_support["price"] - latest_atr * 0.5
            stop_loss_distance = current_price - stop_loss
            decision["stop_loss"] = stop_loss
            decision["stop_loss_pips"] = stop_loss_distance * 10  # 黄金通常以0.1为1pip
            
            # 计算建议仓位
            suggested_position = suggest_position_size(account_size, risk_percent, stop_loss_distance)
            decision["suggested_position"] = suggested_position
            
        elif decision["action"] in ["SELL", "卖出"] and nearest_resistance and latest_atr:
            stop_loss = nearest_resistance["price"] + latest_atr * 0.5
            stop_loss_distance = stop_loss - current_price
            decision["stop_loss"] = stop_loss
            decision["stop_loss_pips"] = stop_loss_distance * 10
            
            # 计算建议仓位
            suggested_position = suggest_position_size(account_size, risk_percent, stop_loss_distance)
            decision["suggested_position"] = suggested_position
        
        # If no clear signal, provide some default action
        if decision["action"] is None:
            if "macd_comment" in base_decision and "多头优势" in base_decision["macd_comment"]:
                decision["action"] = "买入"
                decision["primary_reason"] = "MACD显示多头优势，但价格未处于明确支撑位"
                decision["confidence"] = 0.4
            elif "macd_comment" in base_decision and "空头优势" in base_decision["macd_comment"]:
                decision["action"] = "卖出"
                decision["primary_reason"] = "MACD显示空头优势，但价格未处于明确阻力位"
                decision["confidence"] = 0.4
            else:
                decision["action"] = "观望"
                decision["primary_reason"] = "无明确信号，建议观望"
                decision["confidence"] = 0.5
        
        # Adjust confidence based on backtesting results
        backtest_results = base_decision.get("signal_probabilities", {})
        if decision["action"] in ["BUY", "买入"]:
            # Check relevant bullish signal win rates
            relevant_signals = ["macd_cross_up", "rsi_oversold"]
            avg_win_rate = 0
            count = 0
            
            for signal in relevant_signals:
                if signal in backtest_results and isinstance(backtest_results[signal], dict) and backtest_results[signal].get("total", 0) > 5:
                    avg_win_rate += backtest_results[signal].get("win_rate", 0)
                    count += 1
            
            if count > 0:
                avg_win_rate /= count
                # Adjust confidence (0.5 to 1.5 multiplier based on win rate)
                confidence_multiplier = 0.5 + (avg_win_rate / 100)
                decision["confidence"] *= confidence_multiplier
        
        elif decision["action"] in ["SELL", "卖出"]:
            # Check relevant bearish signal win rates
            relevant_signals = ["macd_cross_down", "rsi_overbought"]
            avg_win_rate = 0
            count = 0
            
            for signal in relevant_signals:
                if signal in backtest_results and isinstance(backtest_results[signal], dict) and backtest_results[signal].get("total", 0) > 5:
                    avg_win_rate += backtest_results[signal].get("win_rate", 0)
                    count += 1
            
            if count > 0:
                avg_win_rate /= count
                confidence_multiplier = 0.5 + (avg_win_rate / 100)
                decision["confidence"] *= confidence_multiplier
        
        # Cap confidence at 1.0
        decision["confidence"] = min(decision["confidence"], 1.0)
        
        # 添加风险评价和交易时间建议
        decision["leverage_risk"] = base_decision.get("leverage_risk", "未评估风险")
        decision["trading_time_comment"] = trading_time_info.get("comment", "未评估交易时间")
        
        # Add decision to results
        base_decision["enhanced_decision"] = decision
    except Exception as e:
        print(f"Error in decision logic: {e}")
        base_decision["enhanced_decision"] = {
            "action": "观望",
            "primary_reason": f"决策过程出错: {e}",
            "confidence": 0.1
        }
    
    # Translate other key messages if Chinese is requested
    if language == "cn":
        try:
            # Translate market context
            if "market_context" in base_decision:
                context = base_decision["market_context"]
                if context.get("market_regime") == "strong_uptrend":
                    context["market_regime"] = "强势上升趋势"
                elif context.get("market_regime") == "strong_downtrend":
                    context["market_regime"] = "强势下降趋势"
                elif context.get("market_regime") == "weak_uptrend":
                    context["market_regime"] = "弱势上升趋势"
                elif context.get("market_regime") == "weak_downtrend":
                    context["market_regime"] = "弱势下降趋势"
                elif context.get("market_regime") == "ranging":
                    context["market_regime"] = "区间震荡"
                    
                if context.get("volatility_regime") == "high":
                    context["volatility_regime"] = "高波动"
                elif context.get("volatility_regime") == "low":
                    context["volatility_regime"] = "低波动"
                elif context.get("volatility_regime") == "normal":
                    context["volatility_regime"] = "正常波动"
            
            # Translate strategy and market condition
            if "market_condition" in base_decision:
                if base_decision["market_condition"] == "high volatility trend or breakout":
                    base_decision["market_condition"] = "高波动趋势或突破行情"
                elif base_decision["market_condition"] == "low volatility range trading":
                    base_decision["market_condition"] = "低波动区间行情"
                else:
                    base_decision["market_condition"] = "未知行情"
                    
            if "strategy_suggestion" in base_decision:
                if base_decision["strategy_suggestion"] == "Trend/Breakout Trading":
                    base_decision["strategy_suggestion"] = "趋势/突破交易"
                elif base_decision["strategy_suggestion"] == "Range Trading":
                    base_decision["strategy_suggestion"] = "区间震荡交易"
                else:
                    base_decision["strategy_suggestion"] = "观望"
        except Exception as e:
            print(f"Error in translation: {e}")
    
    return base_decision
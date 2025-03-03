import pandas as pd
import numpy as np
from indicator import (
    calc_atr_optimized,   # 优化ATR函数（例如支持EWM平滑）
    calc_pivots,          # 计算Pivot、R1、S1
    calc_normality_test,  # 使用Jarque-Bera检验正态性
    calculate_all_indicators,  # 整合 MA、MACD、RSI、Bollinger、Zscore、VolumeDelta 等
    calc_multi_tf_support_resistance_with_volume  # 多周期支撑阻力和VWAP
)

def strategy_decision(df):
    """
    接受 DataFrame（含 Open、High、Low、Close、Volume 等必要列），
    计算各项技术指标，并输出包含 ATR、Pivot、正态性、Z-Score、
    布林带、MACD、Volume Delta 以及多周期支撑阻力（Pivot, R1, S1, R2, S2, VWAP）
    的决策建议。
    
    返回:
      results: dict，包含各项指标分析及交易策略建议。
    """
    results = {}

    # 1) 计算所有指标（返回包含多项指标的 DataFrame）
    df_ind = calculate_all_indicators(df)
    if len(df_ind) == 0:
        results["error"] = "数据不足，无法计算指标"
        return results

    # 取最新一根K线的指标数据
    last_row = df_ind.iloc[-1]

    # 2) ATR 分析：基于优化版ATR
    latest_atr = last_row.get("ATR_optimized", np.nan)
    atr_threshold = 2.0  # 可根据历史统计调整
    if not np.isnan(latest_atr):
        if latest_atr > atr_threshold:
            results["volatility_comment"] = (
                f"当前ATR={latest_atr:.2f} 高于阈值{atr_threshold}，说明波动较大，"
                "可能处于趋势/突破行情。"
            )
            results["market_condition"] = "高波动趋势或突破行情"
        else:
            results["volatility_comment"] = (
                f"当前ATR={latest_atr:.2f} 低于阈值{atr_threshold}，说明波动较小，"
                "行情偏震荡。"
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

    # 4) 对数收益率正态性检验
    df_ind["log_ret"] = np.log(df_ind["Close"] / df_ind["Close"].shift(1))
    normal_test_res = calc_normality_test(df_ind["log_ret"])
    if normal_test_res["is_normal"]:
        results["normality_comment"] = (
            f"JB p-value={normal_test_res['jb_pvalue']:.4f}，对数收益率分布近似正态。"
        )
    else:
        results["normality_comment"] = (
            f"JB p-value={normal_test_res['jb_pvalue']:.4f}，对数收益率分布存在肥尾风险。"
        )

    # 5) 正态偏离分析（Z-Score）
    zscore = last_row.get("Zscore", np.nan)
    z_threshold = 2.0  # 可根据实际情况调整
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

    # 9) 综合决策建议：结合ATR、MACD、Volume Delta及其他指标给出策略建议
    if results["market_condition"] == "高波动趋势或突破行情":
        strategy = "趋势/突破交易"
        reason = (
            "当前波动较大，若MACD呈多头且Volume Delta为正，说明买盘强劲，"
            "适合顺势做多；若相反，则可考虑做空。"
        )
    elif results["market_condition"] == "低波动区间行情":
        strategy = "区间震荡交易"
        reason = (
            "当前波动较低，价格可能在布林中轨附近波动，适合区间震荡交易。"
        )
    else:
        strategy = "观望"
        reason = "ATR或其他指标信号不明朗，建议暂时观望。"
    results["strategy_suggestion"] = strategy
    results["strategy_reason"] = reason

    # 10) 输出所有最新指标值（便于查看）
    indicator_keys = [
        "MA_20", "MA_50", "MACD_line", "MACD_signal", "MACD_hist",
        "RSI", "Bollinger_up", "Bollinger_mid", "Bollinger_down",
        "Zscore", "VolumeDelta", "ATR_optimized"
    ]
    latest_indicators = {k: last_row[k] for k in indicator_keys if k in last_row}
    results["latest_indicators"] = latest_indicators

    # 11) 新增：多周期支撑阻力与VWAP（15T, 1H, 4H, 1D）
    multi_tf_sr = calc_multi_tf_support_resistance_with_volume(df, timeframes=["15T", "1H", "4H", "1D"])
    support_resistance = {}
    for tf, df_sr in multi_tf_sr.items():
        if len(df_sr) > 0:
            last_sr = df_sr.iloc[-1]
            support_resistance[tf] = {
                "Pivot": last_sr["Pivot"],
                "R1": last_sr["R1"],
                "S1": last_sr["S1"],
                "R2": last_sr["R2"],
                "S2": last_sr["S2"],
                "VWAP": last_sr["VWAP"]
            }
        else:
            support_resistance[tf] = {}
    results["support_resistance"] = support_resistance

    return results

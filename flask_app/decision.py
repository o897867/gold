import pandas as pd
import numpy as np
from indicator import (
    calc_atr_optimized,
    calc_pivots,
    calc_normality_test,      # 仍然保留，但我们这边用 normality_test_vp
    normality_test_vp,        # 新的正态性检测函数，基于 VP（成交量直方图）
    calculate_all_indicators,
    calc_multi_tf_support_resistance_with_volume
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
    atr_threshold = 2.0
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
    z_threshold = 2.0
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

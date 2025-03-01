
import pandas as pd
import numpy as np
def strategy_decision(df):
    """
    接受DataFrame(含Open/High/Low/Close/Volume等)，计算各种指标并输出决策信息
    """
    results = {}

    # 计算 ATR(14)
    from indicator import calc_atr, calc_pivots, calc_normality_test
    atr_14 = calc_atr(df, window=14)
    latest_atr = atr_14.iloc[-1] if len(atr_14) > 0 else None
    
    # 当ATR超过阈值（需你根据经验或统计来定）就认定波动较大
    atr_threshold = 2.0  # 示例阈值
    if latest_atr is not None:
        if latest_atr > atr_threshold:
            results["volatility_comment"] = f"当前ATR={latest_atr:.2f}，高于{atr_threshold}，说明波动较大"
            results["market_condition"] = "高波动趋势或突破行情"
        else:
            results["volatility_comment"] = f"当前ATR={latest_atr:.2f}，低于{atr_threshold}，行情或偏震荡"
            results["market_condition"] = "低波动区间行情"
    else:
        results["volatility_comment"] = "无法计算ATR"
        results["market_condition"] = "未知"
    
    # 计算Pivot（日级别）
    daily_pivot_df = calc_pivots(df)
    if len(daily_pivot_df) > 0:
        pivot_today = daily_pivot_df.iloc[-1]
        results["pivots"] = {
            "pivot": pivot_today["Pivot"],
            "R1": pivot_today["R1"],
            "S1": pivot_today["S1"]
        }
    else:
        results["pivots"] = {}

    # 收益率正态性检验
    df["log_ret"] = (df["Close"] / df["Close"].shift(1)).apply(np.log)
    normal_test_res = calc_normality_test(df["log_ret"])
    if normal_test_res["is_normal"]:
        results["normality_comment"] = (
            f"Jarque-Bera p-value={normal_test_res['jb_pvalue']:.4f} > 0.05,"
            "当前分布近似正态"
        )
    else:
        results["normality_comment"] = (
            f"Jarque-Bera p-value={normal_test_res['jb_pvalue']:.4f} < 0.05,"
            "当前分布肥尾风险较大"
        )
    
    # 简单“决策建议”逻辑
    if results["market_condition"] == "高波动趋势或突破行情":
        strategy = "趋势/突破交易"
        reason = "当前波动高，容易走单边或大幅波动"
    else:
        strategy = "区间震荡交易"
        reason = "当前波动低，价格可能在支撑阻力之间往返"
    
    results["strategy_suggestion"] = strategy
    results["strategy_reason"] = reason
    
    # 你还可以组合Volume Profile、Market Profile、以及识别的POC等，补充更多字段
    # ...
    
    return results

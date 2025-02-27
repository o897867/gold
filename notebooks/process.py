import json
import time
import threading
import websocket
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class OrderBookPlotter:
    """ 实时绘制 DOM 订单簿 (Depth of Market) """
    
    def __init__(self, depth_display=20):
        plt.ion()  # 开启交互模式
        self.depth_display = depth_display  # 设定显示的深度级数
        self.fig, self.ax = plt.subplots(figsize=(6, 8))  # 适配 DOM 风格
        self.bids = []
        self.asks = []
    
    def update_data(self, bids, asks):
        """ 更新订单簿数据 """
        self.bids = bids
        self.asks = asks
        self.plot_order_book()

    def plot_order_book(self):
        """ 绘制 DOM 订单簿 """
        self.ax.clear()
        self.ax.axis("off")  # 关闭坐标轴，让图表更整洁

        if not self.bids or not self.asks:
            return

        # 提取价格 & 数量
        bid_prices = [round(price, 2) for price, _ in self.bids]
        ask_prices = [round(price, 2) for price, _ in self.asks]
        bid_volumes = {round(price, 2): volume for price, volume in self.bids}
        ask_volumes = {round(price, 2): volume for price, volume in self.asks}

        # 获取最优买价 & 卖价
        best_bid = min(bid_prices, default=2895.00)  # 最低买价
        best_ask = max(ask_prices, default=2897.00)  # 最高卖价

        # 设定完整的价格范围 (步长 0.01 以确保匹配数据)
        price_range = np.arange(best_bid - 2, best_ask + 2, 0.01)
        price_range = [round(p, 2) for p in price_range]  # 处理浮点精度问题

        # 生成 DOM 表格数据
        table_data = []
        for price in reversed(price_range):  # 从高价到低价排序
            bid_vol = bid_volumes.get(price, "")  # 找不到订单量时显示空白
            ask_vol = ask_volumes.get(price, "")
            table_data.append([bid_vol, f"{price:.2f}", ask_vol])

        # 创建表格
        table = self.ax.table(cellText=table_data,
                              colLabels=["买单量 (Bids)", "价格", "卖单量 (Asks)"],
                              cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2])  # 自动调整列宽

        # 颜色填充 (让买单变蓝色, 卖单变红色)
        for i, price in enumerate(reversed(price_range)):
            if price in bid_volumes:
                table[(i + 1, 0)].set_facecolor("lightblue")  # 买单 (蓝色)
            if price in ask_volumes:
                table[(i + 1, 2)].set_facecolor("lightcoral")  # 卖单 (红色)

        # 刷新图表
        plt.pause(0.1)

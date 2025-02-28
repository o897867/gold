import json
import time
import threading
import websocket    # pip install websocket-client
import json
import time
import threading
import websocket
import matplotlib.pyplot as plt
import numpy as np
import ssl
from flask import Flask,send_file,Response,send_from_directory,render_template
from datetime import datetime
import os
import sys
app = Flask(__name__)
current_orderbook = {
    "bids": [],
    "asks": []
}

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
        """ 绘制订单簿，并保存为图片 """
        self.ax.clear()
        self.ax.axis("off")

        if not self.bids or not self.asks:
            return

        # 价格 & 数量
        bid_prices = [round(price, 2) for price, _ in self.bids]
        ask_prices = [round(price, 2) for price, _ in self.asks]
        bid_volumes = {round(price, 2): volume for price, volume in self.bids}
        ask_volumes = {round(price, 2): volume for price, volume in self.asks}

        # 获取最优买价 & 卖价
        best_bid = min(bid_prices, default=2895.00)
        best_ask = max(ask_prices, default=2897.00)

        # 设定完整价格范围
        price_range = np.arange(best_bid - 2, best_ask + 2, 0.01)
        price_range = [round(p, 2) for p in price_range]

        # 生成 DOM 表格数据
        table_data = []
        for price in reversed(price_range):
            bid_vol = bid_volumes.get(price, "")
            ask_vol = ask_volumes.get(price, "")
            table_data.append([bid_vol, f"{price:.2f}", ask_vol])

        # 创建表格
        table = self.ax.table(cellText=table_data,
                              colLabels=["买单量 (Bids)", "价格", "卖单量 (Asks)"],
                              cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2])

        # 颜色填充
        for i, price in enumerate(reversed(price_range)):
            if price in bid_volumes:
                table[(i + 1, 0)].set_facecolor("lightblue")
            if price in ask_volumes:
                table[(i + 1, 2)].set_facecolor("lightcoral")

        # 保存为图片
        plt.savefig("orderbook.png")
        plt.pause(0.1)

class Feed:
    """ 处理 WebSocket 连接，并实时更新订单簿 """

    def __init__(self, plotter):
        self.url = 'wss://quote.alltick.io/quote-b-ws-api?token=949751a4ea6a586f9e2805a3909d456a-c-app'
        self.plotter = plotter

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,  # 确保这里不会报错
            on_close=self.on_close
        )
    def reconnect(self):
        """ 尝试自动重连 """
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print("restart Script")
            self.restart_script()
        wait_time = 2 ** self.reconnect_attempts  # 指数级退避（2s, 4s, 8s, ...）
        print(f"🔄 尝试在 {wait_time} 秒后重连...")
        time.sleep(wait_time)

        self.reconnect_attempts += 1
        self.start()  # 重新启动 WebSocket

    def on_error(self, ws, error):
        """ 处理 WebSocket 错误 """
        print(f"❌ WebSocket 发生错误: {error}")
        self.reconnect()  # 自动重连

    def on_close(self, ws, close_status_code, close_msg):
        """ WebSocket 关闭时执行 """
        print('❌ WebSocket 连接已关闭！')
        self.reconnect()  # 自动重连
    def on_open(self, ws):
        """ WebSocket 连接成功 """
        print('✅ WebSocket 连接成功！')
        self.reconnect_attempts = 0
        sub_param = {
            "cmd_id": 22002,
            "seq_id": 123,
            "trace": "3baaa938-f92c-4a74-a228-fd49d5e2f8bc-1678419657806",
            "data": {
                "symbol_list": [
                    {"code": "XAUUSD", "depth_level": 10}
                ]
            }
        }
        ws.send(json.dumps(sub_param))
        print("📡 订阅深度报价数据成功！")

    def on_message(self, ws, message):
        """ 处理 WebSocket 返回的市场深度数据 """
        try:
            data = json.loads(message)
            if "data" not in data or "bids" not in data["data"] or "asks" not in data["data"]:
                return
            
            bids = [(float(bid['price']), float(bid['volume'])) for bid in data["data"]["bids"]]
            asks = [(float(ask['price']), float(ask['volume'])) for ask in data["data"]["asks"]]

            # ❷ 更新全局变量
            global current_orderbook
            current_orderbook["bids"] = bids
            current_orderbook["asks"] = asks

            # 你也可以继续 plotter.update_data(...) 做可视化
            self.plotter.update_data(bids, asks)

        except Exception as e:
            print(f"❌ 解析错误: {e}")
    def start(self):
        """ 启动 WebSocket 连接 """
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    def restart_script(self):
        """ 重启脚本 """
        python = sys.executable
        os.execl(python, [python] + sys.argv)


# 初始化绘图工具
plotter = OrderBookPlotter()

# 初始化 WebSocket 订阅
feed = Feed(plotter)
# 启动 WebSocket
@app.route('/')
def index():
    return render_template('d3_orderbook.html')

@app.route('/ob.png')
def get_ob():
    return send_file('orderbook.png', mimetype='image/png')
@app.route("/sse")
def sse_orderbook():
    """ SSE 路由：持续推送订单簿数据给前端 """
    def generate():
        while True:
            # ❸ 读取全局变量 current_orderbook
            data_json = json.dumps(current_orderbook)
            # SSE 格式： data: <json字符串>\n\n
            yield f"data: {data_json}\n\n"

            time.sleep(1)  # 每秒推送一次，可自行调整频率

    # SSE 要用 text/event-stream
    return Response(generate(), mimetype="text/event-stream")

@app.route('/orderbook_view')
def orderbook_view():
    return """
    <html>
    <head>
        <title>实时订单簿</title>
        <script>
        // 每隔 1 秒刷新一次图片
        setInterval(function(){
            var d = new Date();
            // 给图片链接加上时间戳，避免浏览器缓存
            document.getElementById('orderbook_img').src = '/ob.png?ts=' + d.getTime();
        }, 1000);
        </script>
    </head>
    <body>
        <h1>实时订单簿</h1>
        <img id="orderbook_img" src="/ob.png" />
    </body>
    </html>
    """


if __name__ == '__main__':
    def run_ws():
        feed.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    threading.Thread(target=run_ws).start()
    app.run(host='0.0.0.0', port=5000,threaded=True)

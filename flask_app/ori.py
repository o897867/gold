from collections import deque
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
from flask import Flask,send_file,Response,send_from_directory,render_template,request,jsonify
from datetime import datetime, timedelta
import os
from functools import wraps
import sys

app = Flask(__name__)
current_orderbook = {
    "bids": [],
    "asks": []
}
tick_data_list = deque(maxlen=100)
def add_tick_data(data):
    tick_data_list.append(data)

def check_auth(username, password):
    """验证用户名和密码是否正确"""
    # 修改这里的用户名和密码为你需要的
    return username == 'tmgm' and password == 'glzdggls'

def authenticate():
    """返回 401 响应，提示输入正确的凭证"""
    return Response(
        '无法验证您的访问权限，请提供正确的凭证\n',
        401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    """装饰器：如果认证失败则返回 401"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated


class OrderBookPlotter:
    """ 实时绘制 DOM 订单簿 (Depth of Market) """
    
    def __init__(self, depth_display=20):
        #plt.ion()  # 开启交互模式
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
        print('plotted')
        plt.savefig("orderbook.png")
       
        #plt.pause(0.1)

import websocket
import time
import ssl
import json
import threading
import sys
import os

import websocket
import time
import ssl
import json
import threading
import sys
import os

class SingleFeed:
    def __init__(self, plotter):
        # The same WS URL for both depth and tick data
        self.url = 'wss://quote.alltick.io/quote-b-ws-api?token=949751a4ea6a586f9e2805a3909d456a-c-app'
        self.ws = None

        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        # Pass in your existing plotter
        self.plotter = plotter

        # Combine the data structures
        self.current_orderbook = {"bids": [], "asks": []}
        self.tick_data_list = []

    def start(self):
        """Start a single WS connection."""
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        # Run forever in current thread
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def on_open(self, ws):
        """Called when WS is connected."""
        print("✅ SingleFeed: WebSocket connected")
        self.reconnect_attempts = 0

        # 1) Subscribe to depth data
        depth_sub = {
            "cmd_id": 22002,  # e.g. your depth subscription
            "seq_id": 111,
            "trace": "DepthSub",
            "data": {
                "symbol_list": [
                    {"code": "BTCUSDT", "depth_level": 10}
                ]
            }
        }
        ws.send(json.dumps(depth_sub))
        print("📡 Subscribed to depth data")

        # 2) Subscribe to tick data
        tick_sub = {
            "cmd_id": 22004,  # e.g. your tick subscription
            "seq_id": 222,
            "trace": "TickSub",
            "data": {
                "symbol_list": [
                    {"code": "BTCUSDT"}
                ]
            }
        }
        ws.send(json.dumps(tick_sub))
        print("📡 Subscribed to tick data")

        # Heartbeat in a background thread
        threading.Thread(target=self.thread_heartbeat, daemon=True).start()

    def on_message(self, ws, message):
        """Handle all messages (depth + tick) from the single feed."""
        try:
            data = json.loads(message)
            cmd_id = data.get("cmd_id")

            # 1) Depth updates (bids/asks)
            print(data)
            if (
                "data" in data 
                and isinstance(data["data"], dict)
                and "bids" in data["data"] 
                and "asks" in data["data"]
            ):
                bids = [(float(bid['price']), float(bid['volume'])) for bid in data["data"]["bids"]]
                asks = [(float(ask['price']), float(ask['volume'])) for ask in data["data"]["asks"]]
                self.current_orderbook["bids"] = bids
                self.current_orderbook["asks"] = asks

                # Update your orderbook plot
                self.plotter.update_data(bids, asks)
                
            # 2) Tick updates
            #   Suppose your tick data has cmd_id=22998. Adjust as needed:
            if cmd_id == 22005:
                print(data,'tick_data')
                # Or whatever your actual data structure is
                self.tick_data_list.append(data)
                # Possibly truncate the list to some max length
                if len(self.tick_data_list) > 500:
                    self.tick_data_list.pop(0)
                print(f"Received a tick. total ticks: {len(self.tick_data_list)}")

        except Exception as e:
            print("❌ on_message error:", e)

    def on_error(self, ws, error):
        """Called on any WS error."""
        print("❌ SingleFeed WebSocket error:", error)
        self.reconnect()

    def on_close(self, ws, close_status_code, close_msg):
        """Called when WS is closed."""
        print("❌ SingleFeed WebSocket closed:", close_status_code, close_msg)
        self.reconnect()

    def thread_heartbeat(self):
        """Send periodic heartbeat on the same connection."""
        while True:
            time.sleep(10)
            if self.ws and self.ws.sock and self.ws.sock.connected:
                heartbeat = {
                    "cmd_id": 22000,  # Heartbeat
                    "seq_id": 999,
                    "trace": "heartbeat",
                    "data": {}
                }
                try:
                    self.ws.send(json.dumps(heartbeat))
                    print("Sent heartbeat")
                except Exception as e:
                    print("❌ Heartbeat send failed:", e)

    def reconnect(self):
        """Exponential backoff for reconnects."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print("Reached max reconnect attempts, restarting script.")
            self.restart_script()

        wait_time = 2 ** self.reconnect_attempts
        print(f"🔄 SingleFeed reconnecting in {wait_time} seconds...")
        time.sleep(wait_time)

        self.reconnect_attempts += 1
        self.start()

    def restart_script(self):
        """If absolutely necessary. Otherwise, remove or limit usage."""
        python = sys.executable
        os.execl(python, [python] + sys.argv)



# 初始化绘图工具
plotter = OrderBookPlotter()

# 初始化 WebSocket 订阅
feed = SingleFeed(plotter)
#feed2 = FeedDisplay()
# 启动 WebSocket
@app.route('/')
@requires_auth
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
@app.route('/tick_data')
@requires_auth
def tick_data():
    return jsonify(list(tick_data_list))

@app.route('/tick_data_view')
@requires_auth
def tick_data_view():
    return render_template('tick_data_view.html')

if __name__ == '__main__':
    # def run_ws():
    #     feed.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    # threading.Thread(target=run_ws).start()
    threading.Thread(target=feed.start).start()
    #threading.Thread(target=lambda: feed2.start()).start()
    app.run(host='0.0.0.0', port=8153,threaded=True)

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
import klineManager
import indicator
import decision
from flask import request, jsonify
import datetime
from decision import enhanced_strategy_decision,backtest_strategy
import pandas as pd
from trend import generate_trend_report, resample_to_multiple_timeframes, calculate_stop_loss, calculate_dynamic_position_sizing

code = 'XAUUSD'
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
    return username == 'tmgm' or username == 'joyce' and password == 'glzdggls' or password == 'joycepw'

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
                    {"code": code, "depth_level": 10}
                ]
            }
        }
        ws.send(json.dumps(sub_param))
        print("📡 订阅深度报价数据成功！")
        threading.Thread(target=self.thread_heartbeat).start()

    def on_message(self, ws, message):
        """ 处理 WebSocket 返回的市场深度数据 """
        try:
            data = json.loads(message)
            #print(data,'ob')
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
    def thread_heartbeat(self):
        while True:
            time.sleep(10)  # 每 10 秒发送一次心跳
            if self.ws.sock and self.ws.sock.connected:
                heartbeat = {
                    "cmd_id":22000,
                    "seq_id":123,
                    "trace":"asdfsdfa",
                    "data":{
                    }
                }
                self.ws.send(json.dumps(heartbeat))  # 发送心跳消息
                print("Sent heartbeat")

    def restart_script(self):
        """ 重启脚本 """
        python = sys.executable
        os.execl(python, [python] + sys.argv)
class FeedDisplay:
    """
    处理第二个 WebSocket 连接，
    接收到的数据不做处理，仅保存原始返回数据供展示使用。
    """
    def __init__(self):
        # 替换为你的第二个 WS URL
        self.url = 'wss://quote.alltick.io/quote-b-ws-api?token=949751a4ea6a586f9e2805a3909d456a-c-app'
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

    def reconnect(self):
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print("第二个 WS 达到最大重连次数，重启脚本")
            self.restart_script()
        wait_time = 2 ** self.reconnect_attempts
        print(f"🔄 第二个 WS 尝试在 {wait_time} 秒后重连...")
        time.sleep(wait_time)
        self.reconnect_attempts += 1
        self.start()

    def on_error(self, ws, error):
        print(f"❌ 第二个 WS 发生错误: {error}")
        self.reconnect()

    def on_close(self, ws, close_status_code, close_msg):
        print("❌ 第二个 WS 连接已关闭！")
        self.reconnect()

    def on_open(self, ws):
        print("✅ 第二个 WS 连接成功！")
        self.reconnect_attempts = 0
        # 如果需要发送订阅消息，可按需配置
        sub_param = {
            "cmd_id": 22004,
            "seq_id": 125,
            "trace": "WS-Tick",
            "data": {
                "symbol_list": [
                    {"code": code}
                ]
            }
        }
        ws.send(json.dumps(sub_param))
        print("📡 第二个 WS 订阅消息已发送（如果需要）")
        threading.Thread(target=self.thread_heartbeat).start()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get("cmd_id") == 22998:
                #print(data,'tick_data')
                add_tick_data(data)

        except Exception as e:
            print(f"❌ 解析错误: {e}")
        

    def start(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def restart_script(self):
        python = sys.executable
        os.execl(python, [python] + sys.argv)
    def thread_heartbeat(self):
        while True:
            time.sleep(10)  # 每 10 秒发送一次心跳
            if self.ws.sock and self.ws.sock.connected:
                heartbeat = {
                    "cmd_id":22000,
                    "seq_id":123,
                    "trace":"asdfsdfa",
                    "data":{
                    }
                }
                self.ws.send(json.dumps(heartbeat))  # 发送心跳消息
                print("Sent heartbeat")



# 初始化绘图工具
plotter = OrderBookPlotter()
data_manager = klineManager.KlineDataManager(symbol=code)
data_manager.fetch_historical_data()

# 初始化 WebSocket 订阅
feed = Feed(plotter)
feed2 = FeedDisplay()
# 启动 WebSocket
@app.route('/')
@requires_auth
def index():
    return render_template('d3_orderbook.html')
@app.route('/kline')
def kline():
    # 返回最新10条K线数据的HTML表格
    with data_manager.lock:
        html_table = data_manager.df.tail(10).to_html()
    return f"<h1>最新K线数据</h1>{html_table}"

@app.route('/data.json')
def data_json():
    # 返回最新10条K线数据的JSON格式
    with data_manager.lock:
        data_json = data_manager.df.tail(10).to_json(date_format='iso')
    return data_json

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
@app.route('/decision')
def decision():
    # Get the date parameter from the query string; default to today if not provided.
    date_str = request.args.get('date')
    if not date_str:
        date_str = datetime.datetime.today().strftime('%Y-%m-%d')
    
    # Get the language parameter, default to Chinese
    #language = request.args.get('lang', 'cn')
    
    # Filter the historical DataFrame for the given date.
    with data_manager.lock:
        try:
            # Assuming your DataFrame index is a DatetimeIndex
            df_day = data_manager.df.loc[date_str]
            # Convert Series to DataFrame if needed
            if isinstance(df_day, pd.Series):
                df_day = df_day.to_frame().T
        except Exception as e:
            return f"<h3>该日期 {date_str} 没有数据. 错误: {e}</h3>"
    
    # Call the enhanced decision function without order book
    decision_result = enhanced_strategy_decision(df_day)
    print(f"DEBUG: 决策结果中止损位: {decision_result.get('enhanced_decision', {}).get('stop_loss')}")
    print(f"DEBUG: 决策结果中仓位: {decision_result.get('enhanced_decision', {}).get('suggested_position')}")
    # Render the decision page using the template.
    return render_template("decision.html", decision=decision_result, date=date_str)
@app.route('/backtest')
@requires_auth
def backtest_page():
    """回测页面路由"""
    return render_template('backtest.html')

@app.route('/api/backtest', methods=['POST'])
@requires_auth
def run_backtest_api():
    """运行回测的API端点"""
    # 获取请求参数
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    initial_capital = float(data.get('initial_capital', 10000))
    leverage = float(data.get('leverage', 100))
    risk_percent = float(data.get('risk_percent', 2))
    
    # 验证输入
    if not start_date or not end_date:
        return jsonify({"error": "需要提供开始和结束日期"}), 400
    
    try:
        # 获取历史数据
        with data_manager.lock:
            df = data_manager.df.copy()
        
        # 先检查是否有数据
        if df.empty:
            return jsonify({"error": "没有可用的历史数据"}), 400
            
        # 检查日期范围是否有效
        df_start = df.index.min().strftime('%Y-%m-%d')
        df_end = df.index.max().strftime('%Y-%m-%d')
        
        if pd.to_datetime(start_date) < pd.to_datetime(df_start):
            start_date = df_start
            
        if pd.to_datetime(end_date) > pd.to_datetime(df_end):
            end_date = df_end
        
        # 输出调试信息
        print(f"回测日期范围: {start_date} 至 {end_date}")
        print(f"数据范围: {df_start} 至 {df_end}")
        print(f"数据点数: {len(df)}")
        
        # 运行回测
        results = backtest_strategy(df, start_date, end_date, initial_capital, leverage, risk_percent)
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"回测执行出错: {str(e)}"}), 500

@app.route('/api/available-dates')
@requires_auth
def get_available_dates():
    """获取可用于回测的日期范围"""
    try:
        with data_manager.lock:
            # 确保数据不为空
            if data_manager.df.empty:
                return jsonify({"error": "没有可用的历史数据", "dates": []}), 200
                
            dates = data_manager.df.index.strftime('%Y-%m-%d').unique().tolist()
        return jsonify({"dates": dates})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "dates": []}), 200
@app.route('/trend_analysis')
def trend_analysis_page():
        """趋势分析页面 - 分析整个数据集"""
        try:
            # 获取历史数据
            with data_manager.lock:
                df = data_manager.df.copy()
            if df.empty:
                return "<h3>没有可用的数据</h3>"
            
            # 如果数据量太大，可以选择最近的一部分数据进行分析
            
            print(f"分析数据范围: {df.index.min()} 至 {df.index.max()}, 共 {len(df)} 个数据点")
            
            # 生成趋势报告
            report = generate_trend_report(df)
            
            # 渲染模板
            date_range = f"{df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}"
            return render_template(
                "trend_analysis.html", 
                date_range=date_range,
                trend_analysis=report["trend_analysis"],
                visualization=report["visualization"]
            )
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            return f"<h3>趋势分析过程中出错: {e}</h3><pre>{error_traceback}</pre>"
@app.route('/risk_management')
@requires_auth
def risk_management_page():
    """风险管理页面 - 提供止损计算工具"""
    try:
        # Add current datetime to template context
        from datetime import datetime
        now = datetime.now()
        
        # 获取最新价格作为默认入场价
        with data_manager.lock:
            latest_data = data_manager.df.iloc[-1]
            latest_price = latest_data['Close']
        
        # 获取趋势分析结果以设置默认交易方向
        with data_manager.lock:
            df = data_manager.df.copy()
        
        # 生成趋势报告
        report = generate_trend_report(df)
        trend_analysis = report["trend_analysis"]
        
        # 渲染风险管理模板
        return render_template(
            "risk_management.html",
            latest_price=latest_price,
            trend_analysis=trend_analysis,
            now=now  # Pass the current datetime to the template
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        return f"<h3>加载风险管理页面时出错: {e}</h3><pre>{error_traceback}</pre>"

@app.route('/api/calculate_stop_loss', methods=['POST'])
@requires_auth
def api_calculate_stop_loss():
    """计算止损的API端点"""
    try:
        data = request.json
        
        # 获取参数
        entry_price = float(data.get('entry_price'))
        direction = data.get('direction', 'long')
        risk_percentage = float(data.get('risk_percentage', 1.0))
        account_size = float(data.get('account_size', 10000))
        leverage = float(data.get('leverage', 500))
        position_percentage = data.get('position_percentage')
        
        if position_percentage is not None:
            position_percentage = float(position_percentage)
        
        # 使用之前在trend.py中定义的函数计算止损
        result = calculate_stop_loss(
            entry_price=entry_price,
            direction=direction,
            risk_percentage=risk_percentage,
            account_size=account_size,
            leverage=leverage,
            position_percentage=position_percentage
        )
        
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"计算止损时出错: {str(e)}"}), 500

@app.route('/api/calculate_dynamic_stop_loss', methods=['POST'])
@requires_auth
def api_calculate_dynamic_stop_loss():
    """基于ATR计算动态止损的API端点"""
    try:
        data = request.json
        
        # 获取参数
        entry_price = float(data.get('entry_price'))
        direction = data.get('direction', 'long')
        risk_percentage = float(data.get('risk_percentage', 1.0))
        account_size = float(data.get('account_size', 10000))
        leverage = float(data.get('leverage', 500))
        atr_multiplier = float(data.get('atr_multiplier', 1.5))
        timeframe = data.get('timeframe', '1H')
        
        # 获取相应时间框架的数据
        with data_manager.lock:
            df_dict = resample_to_multiple_timeframes(data_manager.df.copy())
            df = df_dict.get(timeframe)
            
            if df is None or df.empty:
                return jsonify({"error": f"无法获取{timeframe}时间框架的数据"}), 400
        
        # 使用之前在trend.py中定义的函数计算动态止损
        result = calculate_dynamic_position_sizing(
            df=df,
            entry_price=entry_price,
            direction=direction,
            risk_percentage=risk_percentage,
            account_size=account_size,
            atr_periods=14,
            atr_multiplier=atr_multiplier,
            leverage=leverage
        )
        
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"计算动态止损时出错: {str(e)}"}), 500

if __name__ == '__main__':
    # def run_ws():
    #     feed.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    # threading.Thread(target=run_ws).start()
    threading.Thread(target=lambda: feed.start()).start()
    threading.Thread(target=lambda: feed2.start()).start()
    threading.Thread(target=data_manager.background_update).start()
    app.run(host='0.0.0.0', port=5000,threaded=True)

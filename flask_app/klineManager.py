import requests
import json
import time
import datetime
import threading
import pandas as pd
import indicator
import decision

class KlineDataManager:
    def __init__(self, symbol='GOLD', api_token='949751a4ea6a586f9e2805a3909d456a-c-app'):
        self.symbol = symbol
        self.api_token = api_token
        self.headers = {'Content-Type': 'application/json'}
        self.df = pd.DataFrame()  # 用来存储所有K线数据
        self.lock = threading.Lock()  # 数据更新时加锁

    def fetch_historical_data(self):
        """批量获取历史K线数据"""
        all_data = []
        end_timestamp = 0  # 从最新数据开始查询
        for i in range(10):
            print("当前 end_timestamp:", end_timestamp)
            params = {
                "trace": "python_http_test1",
                "data": {
                    "code": self.symbol,
                    "kline_type": 1,           # 1分钟K线
                    "kline_timestamp_end": end_timestamp,
                    "query_kline_num": 1000,   # 每次最多1000根
                    "adjust_type": 0           # 不复权
                }
            }
            query = json.dumps(params)
            query = requests.utils.quote(query)
            url = f'https://quote.alltick.io/quote-b-api/kline?token={self.api_token}&query={query}'

            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    current_batch = data['data'].get('kline_list', [])
                    if current_batch:
                        all_data.extend(current_batch)
                        end_timestamp = current_batch[0]['timestamp']
                    else:
                        break
                else:
                    print(f"请求失败，状态码：{response.status_code}")
            except Exception as e:
                print(f"请求出现异常：{e}")
            time.sleep(1)
        # 构造 DataFrame
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
        df.set_index('timestamp', inplace=True)
        for col in ['open_price', 'close_price', 'high_price', 'low_price', 'volume']:
            df[col] = df[col].astype(float)
        df.rename(columns={
            'open_price': 'Open',
            'close_price': 'Close',
            'high_price': 'High',
            'low_price': 'Low',
            'volume': 'Volume'
        }, inplace=True)
        df.sort_index(inplace=True)
        print("历史数据获取完毕，记录数：", len(df))
        with self.lock:
            self.df = df.copy()
        return self.df

    def get_current_minute_timestamp(self):
        """获取当前分钟级的Unix时间戳（秒级，秒置为0）"""
        now = datetime.datetime.now()
        return int(time.mktime(now.timetuple()) // 60 * 60)

    def fetch_new_kline(self):
        """获取最新的一根K线数据"""
        end_timestamp = self.get_current_minute_timestamp()
        params = {
            "trace": "python_http_test1",
            "data": {
                "code": self.symbol,
                "kline_type": 1,
                "kline_timestamp_end": end_timestamp,
                "query_kline_num": 1,
                "adjust_type": 0
            }
        }
        query = json.dumps(params)
        query = requests.utils.quote(query)
        url = f'https://quote.alltick.io/quote-b-api/kline?token={self.api_token}&query={query}'
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                kline_list = data['data'].get('kline_list', [])
                if kline_list:
                    return kline_list[0]
                else:
                    return None
            else:
                print("请求失败，状态码：", response.status_code)
                return None
        except Exception as e:
            print("请求出现异常：", e)
            return None

    def insert_new_kline(self, new_k):
        """将新K线数据转换并追加到df中"""
        new_df = pd.DataFrame([new_k])
        for col in ['open_price', 'close_price', 'high_price', 'low_price', 'volume']:
            new_df[col] = new_df[col].astype(float)
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'].astype(int), unit='s')
        new_df.rename(columns={
            'open_price': 'Open',
            'close_price': 'Close',
            'high_price': 'High',
            'low_price': 'Low',
            'volume': 'Volume'
        }, inplace=True)
        new_df = new_df[['timestamp', 'Open', 'Close', 'High', 'Low', 'Volume']]
        new_df.set_index('timestamp', inplace=True)
        with self.lock:
            self.df = pd.concat([self.df, new_df])
            self.df.sort_index(inplace=True)
        return self.df

    def background_update(self):
        """后台线程，每隔1分钟更新一次最新的K线数据"""
        while True:
            new_k = self.fetch_new_kline()
            if new_k:
                print("获取到新K线数据：", new_k)
                self.insert_new_kline(new_k)
            else:
                print("本次未获取到新数据")
            time.sleep(60)

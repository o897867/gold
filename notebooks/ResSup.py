# %%





# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'PingFang HK', 'Microsoft YaHei', 'SimSun']  # 优先级从左到右
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

df = pd.read_csv('data/init.csv')

df.set_index('timestamp', inplace=True)





# %%
import json
import time
import threading
import websocket    # pip install websocket-client

'''
# Special Note:
# GitHub: https://github.com/alltick/realtime-forex-crypto-stock-tick-finance-websocket-api
# Token Application: https://alltick.co
# Replace "testtoken" in the URL below with your own token
# API addresses for forex, cryptocurrencies, and precious metals:
# wss://quote.tradeswitcher.com/quote-b-ws-api
# Stock API address:
# wss://quote.tradeswitcher.com/quote-stock-b-ws-api
'''

class Feed(object):

    def __init__(self):
        self.url = 'wss://quote.tradeswitcher.com/quote-b-ws-api?token=673a0e6c1d656cea9a7a7b341c76008f-c-app'  # Enter your websocket URL here
        self.ws = None

    def on_open(self, ws):
        """
        Callback object which is called at opening websocket.
        1 argument:
        @ ws: the WebSocketApp object
        """
        print('A new WebSocketApp is opened!')

        # Start subscribing (an example)
        sub_param = {
            "cmd_id": 22002,
            "seq_id": 123,
            "trace":"3baaa938-f92c-4a74-a228-fd49d5e2f8bc-1678419657806",
            "data":{
                "symbol_list":[
                    {
                        "code": "GOLD",
                        "depth_level": 5,
                    }
                ]
            }
        }

        # If you want to run for a long time, you need to modify the code to send heartbeats periodically to avoid disconnection, please refer to the API documentation for details
        sub_str = json.dumps(sub_param)
        ws.send(sub_str)
        print("depth quote are subscribed!")

    def on_data(self, ws, string, type, continue_flag):
        """
        4 arguments.
        The 1st argument is this class object.
        The 2nd argument is utf-8 string which we get from the server.
        The 3rd argument is data type. ABNF.OPCODE_TEXT or ABNF.OPCODE_BINARY will be came.
        The 4th argument is continue flag. If 0, the data continue
        """

    def on_message(self, ws, message):
        """
        Callback object which is called when received data.
        2 arguments:
        @ ws: the WebSocketApp object
        @ message: utf-8 data received from the server
        """
        # Parse the received message
        result = eval(message)
        print(result)

    def on_error(self, ws, error):
        """
        Callback object which is called when got an error.
        2 arguments:
        @ ws: the WebSocketApp object
        @ error: exception object
        """
        print(error)

    def on_close(self, ws, close_status_code, close_msg):
        """
        Callback object which is called when the connection is closed.
        2 arguments:
        @ ws: the WebSocketApp object
        @ close_status_code
        @ close_msg
        """
        print('The connection is closed!')

    def start(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_data=self.on_data,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        threading.Thread(target=self.thread_heartbeat).start()
        self.ws.run_forever()
    
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
                self.ws.send(heartbeat)  # 发送心跳消息
                print("Sent heartbeat")


if __name__ == "__main__":
    feed = Feed()
    feed.start()

# %%
def calculate_atr(df, period=14):
    # 计算三种价格范围
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    # 真实范围取最大值
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    # 使用 Wilder 的平滑方法计算 ATR
    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[period-1] = true_range[:period].mean()  # 第一个ATR值使用简单平均

    # 使用 Wilder 的平滑方法计算后续值
    for i in range(period, len(df)):
        atr.iloc[i] = (atr.iloc[i-1] * (period-1) + true_range.iloc[i]) / period

    return atr

# 计算 ATR
df['ATR'] = calculate_atr(df)
df['ATR_Pct'] = (df['ATR'] / df['close']) * 100

# %% [markdown]
# ## 简易支撑压力 ##
# 

# %%
# 支撑位：最近N根K线的最低价（动态窗口）
def dynamic_support(data, window=30):
    return data['low'].rolling(window).min()

# 阻力位：最近N根K线的最高价（动态窗口）
def dynamic_resistance(data, window=30):
    return data['high'].rolling(window).max()

# 加入波动率调整
df['support'] = dynamic_support(df) * (1 - 0.1*df['ATR_Pct'])
df['resistance'] = dynamic_resistance(df) * (1 + 0.1*df['ATR_Pct'])

# %% [markdown]
# ## 动量压缩比 ##

# %%
def calculate_momentum_compression_ratio(dataframe, short_window, long_window):
    dataframe['Short_Term_Momentum'] = dataframe['close'].diff(periods=short_window)
    dataframe['Long_Term_Momentum'] = dataframe['close'].diff(periods=long_window)
    dataframe['Momentum_Compression_Ratio'] = dataframe['Short_Term_Momentum'] / dataframe['Long_Term_Momentum']
    return dataframe

# 设置参数，例如短期5分钟，长期15分钟
df = calculate_momentum_compression_ratio(df, short_window=5, long_window=15)


# %% [markdown]
# ## 弹性 ##

# %%
def calculate_elasticity_coefficient(dataframe, window):
    roll_max = dataframe['high'].rolling(window=window, min_periods=1).max()
    roll_min = dataframe['low'].rolling(window=window, min_periods=1).min()
    dataframe['Elasticity'] = (dataframe['close'] - roll_min) / (roll_max - roll_min)
    return dataframe

# 应用函数，例如使用14分钟作为窗口
df = calculate_elasticity_coefficient(df, window=14)


# %% [markdown]
# ## 压力 ##

# %%
def calculate_pressure_accumulation(dataframe):
    # 根据收盘价是否高于开盘价来决定多空力量
    dataframe['Pressure'] = np.where(dataframe['close'] > dataframe['open'],
                                     dataframe['volume'], -dataframe['volume'])
    # 累积多空力量
    dataframe['Cumulative_Pressure'] = dataframe['Pressure'].cumsum()

    return dataframe

# 应用函数到实际数据
df = calculate_pressure_accumulation(df)

#


# %% [markdown]
# ## Hurst ##
# 

# %%
from hurst import compute_Hc

def calculate_fractal_dimension(dataframe, window):
    H, c, data = compute_Hc(dataframe['close'].values, kind='price', simplified=True)
    D = 2 - H
    return H, D

# 计算示例数据集的Hurst指数和分形维度
H, D = calculate_fractal_dimension(df, window=60)
print("Hurst指数:", H)
print("分形维度:", D)

# %% [markdown]
# ## RSI ##
# 

# %%


def calculate_rsi_wilder(df, period=14):
    # 计算价格变化
    delta = df['close'].diff()

    # 分别获取上涨和下跌
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's EMA 使用 alpha = 1/period
    # 首次计算简单平均值
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()

    # 准备存放 RSI 的序列
    rsi_series = pd.Series(index=df.index, dtype=float)

    # 第一个 RSI 值
    rsi_series.iloc[period] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # 使用 Wilder 的平滑方法计算后续值
    for i in range(period + 1, len(df)):
        avg_gain = ((avg_gain * (period - 1)) + gain.iloc[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + loss.iloc[i]) / period
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_series.iloc[i] = 100 - (100 / (1 + rs))
        else:
            rsi_series.iloc[i] = 100

    return rsi_series

# 计算 Wilder's RSI

df['RSI_Wilder'] = calculate_rsi_wilder(df)



# %%
df.columns

# %% [markdown]
# ## ML ##

# %%
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 检查并处理缺失值
if df.isnull().any().any():
    # 可以选择填充缺失值或删除含有缺失值的行
    df.fillna(method='ffill', inplace=True)  # 向前填充
    # 或者 data.dropna(inplace=True)  # 删除含有缺失值的行

# 对数变换（对适合的数值列）
for col in ['volume', 'transactions', 'ATR', 'ATR_Pct']:
    # 仅对正数进行对数变换
    df[col] = np.log1p(df[col].clip(lower=0))

# 标准化
scaler = StandardScaler()
features = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'ATR', 'ATR_Pct', 'Short_Term_Momentum', 'Long_Term_Momentum', 'Momentum_Compression_Ratio', 'Elasticity', 'Pressure', 'Cumulative_Pressure', 'RSI_Wilder']
df[features] = scaler.fit_transform(df[features])

# %%
# 按照时间戳索引升序排序
df.sort_index(ascending=True, inplace=True)
df.tail()


# %%
df.dropna(inplace=True)
print(df.isnull().sum())

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 假设 data 是已经加载的DataFrame
# 提取特征和目标
features = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'ATR', 'ATR_Pct', 'Short_Term_Momentum', 'Long_Term_Momentum', 'Momentum_Compression_Ratio', 'Elasticity', 'Pressure', 'Cumulative_Pressure', 'RSI_Wilder']
X = df[features]
y = df['support']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 评估模型
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Cross-validated MSE:", -scores.mean())

# 测试集上的性能
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)


# %% [markdown]
# ### RF ###

# %%


# %%
len(df)

# %%
import cudf
import cuml
from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.model_selection import train_test_split
from cuml.metrics import mean_squared_error

# 假设 data 是已经加载并预处理的 DataFrame
# 将 pandas DataFrame 转换为 cuDF DataFrame
gdf = cudf.DataFrame.from_pandas(df)

# 划分数据集
X = df[features]
y = df['support']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
rf = cuRF(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设X和y已经是预处理并准备好的数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 获取特征重要性
importances = rf.feature_importances_

# 显示特征重要性
feature_names = X.columns
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

# 预测并计算MSE
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)


# %% [markdown]
# ## DS ##

# %%
def calculate_missing_features(df):
    """
    计算缺失的特征
    """
    # 1. 分形维度计算
    
    
    # 3. 赫斯特指数计算
    def calculate_hurst(price_series, lags=range(2, 20)):
        tau = []
        for lag in lags:
            tau.append(np.std(price_series.diff(lag)))
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]
    
    # 使用滚动窗口计算特征
    window = 100  # 可以调整窗口大小
    
    # 计算分形维度

    
    # 计算赫斯特指数
    df['hurst'] = df['close'].rolling(window).apply(
        lambda x: calculate_hurst(x)
    )
    
    return df
calculate_missing_features(df)

# %%
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# 计算5分钟和60分钟滚动波动率
df['volatility'] = df['log_return'].rolling(window=5).std()

# %%
df

# %%
import pandas as pd
from scipy import stats

def dynamic_levels(data, window=30, confidence=0.95):
    """
    返回带置信区间的动态支撑阻力位
    """
    # 计算基础水平
    support_base = data['low'].rolling(window).min()
    resist_base = data['high'].rolling(window).max()

    # 计算置信区间的辅助函数
    def calc_interval(row):
        if pd.isna(row['base']) or pd.isna(row['std']):
            return pd.NA, pd.NA
        ci = stats.t.interval(confidence, window-1, loc=row['base'], scale=row['std'])
        return ci[0], ci[1]

    # 合并基础和标准差，逐行应用 calc_interval 函数
    support_data = pd.DataFrame({
        'base': support_base,
        'std': data['low'].rolling(window).std()
    })
    resist_data = pd.DataFrame({
        'base': resist_base,
        'std': data['high'].rolling(window).std()
    })

    support_ci = support_data.apply(calc_interval, axis=1)
    resist_ci = resist_data.apply(calc_interval, axis=1)

    # 处理结果，分解为上下限
    support_ci_low = support_ci.apply(lambda x: x[0])
    support_ci_high = support_ci.apply(lambda x: x[1])

    resist_ci_low = resist_ci.apply(lambda x: x[0])
    resist_ci_high = resist_ci.apply(lambda x: x[1])

    return {
        'support': (support_base, (support_ci_low, support_ci_high)),
        'resistance': (resist_base, (resist_ci_low, resist_ci_high))
    }

# 示例使用
levels = dynamic_levels(df)
current_support = levels['support'][0].iloc[-1]
current_resistance = levels['resistance'][0].iloc[-1]
support_ci_low, support_ci_high = levels['support'][1][0].iloc[-1], levels['support'][1][1].iloc[-1]


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

class BreakProbabilityModel:
    def __init__(self):
        self.model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
        self.features = ['distance', 'volatility', 'volume_ratio', 'momentum']

    def calculate_features(self, data):
        """
        计算突破相关特征
        """
        # 当前价格到支撑/阻力的距离（标准化）
        data['distance'] = (data['close'] - current_support) / (current_resistance - current_support)

        # 波动率冲击
        data['vol_shock'] = data['ATR_Pct'] / data['ATR_Pct'].rolling(50).mean()

        # 量能比率
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(30).mean()

        # 动量强度
        data['momentum'] = data['close'].pct_change(5)
        return data

    def fit(self, X, y):
        self.model.fit(X[self.features], y)

    def explain_break(self, X):
        # 计算特征重要性
        result = permutation_importance(self.model, X[self.features], X['break'], n_repeats=10)
        return {f: i for f, i in zip(self.features, result.importances_mean)}

    def monte_carlo_simulation(self, current_state, n_sims=1000):
        """
        蒙特卡洛模拟突破概率
        """
        paths = []
        for _ in range(n_sims):
            # 基于当前波动率和动量生成路径
            path = current_state['close'] * np.exp(np.cumsum(
                np.random.normal(current_state['momentum'], current_state['volatility'], 10)
            ))
            paths.append(path)
        # 计算突破次数
        break_count = sum([max(path) > current_resistance or min(path) < current_support for path in paths])
        return break_count / n_sims

# 初始化模型
bpm = BreakProbabilityModel()
data = bpm.calculate_features(df)


# %%
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_fractal_dim(price_series, box_sizes=np.logspace(1, 3, 10)):
    counts = []
    for box_size in box_sizes:
        box_count = (max(price_series) - min(price_series)) / box_size
        counts.append(box_count)
    
    log_box_sizes = np.log(box_sizes)
    log_counts = np.log(counts)
    
    model = LinearRegression()
    model.fit(log_box_sizes.reshape(-1, 1), log_counts)
    return model




# %%
m1

# %%
import pandas as pd

# 假设你已经有了这些特征的值


# %%
df


# %%
from sklearn.ensemble import GradientBoostingRegressor

class NextLevelPredictor:
    def __init__(self):
        self.support_model = GradientBoostingRegressor()
        self.resist_model = GradientBoostingRegressor()
        self.features = ['RSI_Wilder', 'vol_shock', 'Pressure', 'ATR']

    def fit(self, X, y_support, y_resist):
        self.support_model.fit(X[self.features], y_support)
        self.resist_model.fit(X[self.features], y_resist)

    def predict_next_level(self, X, break_direction):
        if break_direction == 'up':
            return self.resist_model.predict(X[self.features])[0] * 1.02  # 2%过滤
        elif break_direction == 'down':
            return self.support_model.predict(X[self.features])[0] * 0.98
        else:
            return None



# %%
def break_significance_test(data, level_type='support'):
    """
 使用Wilcoxon符号秩检验验证突破的有效性
    """
    if level_type == 'support':
        test_data = data[data['close'] < current_support]
        test_values = current_support - test_data['close']
    else:
        test_data = data[data['close'] > current_resistance]
        test_values = test_data['close'] - current_resistance
    
    # 检验价格是否持续停留在突破区域
    return stats.wilcoxon(test_values).pvalue < 0.05

# 示例
is_valid_break = break_significance_test(df[-30:], 'resistance')

# %%
def price_distribution_test(data, level, window=30):
    """
 使用KS检验比较当前价格分布与历史分布的差异
    """
    historical = data['close'].rolling(window).apply(lambda x: stats.ks_2samp(x, data['close'].iloc[-window:])[0])
    current_stat = stats.ks_2samp(data['close'].iloc[-window:], data['close'])[0]
    return current_stat > np.percentile(historical, 95)

# 检测当前价格分布是否发生显著变化
distribution_changed = price_distribution_test(df, current_support)

# %%
class BreakAnalysisSystem:
    def __init__(self):
        self.levels = dynamic_levels(df)
        self.prob_model = BreakProbabilityModel()
        self.next_level_model = NextLevelPredictor()
        
    def run_analysis(self):
        # 计算突破概率
        current_state = {
            'close': df['close'].iloc[-1],
            'momentum': df['momentum'].iloc[-1],
            'volatility': df['volatility'].iloc[-1]
        }
        prob_break = self.prob_model.monte_carlo_simulation(current_state)
        
        # 统计检验
        significance = break_significance_test(df[-30:])
        
        # 生成解释
        explanation = self.generate_explanation(prob_break, significance)
        
        # 预测下一级位
        if prob_break > 0.7:
            direction = 'up' if df['close'].iloc[-1] > current_resistance else 'down'
            next_level = self.next_level_model.predict_next_level(df.iloc[-1], direction)
        else:
            next_level = None
            
        return {
            'current_support': current_support,
            'current_resistance': current_resistance,
            'break_probability': prob_break,
            'statistical_significance': significance,
            'next_level': next_level,
            'explanation': explanation
        }
    
    def generate_explanation(self, prob, sig):
        factors = []
        if df['vol_shock'].iloc[-1] > 2:
            factors.append("当前波动率是平均水平的2倍以上")
        if df['volume_ratio'].iloc[-1] > 1.5:
            factors.append("成交量超过近期均值50%")
        if df['momentum'].iloc[-1] > 0.03:
            factors.append("存在强劲的短期动量")
            
        explanation = f"突破概率{prob:.0%}主要由以下因素驱动：\n" + "\n".join(factors)
        if sig:
            explanation += "\n统计检验确认突破有效性"
        else:
            explanation += "\n当前突破尚未通过统计显著性检验"
        return explanation

# 执行分析
bas = BreakAnalysisSystem()
result = bas.run_analysis()
print(result)

# %% [markdown]
# ## DS 2 ##

# %%
from scipy.stats import linregress

def calculate_features(df):
    """
    计算支撑阻力相关的特征
    """
    # 1. 分形维度
    def fractal_dim(series, window=100):
        return series.rolling(window).apply(
            lambda x: linregress(np.log(range(1, len(x)+1)), np.log(x)).slope
        )
    
    # 2. 赫斯特指数
    def hurst_exponent(series, window=100):
        return series.rolling(window).apply(
            lambda x: linregress(np.log(range(1, len(x)+1)), np.log(x)).slope
        )
    
    # 3. 压力值
    def pressure(price, volume):
        return np.where(price.diff() > 0, volume, -volume).cumsum()
    
    # 4. 波动率聚类
    def vol_cluster(returns, window=20):
        return (returns.abs() > returns.rolling(window).mean()).rolling(3).sum()
    
    # 5. ATR（真实波动幅度）
    def atr(df, window=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(window).mean()
    df['ma'] = df['close'].rolling(window=20).mean()
    df['upper_band'] = df['ma'] + 2 * df['std']
    df['lower_band'] = df['ma'] - 2 * df['std']
    # 应用特征计算
    df['fractal_dim'] = fractal_dim(df['close'])
    df['hurst'] = hurst_exponent(df['close'])
    df['pressure'] = pressure(df['close'], df['volume'])
    df['vol_cluster'] = vol_cluster(df['close'].pct_change())
    df['atr'] = atr(df)
    
    return df

# 示例使用
df = calculate_features(df)

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

class SupportResistanceModel:
    def __init__(self):
        self.support_model = GradientBoostingRegressor()
        self.resist_model = GradientBoostingRegressor()
        self.features = ['fractal_dim', 'vol_cluster', 'pressure', 'hurst', 'atr']
        
    def fit(self, X, y_support, y_resist):
        """训练模型"""
        self.support_model.fit(X[self.features], y_support)
        self.resist_model.fit(X[self.features], y_resist)
        
    def predict(self, X, break_direction):
        """预测支撑位或阻力位"""
        if break_direction == 'up':
            return self.resist_model.predict(X[self.features])[0] * 1.02  # 2% 过滤
        elif break_direction == 'down':
            return self.support_model.predict(X[self.features])[0] * 0.98
        else:
            raise ValueError("break_direction 必须是 'up' 或 'down'")

# 示例使用
X = df[['fractal_dim', 'vol_cluster', 'pressure', 'hurst', 'atr']].dropna()
y_support = df['low'].shift(-1).dropna()
y_resist = df['high'].shift(-1).dropna()

X_train, X_test, y_support_train, y_support_test, y_resist_train, y_resist_test = train_test_split(
    X, y_support, y_resist, test_size=0.2, shuffle=False
)

model = SupportResistanceModel()
model.fit(X_train, y_support_train, y_resist_train)

# %%
def predict_next_level(df, model, break_direction):
    """预测下一个支撑位或阻力位"""
    current_features = df.iloc[-1][model.features].to_frame().T
    return model.predict(current_features, break_direction)

# 示例使用
next_support = predict_next_level(df, model, 'down')
next_resistance = predict_next_level(df, model, 'up')
print(f"下一个支撑位: {next_support}, 下一个阻力位: {next_resistance}")

# %%
import matplotlib.pyplot as plt

def plot_support_resistance(df, support, resistance):
    """绘制支撑阻力线"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    plt.axhline(y=support, color='green', linestyle='--', label='Support')
    plt.axhline(y=resistance, color='red', linestyle='--', label='Resistance')
    plt.title('Support and Resistance Levels')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例使用
plot_support_resistance(df, next_support, next_resistance)



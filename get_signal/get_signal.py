import akshare as ak
import pandas as pd
import numpy as np
import time

def get_realtime_signal(stock_code):
    """
    获取实时行情并结合历史数据计算信号
    stock_code: 股票代码, 如 "601899"
    """
    try:
        # 1. 获取历史日线数据 (取最近 100 天，确保指标计算准确)
        df_hist = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq").tail(100)
        df_hist.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'change', 'turnover']
        
        # 2. 获取当前的实时快照
        df_spot = ak.stock_zh_a_spot_em()
        row_spot = df_spot[df_spot['代码'] == stock_code].iloc[0]
        
        current_price = row_spot['最新价']
        current_high = row_spot['最高']
        current_low = row_spot['最低']
        current_date = pd.to_datetime('today').strftime('%Y-%m-%d')

        # 3. 将实时价格伪装成“今日收盘价”并追加到历史数据末尾
        new_row = pd.DataFrame({
            'date': [current_date],
            'open': [row_spot['今开']],
            'close': [current_price],
            'high': [current_high],
            'low': [current_low],
            'volume': [row_spot['成交量']],
        })
        # 注意：如果今天是交易日，历史数据里可能已经包含今日开盘数据，需去重
        if df_hist.iloc[-1]['date'] == current_date:
            df_hist = df_hist.iloc[:-1]
            
        df = pd.concat([df_hist, new_row], ignore_index=True)

        # 4. 计算技术指标
        # MA
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(20).mean()

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp1 - exp2
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = (df['DIF'] - df['DEA']) * 2

        # RSI
        def rsi(series, period=6):
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=period - 1, adjust=False).mean()
            ema_down = down.ewm(com=period - 1, adjust=False).mean()
            rs = ema_up / ema_down
            return 100 - (100 / (1 + rs))
        
        df['RSI6'] = rsi(df['close'], 6)
        df['RSI12'] = rsi(df['close'], 12)

        # KDJ
        low_9 = df['low'].rolling(9).min()
        high_9 = df['high'].rolling(9).max()
        rsv = (df['close'] - low_9) / (high_9 - low_9) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()

        # BOLL
        df['BOLL_MID'] = df['close'].rolling(20).mean()
        df['BOLL_STD'] = df['close'].rolling(20).std()

        # 5. 获取最新一秒的计算结果
        last = df.iloc[-1]
        
        # 买入判定
        b1 = 1 if (last['MA5'] > last['MA10'] > last['MA20'] and last['close'] > last['MA20']) else 0
        b2 = 1 if (last['DIF'] > last['DEA'] and last['MACD_hist'] > 0) else 0
        b3 = 1 if (last['RSI6'] > 50 and last['RSI6'] > last['RSI12']) else 0
        b4 = 1 if (last['K'] > last['D'] and last['K'] < 80) else 0
        b5 = 1 if (last['close'] > last['BOLL_MID']) else 0
        buy_score = b1 + b2 + b3 + b4 + b5

        # 卖出判定
        s1 = 1 if (last['MA5'] < last['MA10'] or last['close'] < last['MA5']) else 0
        s2 = 1 if (last['DIF'] < last['DEA'] or last['MACD_hist'] < 0) else 0
        s3 = 1 if (last['RSI6'] < 50 or last['RSI6'] < last['RSI12']) else 0
        s4 = 1 if (last['K'] < last['D'] or last['K'] > 80) else 0
        s5 = 1 if (last['close'] < last['BOLL_MID']) else 0
        sell_score = s1 + s2 + s3 + s4 + s5

        return {
            "name": row_spot['名称'],
            "price": current_price,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "time": time.strftime("%H:%M:%S")
        }

    except Exception as e:
        return f"计算出错: {e}"

# 实时轮询监控示例
target_stocks = ["601899", "603993"]
print("开始实时监控 (按 Ctrl+C 停止)...")
while True:
    for code in target_stocks:
        res = get_realtime_signal(code)
        print(f"[{res['time']}] {res['name']}({code}) 现价:{res['price']} | 买入分:{res['buy_score']} | 卖出分:{res['sell_score']}")
    print("-" * 50)
    time.sleep(5)  # 每 5 秒刷新一次
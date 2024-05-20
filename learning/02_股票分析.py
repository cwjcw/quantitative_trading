import tushare as ts
import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
import json

df = pd.read_csv(r'./learning/000001.csv')

# 把日期从YYYYMMDD格式转为date格式
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.set_index('trade_date')
# print(df)

# ---------------------------找出收盘价比开盘价上涨3%的日期----------------------------------

# 计算 close 比 open 高出 3% 的条件
condition = df['close'] > df['open'] * 1.03

# 筛选出符合条件的行
filtered_df = df[condition]

# 获取对应的 trade_date 仅保留日期部分并存入列表
result_dates = filtered_df.index.date

# 将日期格式化为字符串 YYYY-MM-DD
result_dates_str = [date.strftime('%Y-%m-%d') for date in result_dates]

print(result_dates_str)

# ---------------------------每天买入1手，最后一天全部卖出，收益如何----------------------------------

# 获取每个 trade_date 的 open 价格
buy_prices = df['open']

# 获取最后一天的 close 价格
final_close_price = df['close'].iloc[-1]

# 计算总买入成本
total_buy_cost = buy_prices.sum()

# 计算总卖出收益
total_sell_revenue = final_close_price * len(buy_prices)

# 计算净收益
net_profit = total_sell_revenue - total_buy_cost

print("总买入成本:", total_buy_cost)
print("总卖出收益:", total_sell_revenue)
print("净收益:", net_profit)




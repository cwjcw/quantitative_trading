import tushare as ts
import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
import json

# 把tushare的token存在本地，需要的时候调用即可
with open(r'D:\Document\研究\量化交易\tushare.json') as f:
    token_data = json.load(f)
token = token_data['token']

# 初始化pro接口
pro = ts.pro_api(token)

# 获取日线行情数据
source_data = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
print(source_data)

# 把日期从YYYYMMDD格式转为date格式
source_data['trade_date'] = pd.to_datetime(source_data['trade_date'], format='%Y%m%d')

# 设置trade_date为索引
source_data.set_index('trade_date', inplace=True)

# 输出转换后的数据
print(source_data)

# 保存DataFrame到CSV文件
source_data.to_csv(r'./learning/000001.csv')

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
source_data.to_csv('./learning/000001.csv', index=False)
print(source_data)
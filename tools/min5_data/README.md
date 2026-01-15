# 5 分钟行情数据脚本

该目录提供两个脚本，用于把 5 分钟行情写入同一个 SQLite 数据库，后续分析可直接从本地读取。

## 数据库位置
- `data/min5_data.db`
- 表名: `min5_bars`
- 主键: `(symbol, datetime)`

## 初始化（过去 20 个交易日）
```powershell
python tools\min5_data\init_20d_min5.py
```

## 今日更新（仅今天 5 分钟行情）
```powershell
python tools\min5_data\update_today_min5.py
```

## 可调参数
在脚本内可调整：
- `MAX_WORKERS`: 并发线程数
- `MAX_CODES`: 限制股票数量（用于测试或加速）

## 说明
- 股票代码列表会缓存到 `data/stock_codes.txt`，减少重复拉取。
- 数据源为 AkShare: `stock_zh_a_hist_min_em`（5 分钟）。

# 量化交易监控系统

本模块基于 AkShare + SQLite + Streamlit，实现 10 只股票的历史数据、实时快照、策略指标、回测与可视化。

## 目录结构
- `monitoring_system/db.py`：数据库初始化与写入
- `monitoring_system/fetch_history.py`：历史日线与分钟线采集
- `monitoring_system/realtime_collector.py`：实时采集入库（快照 + 5 分钟线）
- `monitoring_system/indicators.py`：布林带 / RSI / VWAP
- `monitoring_system/backtest.py`：组合回测 + 单股全仓回测（一键）
- `monitoring_system/run_backtest_portfolio.py`：仅组合回测
- `monitoring_system/run_backtest_single_full.py`：仅单股全仓回测
- `monitoring_system/app.py`：Streamlit 展示
- `config.json`：配置文件

## 数据库
- 数据文件：`data/monitoring.db`
- 表：`daily_bars`、`minute_bars`、`spot_snapshot`、`bid_ask`、`ticks`、`fund_flow`

## 配置文件
`config.json` 内包含：
- `symbols`：股票代码列表（6 位）
- `market_map`：市场映射（`sh/sz/bj`）
- `initial_cash`：初始资金
- `commission_rate`：佣金率
- `stamp_duty_rate`：印花税率
- `stamp_duty_on_sell`：是否仅卖出时收取印花税

## 使用步骤
1) 拉取历史数据（近 1 年日线 + 5 分钟线）
```powershell
python monitoring_system\fetch_history.py
```

2) 实时采集入库（快照每分钟，5 分钟线每 5 分钟）
```powershell
python monitoring_system\realtime_collector.py
```

3) 启动 Web 界面
```powershell
streamlit run monitoring_system\app.py
```

4) 回测（两种方式）
- 一键启动（组合 + 单股全仓）
```powershell
python monitoring_system\backtest.py
```
- 拆分启动（组合回测）
```powershell
python monitoring_system\run_backtest_portfolio.py
```
- 拆分启动（单股全仓回测）
```powershell
python monitoring_system\run_backtest_single_full.py
```

## 说明
- 回测遵循 T+1，不允许当日买入当日卖出。
- 最后一个交易日不再买入，如有持仓按收盘价强制平仓。
- 指数支持：上证、深证、创业板（由 `config.json` 配置）。

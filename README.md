# 量化交易研究仓库

本仓库聚焦 A 股量化数据采集、实时监控、回测与策略实验，核心数据源为 AkShare，存储使用 SQLite，展示使用 Streamlit。

## 目录结构
- `monitoring_system/`：监控系统（历史/实时采集、回测、Web 展示）
- `strategies/`：策略实验与脚本（按策略独立目录管理）
- `tools/`：通用工具脚本（如分钟线初始化与数据库工具）
- `data/`：本地数据库与回测输出（不纳入 Git）
- `fetch_board_info.py`：行业/概念板块基础信息抓取脚本
- `run_realtime.ps1`：一键启动实时采集 + Web

## 监控系统（monitoring_system）
功能概览：
- 历史数据：日线 + 5 分钟线入库（SQLite）
- 实时数据：快照 + 资金流 + 5 分钟线增量入库
- 回测：组合回测 + 单股全仓回测
- 展示：Streamlit Web

常用命令：
```powershell
# 历史数据采集（增量）
python monitoring_system\fetch_history.py

# 实时采集（快照每分钟，5 分钟线每 5 分钟）
python monitoring_system\realtime_collector.py

# Web 界面
streamlit run monitoring_system\app.py

# 回测一键（组合 + 单股全仓）
python monitoring_system\backtest.py

# 仅组合回测
python monitoring_system\run_backtest_portfolio.py

# 仅单股全仓回测
python monitoring_system\run_backtest_single_full.py

# 一键启动实时采集 + Web
.\run_realtime.ps1
```

输出文件（`data/`）：
- `monitoring.db`：SQLite 数据库
- `backtest_trades.csv`：交易明细
- `backtest_pnl_by_symbol.csv`：组合回测汇总
- `backtest_metrics_summary.csv`：组合绩效指标
- `backtest_single_full_summary.csv`：单股全仓回测汇总

## 市场模式
`config.json` 支持三种模式：`牛市 / 熊市 / 震荡`，不同模式会覆盖策略参数：
- `market_mode`：当前模式
- `market_mode_params`：各模式的参数集合

Web 页面支持直接切换市场模式（仅对当前页面生效）。

## 工具与策略
- `tools/min5_data/`：5 分钟行情初始化与更新脚本（独立数据库）
- `strategies/`：策略实验目录，按子目录维护各自的 README 与脚本

## 依赖与环境
建议使用虚拟环境：
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

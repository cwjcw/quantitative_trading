# 项目功能说明（quantitative_trading）

本仓库面向 A 股量化数据采集、实时监控、回测与策略实验。核心数据源为 AkShare，存储为 SQLite，展示与交互使用 Streamlit。

## 1. 整体架构与数据流
1) 历史数据采集：按配置抓取日线与分钟线，写入 `data/monitoring.db`。
2) 实时采集：按分钟更新快照/盘口/逐笔/资金流/5 分钟线，写入数据库。
3) 策略回测：基于数据库的分钟线与日线进行组合回测与单股全仓回测。
4) 监控展示：Streamlit 页面聚合快照、盘口、信号与历史数据图表。

## 2. 目录与模块功能

### 根目录
- `README.md`：仓库概览、常用命令与输出说明。
- `config.json`：策略参数、标的列表、市场模式、回测与历史参数配置。
- `fetch_board_info.py`：抓取概念/行业板块基础信息并生成 JSON。
- `run_realtime.ps1`：一键启动实时采集 + Streamlit 页面。
- `requirements.txt`：依赖列表。
- `data/`：本地数据库与回测输出（默认不纳入 Git）。

### monitoring_system/（核心监控系统）
- `db.py`：SQLite 初始化、表结构与中文视图、通用 upsert。
- `fetch_history.py`：历史日线 + 分钟线采集与增量写入。
- `realtime_collector.py`：实时采集快照/盘口/逐笔/资金流/分钟线入库。
- `indicators.py`：技术指标（布林带 / RSI / VWAP）。
- `backtest.py`：组合回测 + 单股全仓回测；输出交易明细和绩效指标。
- `run_backtest_portfolio.py`：仅运行组合回测。
- `run_backtest_single_full.py`：仅运行单股全仓回测。
- `app.py`：Streamlit 监控页面。
- `utils.py`：配置读取、股票代码标准化、标的清单。

### tools/min5_data/（独立 5 分钟数据工具）
- `db_utils.py`：独立 SQLite（`data/min5_data.db`）与 min5 数据写入工具。
- `init_20d_min5.py`：初始化近 20 个交易日 5 分钟数据。
- `update_today_min5.py`：更新当日 5 分钟数据。
- `README.md`：该工具的使用说明。

## 3. 配置文件（config.json）
常用字段说明：
- `symbols`：回测/监控标的代码列表（6 位）。
- `market_map`：标的市场映射（`sh/sz/bj`），用于资金流接口。
- `initial_cash`：初始资金。
- `commission_rate`：佣金率。
- `stamp_duty_rate`：印花税率。
- `stamp_duty_on_sell`：是否仅卖出收取印花税。
- `priority_map`：买入优先级（数值越小越优先）。
- `positions`：持仓列表（用于监控页面启用卖出信号判断）。
- `index_symbols`：指数标的（回测基准）。
- `history_params`：历史分钟线时间范围和周期。
- `backtest_params`：回测窗口开关（如是否使用分钟线范围）。
- `strategy_params`：策略参数（基础参数）。
- `market_mode`：当前市场模式（`bull/bear/range`）。
- `market_mode_params`：不同市场模式对策略参数的覆盖。

说明：`market_mode_params` 会覆盖 `strategy_params` 中同名参数。

## 4. 数据库结构

### 4.1 监控数据库（data/monitoring.db）
表结构（核心字段）：
- `daily_bars`：日线 OHLCV（`symbol`, `date`, `open`, `close`, `high`, `low`, `volume`, `amount`, `pct_chg`, `change`, `turnover`）。
- `minute_bars`：分钟线 OHLCV（`symbol`, `datetime`, `period`, `open`, `close`, `high`, `low`, `volume`, `amount` 等）。
- `spot_snapshot`：实时快照（最新价、涨跌幅、成交量/额、市值等）。
- `bid_ask`：盘口五档。
- `ticks`：逐笔成交。
- `fund_flow`：个股资金流。

同时创建了中文视图（如 `v_daily_bars_cn`、`v_minute_bars_cn`），便于直接查询。

### 4.2 分钟线工具库（data/min5_data.db）
- `min5_bars`：5 分钟 OHLCV（`symbol`, `datetime` 为联合主键）。

## 5. 历史数据采集（monitoring_system/fetch_history.py）
- 日线：默认抓取近 365 天（日线数据，增量更新）。
- 分钟线：按 `history_params` 指定的时间范围与周期（默认 5 分钟）。
- 数据增量：
  - 日线按 `daily_bars` 中最新日期继续往后拉取。
  - 分钟线按已存在的日期集合跳过已抓取交易日。
- 同时支持指数（日线 + 分钟）抓取，用于回测对标。

## 6. 实时采集（monitoring_system/realtime_collector.py）
- **快照**：每分钟抓取 A 股实时行情（`stock_zh_a_spot_em`）。
- **指数快照**：抓取指数实时行情（`stock_zh_index_spot_em`）。
- **盘口五档**：逐标的拉取（`stock_bid_ask_em`）。
- **逐笔成交**：`stock_intraday_em`。
- **资金流**：`stock_individual_fund_flow`，按 `FUND_FLOW_INTERVAL_MIN` 更新。
- **分钟线**：按 `MINUTE_FETCH_INTERVAL_MIN` 拉取 5 分钟线增量。
- 数据全部写入 `monitoring.db` 并可被 Streamlit 页面实时展示。

## 7. 指标与信号逻辑
- 指标：布林带（中轨/上轨/下轨）、RSI(14)、VWAP、成交量均线（20）。
- 买入信号（基础）：
  - 价格接近布林下轨（`band_tol`）。
  - RSI 超卖（`buy_rsi_threshold`）。
  - VWAP 连续确认上穿（`vwap_cross_k`）。
- 牛市模式增强（`market_mode = bull`）：
  - 价格站上中轨与 VWAP（可配置）。
  - 价格抬升与趋势 RSI（`trend_rsi_threshold`）。
- 卖出信号：
  - 可启用条件：触及上轨、RSI 超买、VWAP 偏离、放量。
  - 要求满足 `sell_require_min` 个条件，并可设置必须条件（`sell_must_*`）。
- 时间过滤：可配置开盘/收盘时间段过滤。

## 8. 回测逻辑（monitoring_system/backtest.py）
- **T+1 规则**：当日买入不允许当日卖出。
- **买入**：符合信号后按仓位比例买入，尊重最大持仓比例与现金预留。
- **加仓**：仅在牛市模式 + 趋势条件满足时允许。
- **卖出**：
  - 触发卖出信号 OR 止盈 OR 止损。
  - 限制涨跌停：遇到跌停价不卖。
- **最后一天强平**：回测结束日按收盘价强制平仓。
- **回测输出**：
  - `backtest_trades.csv`：交易明细（含 `reason` 卖出原因）。
  - `backtest_pnl_by_symbol.csv`：按标的汇总收益。
  - `backtest_metrics_summary.csv`：胜率/盈亏比/总收益/最大回撤。
  - `backtest_single_full_summary.csv`：单股全仓回测汇总。

## 9. Streamlit 监控页面（monitoring_system/app.py）
- 实时快照 + 盘口信息表。
- 计算并展示买入/卖出信号与未触发原因。
- 支持选择市场模式（仅对当前页面生效，不写回配置）。
- 日线历史数据表展示。
- 指定日期分钟线价格折线图。

## 10. 板块信息抓取（fetch_board_info.py）
- 抓取概念板块与行业板块列表。
- 提取板块简介，输出：
  - `data/concept_boards.json`
  - `data/industry_boards.json`

## 11. 常用命令
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

## 12. 输出文件说明（data/）
- `monitoring.db`：监控数据库。
- `backtest_trades.csv`：交易明细（含 `reason`）。
- `backtest_pnl_by_symbol.csv`：组合回测收益汇总。
- `backtest_metrics_summary.csv`：回测绩效指标汇总。
- `backtest_single_full_summary.csv`：单股全仓回测汇总。
- `min5_data.db`：5 分钟独立数据库。
- `concept_boards.json` / `industry_boards.json`：板块信息。

## 13. 重要行为与约束
- 交易按 100 股一手计。
- 涨跌停限制：主板 10%，创业板/科创板等 20%。
- 最后交易日不再开新仓，持仓强制平仓。
- 回测依赖历史分钟线与日线数据完整性。

## 14. 可能的扩展点
- 新增策略目录（`strategies/`）作为独立策略实验空间。
- 增加更多指标/信号源（如均线、MACD、资金流因子）。
- 将回测结果与基准指数收益做更系统的对比分析。

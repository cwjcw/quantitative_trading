# 数据库结构概览

本文是当前量化研究数据库的总览。逐表字段说明见 [database_objects/README.md](database_objects/README.md)。

## 总体分层

当前数据库主要分为三层：

| 层级 | Schema / 表 | 主要用途 |
| --- | --- | --- |
| QMT 行情层 | `public.stock_5m_bars`、`public.stock_snapshots`、`public.stock_instruments` | 本地可交易股票池、5 分钟 K 线、实时快照和合约信息。 |
| Tushare 原始层 | `public.tushare_raw_records`、`public.tushare_raw_runs`、`public.tushare_collection_checkpoints` | 保存 Tushare 原始 JSON、采集运行记录和逐股任务断点。 |
| 分析视图层 | `analytics.*` | 把原始 JSON 映射为可直接研究和选股的标准字段视图。 |

TimescaleDB 自带的 `_timescaledb_*` 和 `timescaledb_information` 对象属于扩展内部结构，不作为业务数据入口。

## 核心数据流

```text
QMT 本地行情
  -> public.stock_5m_bars / stock_snapshots / stock_instruments

Tushare 接口
  -> public.tushare_raw_records
  -> analytics.tushare_* 视图
  -> analytics.tushare_stock_feature_daily / tushare_financial_quality_latest
```

`tushare_raw_records` 是 Tushare 的统一原始层。当前采集策略是“按切片替换”，即同一个接口、日期/股票/口径的旧切片会先删除，再写入本次接口返回的最新版数据，不保留旧版本。

## 主要业务表

| 表 | 内容 | 使用建议 |
| --- | --- | --- |
| `public.stock_instruments` | 本地股票合约和基础信息 | 本地可交易股票池来源。 |
| `public.stock_5m_bars` | 标准 5 分钟 OHLCV | 日内特征、回测、尾盘行为、分钟级风控。 |
| `public.stock_snapshots` | 实时快照 | 盘中状态、盘口/实时价格观察。 |
| `public.tushare_raw_records` | Tushare 所有接口原始 JSON | 新接口先落这里，再映射分析视图。 |
| `public.tushare_raw_runs` | 每次 Tushare 抓取运行记录 | 查任务是否成功、耗时、错误数和抓取区间。 |
| `public.tushare_collection_checkpoints` | 逐股财报等长任务 checkpoint | 财报全市场任务断点续跑、跳过已完成接口。 |

## 主要分析视图

| 视图 | 内容 | 用途 |
| --- | --- | --- |
| `analytics.tushare_daily_price` | Tushare 日线开高低收、涨跌幅、成交量/额 | 日线收益、缺口、趋势、波动和日级回测。 |
| `analytics.tushare_daily_basic` | PE/PB、市值、换手率、量比等 | 估值、流动性、市值风格。 |
| `analytics.tushare_moneyflow_stock` | 个股小/中/大/特大单和主力净流入 | 资金流因子。 |
| `analytics.tushare_cyq_perf` | 筹码成本、获利盘比例 | 筹码结构和压力/支撑判断。 |
| `analytics.tushare_sector_moneyflow_dc` | 东方财富板块/概念资金流 | 板块轮动、强势板块筛选。 |
| `analytics.tushare_market_sentiment_daily` | 涨跌停、龙虎榜、公告、研报等市场统计 | 市场情绪和择时过滤。 |
| `analytics.tushare_stock_events` | ST、停牌、解禁、质押、减持、回购、公告、研报 | 风险事件过滤。 |
| `analytics.tushare_financial_quality_latest` | 最新财务质量宽表 | 基本面质量过滤。 |
| `analytics.tushare_stock_feature_daily` | 日线价格 + 估值 + 资金流 + 筹码宽表 | 个股日频选股模型首选入口。 |

## 选股模型推荐读取入口

日频选股优先从：

```sql
SELECT *
FROM analytics.tushare_stock_feature_daily
WHERE trade_date >= current_date - interval '30 days';
```

基本面质量过滤优先从：

```sql
SELECT *
FROM analytics.tushare_financial_quality_latest;
```

事件风险过滤优先从：

```sql
SELECT *
FROM analytics.tushare_stock_events
WHERE event_date >= current_date - interval '30 days';
```

本地可交易池和分钟级行情优先从：

```sql
SELECT *
FROM public.stock_instruments;

SELECT *
FROM public.stock_5m_bars
WHERE bar_time >= now() - interval '30 days';
```

## Tushare 抓取安排

统一入口是 `main_get_info`。n8n 每天只需要调用同一个脚本：

```bash
uv run main_get_info --lookback-days 7
```

脚本默认 `financial-mode=auto`：

| 日期 | 行为 |
| --- | --- |
| 周一到周六 | 抓日常数据，不跑逐股财报。 |
| 周日 | 抓日常数据，并对本地股票池全量跑财报接口。 |

日常数据包括：

- Tushare 日线：`daily`
- 日频估值/流动性：`daily_basic`
- 个股/行业/板块资金流
- 筹码分布
- 涨跌停、龙虎榜、融资融券、陆股通
- 停复牌、ST、解禁、质押、股东变动、回购、大宗交易
- 公告、研报、业绩预告、快报、分红、披露计划
- 股票基础资料、公司资料、指数/板块/成分

周日财报全量覆盖本地可交易股票池，当前约 `5208` 只，接口包括：

- `fina_indicator`
- `income`
- `balancesheet`
- `cashflow`
- `fina_mainbz`

## 文档目录

- 全局结构：本文
- 数据架构规划：[tushare_data_architecture.md](tushare_data_architecture.md)
- 回填报告：[tushare_backfill_report.md](tushare_backfill_report.md)
- 逐对象说明：[database_objects/README.md](database_objects/README.md)

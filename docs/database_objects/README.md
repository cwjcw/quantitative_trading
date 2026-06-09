# 数据库对象说明

本目录按对象拆分记录当前业务数据库里的主要表和分析视图。

说明：仅包含 `public` 和 `analytics` schema；TimescaleDB 内部 `_timescaledb_*` 与 `timescaledb_information` 对象未纳入。

| Schema | 对象 | 类型 | 说明 |
| --- | --- | --- | --- |
| `analytics` | [tushare_balance_sheet](analytics.tushare_balance_sheet.md) | `VIEW` | 资产负债表视图，包含货币资金、应收账款、存货、总资产、负债和股东权益等。 |
| `analytics` | [tushare_cashflow_statement](analytics.tushare_cashflow_statement.md) | `VIEW` | 现金流量表视图，包含经营、投资、筹资现金流和自由现金流等。 |
| `analytics` | [tushare_cyq_perf](analytics.tushare_cyq_perf.md) | `VIEW` | 筹码分布表现视图，包含成本分位、平均成本和获利盘比例。 |
| `analytics` | [tushare_daily_basic](analytics.tushare_daily_basic.md) | `VIEW` | Tushare 日频估值和流动性指标视图，包含 PE/PB、市值、换手率、量比等。 |
| `analytics` | [tushare_daily_price](analytics.tushare_daily_price.md) | `VIEW` | Tushare A 股日线价格视图，包含开高低收、昨收、涨跌额、涨跌幅、成交量和成交额。 |
| `analytics` | [tushare_fina_indicator](analytics.tushare_fina_indicator.md) | `VIEW` | 财务指标视图，包含 ROE、ROA、毛利率、净利率、偿债能力和运营效率等。 |
| `analytics` | [tushare_financial_quality_latest](analytics.tushare_financial_quality_latest.md) | `VIEW` | 最新一期财务质量宽表，合并财务指标、利润表、资产负债表和现金流量表。 |
| `analytics` | [tushare_income_statement](analytics.tushare_income_statement.md) | `VIEW` | 利润表视图，包含营业收入、营业成本、利润总额、归母净利润等。 |
| `analytics` | [tushare_main_business](analytics.tushare_main_business.md) | `VIEW` | 主营业务构成视图，包含业务项目、收入、成本和毛利等。 |
| `analytics` | [tushare_market_sentiment_daily](analytics.tushare_market_sentiment_daily.md) | `VIEW` | 市场情绪日频聚合视图，统计涨跌停、龙虎榜、大宗交易、公告和研报活跃度。 |
| `analytics` | [tushare_moneyflow_stock](analytics.tushare_moneyflow_stock.md) | `VIEW` | 个股资金流视图，包含小单、中单、大单、特大单和主力净流入等字段。 |
| `analytics` | [tushare_sector_moneyflow_dc](analytics.tushare_sector_moneyflow_dc.md) | `VIEW` | 东方财富行业/概念板块资金流视图，支持板块轮动和强势板块筛选。 |
| `analytics` | [tushare_stock_events](analytics.tushare_stock_events.md) | `VIEW` | 股票事件统一视图，整合 ST、停复牌、解禁、质押、股东变动、回购、公告、研报等事件。 |
| `analytics` | [tushare_stock_feature_daily](analytics.tushare_stock_feature_daily.md) | `VIEW` | 个股日频特征宽表，整合日线价格、估值、资金流和筹码指标，供选股模型直接使用。 |
| `public` | [stock_5m_bars](public.stock_5m_bars.md) | `BASE TABLE` | QMT 标准 5 分钟行情表，保存股票分钟级 OHLCV，是日内特征和回测的核心行情源。 |
| `public` | [stock_instruments](public.stock_instruments.md) | `BASE TABLE` | 本地股票合约/基础信息表，是本地可交易股票池的重要来源。 |
| `public` | [stock_snapshots](public.stock_snapshots.md) | `BASE TABLE` | QMT 实时快照表，保存采集时点的盘口/价格/成交等实时状态。 |
| `public` | [tushare_collection_checkpoints](public.tushare_collection_checkpoints.md) | `BASE TABLE` | 逐股或长任务采集 checkpoint 表，用于断点续跑、跳过已完成任务和记录失败原因。 |
| `public` | [tushare_raw_records](public.tushare_raw_records.md) | `BASE TABLE` | 新版 Tushare 原始数据统一表，保存接口原始 JSON；当前按切片替换，保留最新版数据。 |
| `public` | [tushare_raw_runs](public.tushare_raw_runs.md) | `BASE TABLE` | 新版 Tushare 通用采集运行记录，记录每次抓取的接口、区间、行数、耗时和错误数。 |

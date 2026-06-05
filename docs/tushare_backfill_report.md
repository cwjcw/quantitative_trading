# Tushare 数据回填报告

生成时间：2026-06-04

## 回填范围

- 时间范围：2025-06-03 至 2026-06-03
- 存储表：
  - `tushare_raw_runs`
  - `tushare_raw_records`
- 分析视图 schema：`analytics`
- 数据库连接：从 `.env` 读取 `DATABASE_URL` 或 `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD`

## 已完成阶段

| 阶段 | 说明 | 状态 |
| --- | --- | --- |
| `core_daily` | 每日指标、资金流、涨跌停、龙虎榜、融资融券、北向/港股通、筹码分布 | 完成 |
| `risk_events` | 停复牌、ST、股本、质押、股东人数/增减持、回购、大宗交易、公告、研报 | 完成 |
| `static_reference` | 股票基础、公司资料、名称变更、交易日历、指数/板块/成分 | 完成 |
| `fundamental_events` | 业绩预告、业绩快报、分红、披露计划、券商研报 | 完成 |
| `monthly_research` | 月度券商研报补跑 | 完成，主键去重后无新增 |

正式分阶段回填运行均为 `success`。

## 数据总量

- `tushare_raw_records`：15,088,874 行
- `tushare_raw_runs`：
  - `success`：72 次，15,088,874 行，0 个错误
  - `partial_success`：1 次，0 行，5 个错误

`partial_success` 来自早前探测需要 `ts_code` 参数的单股票财报接口，不属于本次正式分阶段回填。

## 主要接口行数

| endpoint | 行数 | 覆盖区间 |
| --- | ---: | --- |
| `dc_member` | 1,952,000 | 2025-06-03 至 2026-06-03 |
| `stk_limit` | 1,798,766 | 2025-06-03 至 2026-06-03 |
| `moneyflow_dc` | 1,449,947 | 2025-06-03 至 2026-06-03 |
| `cyq_perf` | 1,349,182 | 2025-06-03 至 2026-06-03 |
| `daily_basic` | 1,328,602 | 2025-06-03 至 2026-06-03 |
| `moneyflow` | 1,259,044 | 2025-06-03 至 2026-06-03 |
| `moneyflow_ths` | 1,244,953 | 2025-06-03 至 2026-06-03 |
| `index_weight` | 1,089,023 | 2025-06-03 至 2026-06-03 |
| `margin_detail` | 1,036,033 | 2025-06-03 至 2026-06-03 |
| `anns_d` | 607,527 | 2025-06-03 至 2026-06-03 |
| `share_float` | 394,798 | 2025-06-03 至 2026-06-03 |
| `hk_hold` | 213,135 | 2025-06-03 至 2026-06-03 |
| `top_inst` | 196,795 | 2025-06-03 至 2026-06-03 |
| `report_rc` | 184,885 | 2025-06-03 至 2026-06-03 |
| `dc_index` | 165,889 | 2025-06-03 至 2026-06-03 |
| `pledge_stat` | 157,760 | 2025-06-06 至 2026-05-29 |

## 分析视图

| 视图 | 行数 | 用途 |
| --- | ---: | --- |
| `analytics.tushare_daily_basic` | 1,328,602 | 日频估值、换手、市值、量比等 |
| `analytics.tushare_moneyflow_stock` | 1,259,044 | 个股主力/大单/小单资金流 |
| `analytics.tushare_cyq_perf` | 1,349,182 | 筹码成本、集中度、获利盘 |
| `analytics.tushare_market_sentiment_daily` | 356 | 市场级涨跌停、龙虎榜、融资融券、北向等情绪指标 |
| `analytics.tushare_sector_moneyflow_dc` | 165,889 | 东方财富板块/概念资金流 |
| `analytics.tushare_stock_events` | 1,333,197 | 公告、ST、停复牌、股本、质押、股东、回购、大宗等事件 |
| `analytics.tushare_stock_feature_daily` | 1,349,356 | 个股日频基础特征宽表，后续可与 QMT 行情 join |

## 初步分析框架

1. 股票级日频特征：
   - 从 `analytics.tushare_stock_feature_daily` 读取 Tushare 非交易行情特征。
   - 与 `stock_5m_bars` 聚合出的日内收益、波动率、成交量形态、尾盘特征按 `stock_code + trade_date` 合并。

2. 资金流强度：
   - 使用 `analytics.tushare_moneyflow_stock` 构造主力净流入、市值标准化净流入、大单/小单分歧。
   - 对比未来 1 日、3 日、5 日收益，先做分组收益和 IC。

3. 市场情绪过滤：
   - 使用 `analytics.tushare_market_sentiment_daily` 构造涨停强度、跌停压力、龙虎榜活跃度、融资融券变化、北向资金方向。
   - 作为择时过滤条件，不直接替代个股信号。

4. 板块轮动：
   - 使用 `analytics.tushare_sector_moneyflow_dc` 跟踪板块资金流排名、连续性、扩散强度。
   - 与个股所属指数/板块成分表结合，筛选强板块内的强个股。

5. 风险事件排除：
   - 使用 `analytics.tushare_stock_events` 过滤 ST、停牌、重大质押、减持、公告密集等事件风险。
   - 后续可拆成事件窗口，分析公告/分红/业绩预告前后的超额收益。

## 已知限制与下一步

- `fina_indicator`、`income`、`balancesheet`、`cashflow`、`fina_mainbz` 这类完整财报接口需要按 `ts_code` 逐只股票抓取，数据量和调用量都很大；当前代码已支持 `--all-codes --max-codes`，建议单独做分批任务。
- `anns_d`、`share_float`、`pledge_stat`、`dc_member`、`index_weight` 部分接口返回量接近接口上限，若要做严格全量，需要增加更细粒度分页或按股票/指数拆分补抓。
- 当前分析层先用 SQL 视图搭骨架，下一步可以增加因子计算脚本、回测样例和每日增量调度。

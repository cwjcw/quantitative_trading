# quantitative_trading

A 股量化交易研究仓库。当前项目以 PostgreSQL 为中心，把本地 QMT 行情数据和 Tushare 外生数据统一沉淀到数据库，再通过 SQL 视图和后续特征脚本服务研究、筛选、回测和策略实验。

本项目现在的核心思路是：

- QMT 负责交易行情数据，尤其是 5 分钟线、实时快照和股票合约信息。
- Tushare 负责非交易类、交易决策强相关的外生数据，例如资金流、每日指标、筹码分布、涨跌停、龙虎榜、融资融券、沪深股通、板块概念、公告研报、风险事件和基本面。
- 所有 Tushare 数据先进入通用原始层，完整保留 `raw jsonb`，再按研究需求映射到 `analytics` schema 下的分析视图。
- 采集任务必须可重跑、可去重、可补采、可追踪失败原因。

## 当前状态

截至 2026-06-05，已经完成：

- PostgreSQL 配置改为从 `.env` / 环境变量读取，不再在代码里硬编码数据库连接。
- 新增通用 Tushare 原始数据表：
  - `tushare_raw_runs`
  - `tushare_raw_records`
  - `tushare_collection_checkpoints`
- 新增 Tushare 通用采集器：
  - `qt-tushare-raw`
  - `scripts/tushare_backfill.py`
- 新增逐股财报采集器：
  - `qt-tushare-financials`
  - `scripts/tushare_financial_backfill.py`
- 新增分析视图：
  - `analytics.tushare_daily_basic`
  - `analytics.tushare_moneyflow_stock`
  - `analytics.tushare_cyq_perf`
  - `analytics.tushare_market_sentiment_daily`
  - `analytics.tushare_sector_moneyflow_dc`
  - `analytics.tushare_stock_events`
  - `analytics.tushare_stock_feature_daily`
  - `analytics.tushare_fina_indicator`
  - `analytics.tushare_income_statement`
  - `analytics.tushare_balance_sheet`
  - `analytics.tushare_cashflow_statement`
  - `analytics.tushare_main_business`
  - `analytics.tushare_financial_quality_latest`
- 已完成 Tushare 非财报类分阶段回填：
  - 时间范围：`2025-06-03` 至 `2026-06-03`
  - 原始记录：约 `15,088,874` 行
  - 正式分阶段回填：`72` 次成功，`0` 个错误
- 已完成第一批逐股财报回填：
  - 时间范围：`2025-01-01` 至 `2026-06-03`
  - 覆盖前 `500` 只股票
  - 财报接口：`fina_indicator`、`income`、`balancesheet`、`cashflow`、`fina_mainbz`
  - 已验证 checkpoint 断点续跑和去重逻辑

更详细的阶段报告见：

- [docs/tushare_data_architecture.md](docs/tushare_data_architecture.md)
- [docs/tushare_backfill_report.md](docs/tushare_backfill_report.md)

## 目录结构

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── tushare_data_architecture.md
│   └── tushare_backfill_report.md
├── scripts/
│   ├── tushare_backfill.py
│   └── tushare_financial_backfill.py
├── sql/
│   └── analysis_views.sql
└── src/
    └── quantitative_trading/
        ├── __init__.py
        ├── config.py
        ├── db.py
        ├── analysis/
        │   ├── __init__.py
        │   └── apply_views.py
        └── tushare/
            ├── __init__.py
            ├── collect_financials.py
            ├── collect_raw.py
            ├── endpoints.py
            └── schema.py
```

### 关键文件说明

| 文件 | 说明 |
| --- | --- |
| `src/quantitative_trading/config.py` | 读取 `.env` 和环境变量，生成运行配置 |
| `src/quantitative_trading/db.py` | 创建 SQLAlchemy engine |
| `src/quantitative_trading/tushare/schema.py` | 创建 Tushare 原始层表和 checkpoint 表 |
| `src/quantitative_trading/tushare/endpoints.py` | 定义 Tushare 接口、优先级、日期字段、代码字段和默认参数 |
| `src/quantitative_trading/tushare/collect_raw.py` | 通用 Tushare 原始数据采集器 |
| `src/quantitative_trading/tushare/collect_financials.py` | 逐股财报采集器，适合需要 `ts_code` 的接口 |
| `scripts/tushare_backfill.py` | 按阶段、按月分片回填非财报类数据 |
| `scripts/tushare_financial_backfill.py` | 按股票批次回填完整财报类数据 |
| `sql/analysis_views.sql` | 创建 `analytics` schema 下的分析视图 |
| `src/quantitative_trading/analysis/apply_views.py` | 应用 / 刷新分析视图 |

## 环境准备

### Python

项目要求 Python `>=3.11`。当前推荐使用 `uv` 管理依赖：

```bash
uv sync
```

安装后可以使用项目命令：

```bash
uv run qt-tushare-raw --help
uv run qt-tushare-financials --help
uv run qt-apply-analysis-views
```

### 依赖

依赖定义在 [pyproject.toml](pyproject.toml)：

```toml
dependencies = [
    "pandas>=2.2.0",
    "psycopg[binary]>=3.2.0",
    "sqlalchemy>=2.0.0",
    "tushare>=1.4.0",
]
```

当前已验证 `tushare==1.4.29` 可用。

### 环境变量

项目会自动读取仓库根目录的 `.env`。`.env` 已被 `.gitignore` 忽略，不能提交真实 token、数据库密码或证书。

最小配置：

```text
TUSHARE_TOKEN=your_tushare_token
```

推荐数据库配置：

```text
DATABASE_URL=postgresql+psycopg://user:password@host:port/database
```

也支持备用名称：

```text
SMART_STOCK_DATABASE_URL=postgresql+psycopg://user:password@host:port/database
```

或者使用拆分字段自动拼接：

```text
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=smart_stock
DB_USER=smart_stock
DB_PASSWORD=your_password
```

读取优先级：

1. `DATABASE_URL`
2. `SMART_STOCK_DATABASE_URL`
3. `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD`

## 数据库分层

### QMT 行情层

这些表由 QMT 或已有本地流程维护，Tushare 不重复采集行情 K 线：

| 表 | 说明 |
| --- | --- |
| `stock_5m_bars` | 标准 5 分钟线，后续策略和研究优先读取 |
| `stock_snapshots` | 实时快照 |
| `snapshot_runs` | 快照采集运行记录 |
| `stock_instruments` | 股票基础 / 合约信息 |
| `five_min_bars` | 旧表 / 遗留表，不建议新项目使用 |

建议读取 5 分钟线：

```sql
SELECT *
FROM stock_5m_bars
WHERE trade_date >= '2025-06-03'
ORDER BY stock_code, bar_time;
```

建议读取实时快照：

```sql
SELECT *
FROM stock_snapshots
ORDER BY captured_at DESC, stock_code;
```

### 已有 Tushare 资金流表

历史上已经存在：

| 表 | 说明 |
| --- | --- |
| `tushare_moneyflow_runs` | 资金流采集运行记录 |
| `tushare_moneyflow_records` | 资金流原始记录，原始字段在 `raw jsonb` |

这两张表可以继续保留，但新采集器统一写入 `tushare_raw_records`。

### 新增 Tushare 原始层

#### `tushare_raw_runs`

记录每次采集运行：

| 字段 | 说明 |
| --- | --- |
| `run_id` | UUID 主键 |
| `started_at` | 开始时间 |
| `finished_at` | 结束时间 |
| `status` | `running` / `success` / `partial_success` |
| `start_date` | 本次采集起始日期 |
| `end_date` | 本次采集结束日期 |
| `endpoints` | 本次选择的接口配置 JSON |
| `trade_dates` | 本次日期范围 JSON |
| `row_count` | 成功插入行数 |
| `error_count` | 错误数量 |
| `elapsed_seconds` | 耗时 |
| `note` | 错误摘要或备注 |

#### `tushare_raw_records`

统一保存所有 Tushare 原始数据：

| 字段 | 说明 |
| --- | --- |
| `endpoint` | Tushare 接口名，例如 `daily_basic`、`moneyflow` |
| `date_key` | 日期键，可能是交易日、公告日、报告期或月份 |
| `date_type` | 日期口径，例如 `trade_date`、`ann_date`、`month` |
| `ts_code` | 股票、指数、行业、概念等代码 |
| `content_type` | 同一接口的不同口径，例如 `stock_basic` 的 `L` / `D` / `P` |
| `row_hash` | 原始行 JSON 的稳定哈希 |
| `fetched_at` | 抓取时间 |
| `raw` | 原始 JSONB |

主键：

```text
endpoint, date_key, ts_code, content_type, row_hash
```

这个主键设计允许：

- 重复运行不重复写入。
- 同一 `endpoint + date_key + ts_code` 下如果 Tushare 返回了不同版本的行，可以通过 `row_hash` 保留差异。
- 静态数据、月度数据、日频数据、财报数据都能进入同一张原始表。

#### `tushare_collection_checkpoints`

逐股财报采集使用的断点续跑表：

| 字段 | 说明 |
| --- | --- |
| `checkpoint_key` | `endpoint + content_type + ts_code + 日期范围` |
| `endpoint` | 财报接口名 |
| `ts_code` | 股票代码 |
| `content_type` | 接口口径 |
| `range_start` | 采集起始日期 |
| `range_end` | 采集结束日期 |
| `status` | `success` / `failed` |
| `attempts` | 尝试次数 |
| `row_count` | 当前 checkpoint 插入行数 |
| `error_message` | 错误信息 |
| `updated_at` | 更新时间 |

## Tushare 接口优先级

接口配置集中在 [src/quantitative_trading/tushare/endpoints.py](src/quantitative_trading/tushare/endpoints.py)。

### P0：短线交易强相关

| 接口 | 说明 |
| --- | --- |
| `daily_basic` | 每日指标：估值、换手、市值、量比等 |
| `moneyflow` | Tushare 个股资金流 |
| `moneyflow_ths` | 同花顺个股资金流 |
| `moneyflow_dc` | 东方财富个股资金流 |
| `moneyflow_ind_ths` | 同花顺行业资金流 |
| `moneyflow_cnt_ths` | 同花顺概念资金流 |
| `moneyflow_ind_dc` | 东方财富行业资金流 |
| `moneyflow_mkt_dc` | 东方财富市场资金流 |
| `moneyflow_hsgt` | 沪深港通资金流 |
| `stk_limit` | 每日涨跌停价格 |
| `limit_list_d` | 涨跌停事件 |
| `limit_list_ths` | 同花顺涨跌停事件 |
| `top_list` | 龙虎榜个股 |
| `top_inst` | 龙虎榜机构席位 |
| `margin` | 融资融券汇总 |
| `margin_detail` | 融资融券明细 |
| `hk_hold` | 陆股通持股 |
| `hsgt_top10` | 沪深股通十大成交 |
| `ggt_top10` | 港股通十大成交 |
| `cyq_perf` | 筹码分布日频指标 |

### P1：事件和风险过滤

| 接口 | 说明 |
| --- | --- |
| `suspend_d` | 停复牌 |
| `stock_st` | ST 状态 |
| `share_float` | 限售解禁 |
| `pledge_stat` | 股权质押统计 |
| `pledge_detail` | 股权质押明细 |
| `stk_holdernumber` | 股东户数 |
| `stk_holdertrade` | 股东增减持 |
| `repurchase` | 回购 |
| `block_trade` | 大宗交易 |
| `stock_basic` | 股票基础资料 |
| `stock_company` | 公司资料 |
| `namechange` | 名称变更 |
| `anns_d` | 公告 |
| `report_rc` | 卖方研报 |

### P2：基本面和中长期筛选

| 接口 | 说明 |
| --- | --- |
| `fina_indicator` | 财务指标 |
| `income` | 利润表 |
| `balancesheet` | 资产负债表 |
| `cashflow` | 现金流量表 |
| `forecast` | 业绩预告 |
| `express` | 业绩快报 |
| `dividend` | 分红送股 |
| `fina_mainbz` | 主营业务构成 |
| `disclosure_date` | 财报披露计划 |
| `broker_recommend` | 券商月度推荐 |

### P3：市场环境和基准

| 接口 | 说明 |
| --- | --- |
| `index_dailybasic` | 指数每日指标 |
| `index_weight` | 指数权重 |
| `trade_cal` | 交易日历 |
| `ths_index` | 同花顺行业 / 概念 |
| `dc_index` | 东方财富板块 |
| `dc_member` | 东方财富板块成分 |
| `index_classify` | 申万行业分类 |
| `index_member_all` | 指数 / 行业成分 |

## 采集命令

### 创建表和采集单日 P0

```bash
uv run qt-tushare-raw \
  --start-date 20260603 \
  --end-date 20260603 \
  --priority P0
```

### 只采单个接口

```bash
uv run qt-tushare-raw \
  --start-date 20260603 \
  --end-date 20260603 \
  --endpoint daily_basic
```

### 采集多个优先级

```bash
uv run qt-tushare-raw \
  --start-date 20260603 \
  --end-date 20260603 \
  --priority P0 \
  --priority P1
```

### 分阶段回填

脚本 [scripts/tushare_backfill.py](scripts/tushare_backfill.py) 会按月份拆分，降低单次任务失败的影响。

可用阶段：

| stage | 内容 |
| --- | --- |
| `core_daily` | 每日指标、资金流、涨跌停、龙虎榜、融资融券、北向/港股通、筹码分布 |
| `risk_events` | 停复牌、ST、股本、质押、股东户数/增减持、回购、大宗交易、公告、研报 |
| `static_reference` | 股票基础、公司资料、名称变更、交易日历、指数/板块/成分 |
| `monthly_research` | 券商月度推荐 |
| `fundamental_events` | 业绩预告、业绩快报、分红、披露计划、券商研报 |

示例：

```bash
uv run python scripts/tushare_backfill.py \
  --stage core_daily \
  --start-date 20250603 \
  --end-date 20260603
```

建议顺序：

```bash
uv run python scripts/tushare_backfill.py --stage core_daily --start-date 20250603 --end-date 20260603
uv run python scripts/tushare_backfill.py --stage risk_events --start-date 20250603 --end-date 20260603
uv run python scripts/tushare_backfill.py --stage static_reference --start-date 20250603 --end-date 20260603
uv run python scripts/tushare_backfill.py --stage fundamental_events --start-date 20250603 --end-date 20260603
uv run python scripts/tushare_backfill.py --stage monthly_research --start-date 20250603 --end-date 20260603
```

### 逐股财报采集

单只股票：

```bash
uv run qt-tushare-financials \
  --start-date 20250101 \
  --end-date 20260603 \
  --ts-code 000001.SZ \
  --sleep-seconds 0.2 \
  --retry-failed
```

前 200 只股票：

```bash
uv run qt-tushare-financials \
  --start-date 20250101 \
  --end-date 20260603 \
  --all-codes \
  --max-codes 200 \
  --sleep-seconds 0.2 \
  --retry-failed
```

按批次跑全市场：

```bash
uv run python scripts/tushare_financial_backfill.py \
  --start-date 20250101 \
  --end-date 20260603 \
  --batch-size 300 \
  --offset 500 \
  --max-batches 1 \
  --sleep-seconds 0.2 \
  --batch-pause-seconds 5 \
  --retry-failed
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--batch-size` | 每批股票数量 |
| `--offset` | 跳过前 N 个股票代码，适合分段继续跑 |
| `--max-batches` | 本次最多跑几批 |
| `--sleep-seconds` | 每次 Tushare API 请求后的等待秒数 |
| `--batch-pause-seconds` | 批次之间等待秒数 |
| `--retry-failed` | 重试 checkpoint 里失败的任务 |
| `--include-delisted` | 当本地股票代码不可用时，Tushare `stock_basic` 同时包含退市和暂停上市 |

### 刷新分析视图

```bash
uv run qt-apply-analysis-views
```

这个命令会执行 [sql/analysis_views.sql](sql/analysis_views.sql)，创建或替换 `analytics` schema 下的视图。

## 分析视图

### 日频特征

#### `analytics.tushare_daily_basic`

从 `daily_basic` 拆出日频估值、换手、市值等字段。

常用字段：

- `ts_code`
- `trade_date`
- `turnover_rate`
- `turnover_rate_f`
- `volume_ratio`
- `pe`
- `pe_ttm`
- `pb`
- `ps_ttm`
- `dv_ttm`
- `total_share`
- `float_share`
- `free_share`
- `total_mv`
- `circ_mv`

#### `analytics.tushare_moneyflow_stock`

从 `moneyflow` 拆出个股资金流。

常用字段：

- `buy_sm_amount`
- `sell_sm_amount`
- `buy_md_amount`
- `sell_md_amount`
- `buy_lg_amount`
- `sell_lg_amount`
- `buy_elg_amount`
- `sell_elg_amount`
- `net_mf_amount`
- `net_mf_vol`

#### `analytics.tushare_cyq_perf`

从 `cyq_perf` 拆出筹码分布指标。

常用字段：

- `his_low`
- `his_high`
- `cost_5pct`
- `cost_15pct`
- `cost_50pct`
- `cost_85pct`
- `cost_95pct`
- `weight_avg`
- `winner_rate`

#### `analytics.tushare_stock_feature_daily`

将每日指标、资金流、筹码分布合并成股票日频特征宽表。

适合与 `stock_5m_bars` 聚合出的日内行情特征按 `ts_code + trade_date` 合并。

### 市场情绪

#### `analytics.tushare_market_sentiment_daily`

按日期聚合事件数量：

- 涨跌停事件数量
- 龙虎榜数量
- 机构龙虎榜数量
- 大宗交易数量
- 公告数量
- 研报数量

适合作为择时和风险偏好过滤器。

### 板块资金

#### `analytics.tushare_sector_moneyflow_dc`

从 `dc_index` 拆出东方财富板块 / 概念资金流。

常用字段：

- `sector_code`
- `sector_name`
- `trade_date`
- `leading_code`
- `leading_name`
- `pct_change`
- `leading_pct`
- `net_amount`
- `net_amount_rate`
- `total_mv`

### 事件风险

#### `analytics.tushare_stock_events`

统一事件视图，包含：

- ST
- 停复牌
- 股本 / 解禁
- 质押
- 股东人数
- 股东增减持
- 回购
- 公告
- 研报
- 名称变更

### 财报质量

#### `analytics.tushare_fina_indicator`

从 `fina_indicator` 拆出：

- EPS
- 每股经营现金流
- 毛利率
- 净利率
- ROE
- ROA
- ROIC
- 资产负债率
- 流动比率
- 速动比率
- 周转率

#### `analytics.tushare_income_statement`

从 `income` 拆出利润表核心字段：

- 营业总收入
- 营业收入
- 营业成本
- 销售费用
- 管理费用
- 财务费用
- 营业利润
- 利润总额
- 净利润
- 归母净利润

#### `analytics.tushare_balance_sheet`

从 `balancesheet` 拆出资产负债表核心字段：

- 货币资金
- 应收账款
- 存货
- 流动资产
- 总资产
- 流动负债
- 总负债
- 归母权益

#### `analytics.tushare_cashflow_statement`

从 `cashflow` 拆出现金流核心字段：

- 销售商品收到的现金
- 购买商品支付的现金
- 经营现金流净额
- 投资现金流净额
- 筹资现金流净额
- 自由现金流
- 期末现金及等价物

#### `analytics.tushare_main_business`

从 `fina_mainbz` 拆出主营业务构成：

- 主营项目
- 主营收入
- 主营利润
- 主营成本

#### `analytics.tushare_financial_quality_latest`

按股票取最新一期财务质量快照，便于做基本面过滤或股票池筛选。

## 常用 SQL

### 查看采集运行状态

```sql
SELECT
    status,
    count(*) AS runs,
    sum(row_count) AS rows,
    sum(error_count) AS errors
FROM tushare_raw_runs
GROUP BY status
ORDER BY status;
```

### 查看接口覆盖情况

```sql
SELECT
    endpoint,
    count(*) AS rows,
    count(DISTINCT ts_code) AS codes,
    min(date_key) AS min_date,
    max(date_key) AS max_date
FROM tushare_raw_records
GROUP BY endpoint
ORDER BY rows DESC;
```

### 查看财报 checkpoint

```sql
SELECT
    endpoint,
    status,
    count(*) AS checkpoints,
    sum(row_count) AS rows,
    max(updated_at) AS latest_update
FROM tushare_collection_checkpoints
GROUP BY endpoint, status
ORDER BY endpoint, status;
```

### 查某天个股资金流

```sql
SELECT *
FROM analytics.tushare_moneyflow_stock
WHERE trade_date = DATE '2026-06-03'
ORDER BY net_mf_amount DESC NULLS LAST
LIMIT 50;
```

### 查市场情绪

```sql
SELECT *
FROM analytics.tushare_market_sentiment_daily
ORDER BY trade_date DESC
LIMIT 30;
```

### 查板块资金流

```sql
SELECT *
FROM analytics.tushare_sector_moneyflow_dc
WHERE trade_date = DATE '2026-06-03'
ORDER BY net_amount DESC NULLS LAST
LIMIT 50;
```

### 查最新财务质量

```sql
SELECT
    ts_code,
    ann_date,
    end_date,
    roe,
    grossprofit_margin,
    netprofit_margin,
    debt_to_assets,
    n_cashflow_act,
    free_cashflow
FROM analytics.tushare_financial_quality_latest
ORDER BY roe DESC NULLS LAST
LIMIT 50;
```

### 与 QMT 5 分钟线聚合结果衔接

Tushare 股票代码是 `000001.SZ` 格式，QMT 表中可能是 `000001` + `exchange`。实际 join 前建议统一代码格式。

示例思路：

```sql
WITH qmt_daily AS (
    SELECT
        CASE
            WHEN exchange IN ('SH', 'SSE', 'SHSE') THEN stock_code || '.SH'
            WHEN exchange IN ('SZ', 'SZSE') THEN stock_code || '.SZ'
            WHEN exchange IN ('BJ', 'BSE') THEN stock_code || '.BJ'
            WHEN stock_code LIKE '6%' THEN stock_code || '.SH'
            WHEN stock_code LIKE '0%' OR stock_code LIKE '3%' THEN stock_code || '.SZ'
            WHEN stock_code LIKE '4%' OR stock_code LIKE '8%' THEN stock_code || '.BJ'
            ELSE stock_code
        END AS ts_code,
        trade_date,
        min(low_price) AS day_low,
        max(high_price) AS day_high,
        first(open_price ORDER BY bar_time) AS day_open,
        last(close_price ORDER BY bar_time) AS day_close,
        sum(volume) AS day_volume,
        sum(amount) AS day_amount
    FROM stock_5m_bars
    GROUP BY 1, 2
)
SELECT
    q.*,
    f.turnover_rate,
    f.volume_ratio,
    f.net_mf_amount,
    f.winner_rate,
    f.pe_ttm,
    f.pb
FROM qmt_daily q
LEFT JOIN analytics.tushare_stock_feature_daily f
    ON q.ts_code = f.ts_code
   AND q.trade_date = f.trade_date;
```

注意：上面的 `first()` / `last()` 需要数据库里有对应聚合函数，或者改用窗口函数实现。

## 初步研究框架

### 1. 日内行情特征

来源：`stock_5m_bars`

建议特征：

- 日收益
- 隔夜跳空
- 日内振幅
- 前 30 分钟收益
- 尾盘 30 分钟收益
- 上午 / 下午成交占比
- 成交量放大倍数
- VWAP 偏离
- 日内最大回撤
- 价格位置：收盘价在日内高低区间的位置

### 2. 日频外生特征

来源：`analytics.tushare_stock_feature_daily`

建议特征：

- 换手率
- 自由流通换手
- 量比
- PE / PB / 总市值 / 流通市值
- 主力净流入
- 主力净流入 / 流通市值
- 超大单净流入
- 大单与小单分歧
- 筹码获利比例
- 筹码成本中位数

### 3. 市场情绪

来源：`analytics.tushare_market_sentiment_daily`

建议特征：

- 涨停事件数量
- 龙虎榜活跃度
- 大宗交易活跃度
- 公告密度
- 研报密度
- 事件数量的 5 日 / 20 日均值
- 情绪强弱分位数

### 4. 板块轮动

来源：

- `analytics.tushare_sector_moneyflow_dc`
- `dc_member`
- `index_member_all`
- `index_classify`

建议特征：

- 板块涨跌幅排名
- 板块净流入排名
- 板块净流入连续性
- 龙头股涨幅
- 板块内部上涨扩散度
- 个股所属强势板块数量

### 5. 风险事件过滤

来源：`analytics.tushare_stock_events`

建议规则：

- 排除 ST 股票。
- 排除停牌 / 即将停牌股票。
- 对重大质押、减持、密集公告做风险标记。
- 对业绩预告、分红、披露计划做事件窗口分析，不一刀切排除。

### 6. 基本面质量过滤

来源：

- `analytics.tushare_fina_indicator`
- `analytics.tushare_income_statement`
- `analytics.tushare_balance_sheet`
- `analytics.tushare_cashflow_statement`
- `analytics.tushare_financial_quality_latest`

建议规则：

- ROE 大于行业中位数。
- 资产负债率处于合理区间。
- 经营现金流为正或改善。
- 营收 / 净利润同比改善。
- 毛利率、净利率稳定。
- 排除连续亏损或现金流恶化严重的公司。

## 建议工作流

### 每日增量

交易日收盘后：

```bash
TRADE_DATE=20260603

uv run qt-tushare-raw --start-date ${TRADE_DATE} --end-date ${TRADE_DATE} --priority P0
uv run python scripts/tushare_backfill.py --stage risk_events --start-date ${TRADE_DATE} --end-date ${TRADE_DATE}
uv run python scripts/tushare_backfill.py --stage fundamental_events --start-date ${TRADE_DATE} --end-date ${TRADE_DATE}
uv run qt-apply-analysis-views
```

### 每周补全

周末或低峰期：

```bash
uv run python scripts/tushare_backfill.py --stage static_reference --start-date 20260601 --end-date 20260607
uv run python scripts/tushare_financial_backfill.py --start-date 20250101 --end-date 20260607 --batch-size 300 --offset 500 --max-batches 2 --sleep-seconds 0.2 --retry-failed
uv run qt-apply-analysis-views
```

### 新机器初始化

```bash
uv sync
uv run qt-apply-analysis-views
uv run qt-tushare-raw --start-date 20260603 --end-date 20260603 --priority P0
```

## 数据质量检查

### 运行是否失败

```sql
SELECT *
FROM tushare_raw_runs
WHERE status <> 'success'
ORDER BY started_at DESC;
```

### 某个接口某天是否有数据

```sql
SELECT count(*)
FROM tushare_raw_records
WHERE endpoint = 'moneyflow'
  AND date_key = '20260603';
```

### 财报是否漏抓

```sql
SELECT *
FROM tushare_collection_checkpoints
WHERE status <> 'success'
ORDER BY updated_at DESC;
```

### 重复运行是否产生大量新增

重复跑同一日期后，如果 Tushare 返回内容不变，`inserted` 应接近 `0`。如果仍大量新增，说明接口数据版本变化、字段变化或主键策略需要复核。

## 已知限制

- Tushare 接口权限和积分会影响可获取数据范围。采集器会记录错误，不会伪造数据。
- `fina_indicator`、`income`、`balancesheet`、`cashflow`、`fina_mainbz` 必须按 `ts_code` 逐只股票抓取，全市场回填需要分批执行。
- 部分接口返回量接近上限，例如 `anns_d`、`share_float`、`pledge_stat`、`dc_member`、`index_weight`。如果要追求严格全量，需要继续按股票、指数、公告日期或分页条件拆分。
- `analytics.tushare_financial_quality_latest` 是普通视图，随着财报数据量变大，复杂 join 可能变慢。后续建议改成物化视图或标准化宽表。
- `tushare_raw_records` 是 JSONB 原始层，灵活但不适合所有高频查询。频繁使用的字段应下沉为标准表或物化视图。
- QMT 和 Tushare 股票代码格式不同，join 前必须统一为 `ts_code`。

## 后续路线

### 短期

- 继续分批补齐全市场完整财报。
- 把 `analytics.tushare_financial_quality_latest` 改为物化视图或增量表。
- 增加 QMT 5 分钟线日频聚合视图。
- 增加基础因子计算脚本。
- 增加每日增量采集脚本。

### 中期

- 建立标准化特征表，例如：
  - `features.stock_daily_qmt`
  - `features.stock_daily_tushare`
  - `features.stock_daily_merged`
  - `features.market_sentiment_daily`
  - `features.sector_rotation_daily`
- 增加 IC、分组收益、换手、覆盖率、缺失率分析。
- 增加简单回测样例。
- 增加数据质量报告。

### 长期

- 引入任务调度，例如 cron、systemd timer、Airflow 或 Dagster。
- 将常用 SQL view 升级为物化视图或标准宽表。
- 建立策略研究 notebook / report 模板。
- 增加模型训练数据集导出。
- 增加生产前风控检查。

## Git 和安全约定

- `.env`、`.venv`、日志、缓存和临时文件不提交。
- 不提交真实数据库密码、Tushare token、证书或私钥。
- 当前仓库存在一些历史文件删除状态，提交前需要确认是否要纳入本次提交。
- 建议一次提交只包含同一主题的代码、SQL、文档和锁文件。
- 提交前至少运行：

```bash
uv run python -m compileall src scripts
uv run qt-apply-analysis-views
```

## 常用开发命令

```bash
# 安装依赖
uv sync

# 编译检查
uv run python -m compileall src scripts

# 刷新分析视图
uv run qt-apply-analysis-views

# 查看 Git 状态
git status --short --branch

# 查看最近采集运行
uv run python - <<'PY'
from sqlalchemy import text
from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine

engine = make_engine(get_settings().database_url)
with engine.connect() as conn:
    for row in conn.execute(text("""
        SELECT started_at, finished_at, status, row_count, error_count, note
        FROM tushare_raw_runs
        ORDER BY started_at DESC
        LIMIT 10
    """)).mappings():
        print(dict(row))
PY
```

## Codex / Tushare Skill

本机 Codex 已安装 Tushare skill：

```text
/home/Jerry/.codex/skills/tushare
```

该 skill 用于把自然语言金融研究请求转成 Tushare 数据获取、清洗、对比、筛选和导出流程。当前只保留 `tushare` 一个 skill 目录，避免 `tushare` 与 `tushare-data` 内容重复。

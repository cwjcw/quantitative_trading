# Tushare 非交易数据架构规划

本项目的交易行情数据由 QMT 负责采集，Tushare 侧主要补齐和交易决策相关的外生数据：资金、情绪、板块、风险事件、基本面和市场环境。

## 目标

- 不重复采集 QMT 已覆盖的日线、分钟线和实时行情。
- Tushare 数据先进入原始层，保留完整 `raw jsonb`，避免接口字段变化影响采集。
- 后续按研究需要从原始层抽取标准化视图或宽表。
- 采集任务可重跑、可补采、可记录失败原因。

## 数据分层

### 1. QMT 行情层

已有核心表：

- `stock_5m_bars`：标准 5 分钟线，策略和分析优先读取。
- `stock_snapshots`：实时快照。
- `stock_instruments`：股票基础信息。

这一层不由 Tushare 覆盖。

### 2. Tushare 原始层

新增通用表：

- `tushare_raw_runs`：每次采集运行记录。
- `tushare_raw_records`：所有非交易类 Tushare 接口的原始记录。

通用主键：

```text
endpoint, date_key, ts_code, content_type, row_hash
```

其中：

- `endpoint`：Tushare 接口名。
- `date_key`：交易日、公告日、报告期或静态数据批次键。
- `ts_code`：股票、指数、行业、概念等代码；没有代码时为空字符串。
- `content_type`：用于区分同一接口的不同口径。
- `row_hash`：基于原始行 JSON 计算，保证重复运行不重复写入。
- `raw`：完整原始 JSON。

### 3. Tushare 标准层

后续按使用频率建立标准表或物化视图，例如：

- `factor_daily_basic`：日频估值、换手、市值、量比等。
- `event_limit_list`：涨跌停、连板、炸板、封单等情绪事件。
- `event_top_list`：龙虎榜个股与机构席位。
- `flow_margin`：融资融券。
- `flow_hsgt`：沪深股通持仓和成交。
- `sector_membership`：行业、概念与成分关系。
- `risk_events`：ST、停复牌、解禁、质押、减持、回购。
- `fundamental_reports`：财务指标、业绩预告、业绩快报。

### 4. 分析特征层

最终服务策略和研究：

- 情绪周期：涨停数量、连板高度、炸板率、跌停数量。
- 资金偏好：主力净流入、北向持仓变化、两融余额变化。
- 板块轮动：行业/概念强度、资金净流入、成分扩散度。
- 风险过滤：ST、停牌、解禁压力、质押比例、减持公告。
- 基本面过滤：ROE、利润增速、现金流质量、估值分位。

## 采集优先级

### P0：短线交易强相关

- `daily_basic`：每日指标，使用日频字段，不再采 Tushare 日线价格。
- `moneyflow`、`moneyflow_ths`、`moneyflow_dc`：个股资金流。
- `moneyflow_ind_ths`、`moneyflow_cnt_ths`、`moneyflow_ind_dc`、`moneyflow_mkt_dc`、`moneyflow_hsgt`：行业、概念、市场和北向资金流。
- `hsgt_top10`、`ggt_top10`：沪深股通/港股通十大成交。
- `cyq_perf`：筹码分布日频指标。
- `stk_limit`、`limit_list_d`、`limit_list_ths`：涨跌停与市场情绪。
- `top_list`、`top_inst`：龙虎榜。
- `margin`、`margin_detail`：融资融券。
- `hk_hold`：陆股通持股。

### P1：事件和风险过滤

- `suspend_d`：停复牌。
- `stock_st`：ST 状态。
- `share_float`：限售解禁。
- `pledge_stat`、`pledge_detail`：股权质押。
- `stk_holdernumber`：股东户数。
- `stk_holdertrade`：股东增减持。
- `repurchase`：回购。
- `block_trade`：大宗交易。
- `stock_basic`、`stock_company`、`namechange`：股票基础资料、公司资料和名称变更。
- `anns_d`：公告。
- `report_rc`：卖方研报。

### P2：基本面和中长期筛选

- `fina_indicator`：财务指标。
- `income`、`balancesheet`、`cashflow`：三大报表。
- `forecast`：业绩预告。
- `express`：业绩快报。
- `dividend`：分红送股。
- `fina_mainbz`：主营业务构成。
- `disclosure_date`：财报披露计划。
- `broker_recommend`：券商月度推荐。

### P3：市场环境和基准

- `index_dailybasic`：指数每日指标。
- `index_weight`：指数权重。
- `trade_cal`：交易日历。
- `ths_index`：同花顺行业/概念基础信息。
- `dc_index`、`dc_member`：东方财富板块和成分。
- `index_classify`、`index_member_all`：申万行业分类和成分。
- 宏观数据：CPI、PPI、PMI、社融、货币供应、利率。

## 建议目录结构

```text
src/quantitative_trading/
  config.py
  db.py
  tushare/
    endpoints.py
    schema.py
    collect_raw.py
docs/
  tushare_data_architecture.md
```

后续可以继续扩展：

```text
src/quantitative_trading/features/
  market_sentiment.py
  sector_rotation.py
  capital_flow.py
  risk_filters.py
sql/
  views/
  materialized_views/
```

## 实施路线

1. 先跑 P0，确认 token 权限、数据库写入和每日调度链路。
2. 再跑 P1，把策略风控和事件过滤补齐。
3. P2 使用公告日、报告期做增量，不需要每日全量重刷。
4. P3 用于大盘和风格状态判断，频率低于 P0。
5. 数据验证以 endpoint 行数、日期覆盖、空值比例和重复率为核心。

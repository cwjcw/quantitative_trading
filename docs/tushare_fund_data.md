# Tushare 基金数据说明

本文记录基金数据的采集口径、落库结构和当前回填状态。

## 数据来源

基金数据唯一来源为 Tushare。

当前已验证可用接口：

| 接口 | 内容 | 入库表 |
| --- | --- | --- |
| `fund_basic` | 公募基金基础信息，包含开放式基金和场内基金 | `public.tushare_fund_basic` |
| `fund_company` | 基金公司、资管和投资管理机构信息 | `public.tushare_fund_company` |
| `fund_manager` | 基金经理任职和履历 | `public.tushare_fund_manager` |
| `fund_nav` | 基金净值 | `public.tushare_fund_nav` |
| `fund_daily` | 场内基金日行情 | `public.tushare_fund_daily` |
| `fund_share` | 基金份额 | `public.tushare_fund_share` |
| `fund_portfolio` | 基金持仓 | `public.tushare_fund_portfolio` |
| `fund_div` | 基金分红 | `public.tushare_fund_div` |

已探测但当前不可用的私募产品级接口候选包括 `pri_fund_basic`、`private_fund_basic`、`fund_private`、`fund_pri`、`fund_basic_private`。Tushare 当前返回接口名不正确。因此本项目当前可落库的是公募基金产品数据；`fund_company` 中包含大量私募/投资管理机构，但不是私募产品净值。

## 数据分层

基金数据采用 raw + 窄表两层：

```text
Tushare 基金接口
  -> public.tushare_fund_records
  -> public.tushare_fund_basic / fund_nav / fund_daily / ...
```

`public.tushare_fund_records` 是基金接口统一原始层，保留完整 `raw jsonb`。标准窄表按分析场景拆分，供查询、筛选、回测和统计使用。

## 当前数据

截至 2026-06-17，已完成 metadata 入库：

| 表 | 行数 | 内容 |
| --- | ---: | --- |
| `public.tushare_fund_basic` | 31,562 | 基金基础信息 |
| `public.tushare_fund_company` | 14,114 | 基金公司和管理机构 |
| `public.tushare_fund_manager` | 81,830 | 基金经理任职记录 |

基金基础信息覆盖：

| 市场 | 基金数 | 说明 |
| --- | ---: | --- |
| `O` | 28,752 | 开放式基金 |
| `E` | 2,810 | 场内基金 |

`fund_nav` 全量回填已启动，范围为 `2000-01-01` 至 `2026-06-17`。该任务按基金代码逐只抓取并支持断点续跑。日志路径：

```bash
logs/tushare_fund_nav_2026-06-17.log
```

查看进度：

```bash
tail -f logs/tushare_fund_nav_2026-06-17.log
```

查看净值入库状态：

```sql
SELECT count(*) AS rows,
       count(DISTINCT ts_code) AS funds,
       min(nav_date) AS min_nav_date,
       max(nav_date) AS max_nav_date
FROM public.tushare_fund_nav;
```

## 回填脚本

脚本：

```bash
scripts/tushare_fund_backfill.py
```

初始化或刷新基金基础信息：

```bash
uv run python scripts/tushare_fund_backfill.py --metadata --sleep-seconds 0.02
```

回填基金净值：

```bash
uv run python scripts/tushare_fund_backfill.py \
  --nav \
  --start-date 20000101 \
  --end-date 20260617 \
  --sleep-seconds 0.02
```

后台执行示例：

```bash
setsid uv run python -u scripts/tushare_fund_backfill.py \
  --nav \
  --start-date 20000101 \
  --end-date 20260617 \
  --sleep-seconds 0.02 \
  > logs/tushare_fund_nav_2026-06-17.log 2>&1 < /dev/null &
```

脚本默认会跳过已经存在 `fund_nav` 记录的基金代码。需要重抓时使用 `--no-resume`。

## 与每日任务一起启动

`main_get_info` 已支持基金任务参数。推荐 N8N SSH command 使用：

```bash
cd /data/automation/code/personal/quantitative_trading && .venv/bin/python -u scripts/main_get_info.py --lookback-days 7 --fund-mode all --fund-background --fund-log-path logs/tushare_fund_$(date +%F).log >> logs/main_get_info_$(date +%F).log 2>&1
```

参数含义：

| 参数 | 说明 |
| --- | --- |
| `--fund-mode all` | 先刷新基金 metadata，再抓基金净值。 |
| `--fund-background` | 基金任务后台启动，避免 N8N 等待长时间净值回填。 |
| `--fund-log-path` | 指定基金任务日志文件。 |

默认情况下，每日入口不会传 `--fund-resume`，因此会按本次 `lookback-days` 对所有基金刷新最近净值。初始长历史回填或断点续跑时，建议直接调用底层脚本；底层脚本默认会跳过已经存在 `fund_nav` 记录的基金代码。

## 推荐查询

基金池：

```sql
SELECT ts_code, name, market, fund_type, management, found_date, status
FROM public.tushare_fund_basic
WHERE found_date >= DATE '2000-01-01';
```

基金净值序列：

```sql
SELECT n.ts_code, b.name, n.nav_date, n.unit_nav, n.accum_nav, n.adj_nav
FROM public.tushare_fund_nav n
JOIN public.tushare_fund_basic b USING (ts_code)
WHERE n.ts_code = '000001.OF'
ORDER BY n.nav_date;
```

基金经理任职：

```sql
SELECT m.ts_code, b.name AS fund_name, m.name AS manager_name,
       m.begin_date, m.end_date, m.resume
FROM public.tushare_fund_manager m
LEFT JOIN public.tushare_fund_basic b USING (ts_code)
WHERE m.ts_code = '000001.OF'
ORDER BY m.begin_date;
```

## 注意事项

- `fund_nav` 不能按日期直接全市场拉取，必须按 `ts_code` 逐只基金抓取。
- `fund_basic` 需要使用 `limit` / `offset` 分页，否则默认返回不完整。
- `fund_company` 是机构层数据，不等同于私募产品明细。
- 标准窄表用于分析查询；如需未映射字段，回到 `public.tushare_fund_records.raw` 查询。

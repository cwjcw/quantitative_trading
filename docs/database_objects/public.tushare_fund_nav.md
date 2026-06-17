# public.tushare_fund_nav

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：基金净值序列，来自 Tushare `fund_nav`。
- 主键：`ts_code, nav_date`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `NO` |
| `ann_date` | `date` | `YES` |
| `nav_date` | `date` | `NO` |
| `unit_nav` | `numeric` | `YES` |
| `accum_nav` | `numeric` | `YES` |
| `accum_div` | `numeric` | `YES` |
| `net_asset` | `numeric` | `YES` |
| `total_netasset` | `numeric` | `YES` |
| `adj_nav` | `numeric` | `YES` |
| `update_flag` | `text` | `YES` |
| `fetched_at` | `timestamp with time zone` | `NO` |

## 使用建议

基金收益、回撤、波动、排名等分析优先读取本表。`fund_nav` 需要按 `ts_code` 逐只基金抓取，当前全量回填任务正在运行。

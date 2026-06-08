# analytics.tushare_cyq_perf

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：筹码分布表现视图，包含成本分位、平均成本和获利盘比例。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `YES` |
| `trade_date` | `date` | `YES` |
| `his_low` | `numeric` | `YES` |
| `his_high` | `numeric` | `YES` |
| `cost_5pct` | `numeric` | `YES` |
| `cost_15pct` | `numeric` | `YES` |
| `cost_50pct` | `numeric` | `YES` |
| `cost_85pct` | `numeric` | `YES` |
| `cost_95pct` | `numeric` | `YES` |
| `weight_avg` | `numeric` | `YES` |
| `winner_rate` | `numeric` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

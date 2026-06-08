# public.tushare_moneyflow_runs

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：旧版 Tushare 资金流采集运行记录，保留作历史兼容。
- 估算行数：`-1`
- 存储大小：`16.0 KB`
- 主键：`run_id`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `run_id` | `uuid` | `NO` |
| `started_at` | `timestamp with time zone` | `NO` |
| `finished_at` | `timestamp with time zone` | `YES` |
| `status` | `text` | `NO` |
| `start_date` | `date` | `NO` |
| `end_date` | `date` | `NO` |
| `endpoints` | `ARRAY` | `NO` |
| `trade_dates` | `integer` | `NO` |
| `row_count` | `integer` | `NO` |
| `elapsed_seconds` | `double precision` | `YES` |
| `note` | `text` | `YES` |

## 使用建议

结合采集运行记录和字段含义使用；生产查询建议优先走业务视图或带过滤条件查询。

# public.tushare_raw_runs

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：新版 Tushare 通用采集运行记录，记录每次抓取的接口、区间、行数、耗时和错误数。
- 估算行数：`110`
- 存储大小：`256.0 KB`
- 主键：`run_id`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `run_id` | `uuid` | `NO` |
| `started_at` | `timestamp with time zone` | `NO` |
| `finished_at` | `timestamp with time zone` | `YES` |
| `status` | `text` | `NO` |
| `start_date` | `date` | `YES` |
| `end_date` | `date` | `YES` |
| `endpoints` | `jsonb` | `NO` |
| `trade_dates` | `jsonb` | `NO` |
| `row_count` | `integer` | `NO` |
| `error_count` | `integer` | `NO` |
| `elapsed_seconds` | `numeric` | `YES` |
| `note` | `text` | `YES` |

## 使用建议

结合采集运行记录和字段含义使用；生产查询建议优先走业务视图或带过滤条件查询。

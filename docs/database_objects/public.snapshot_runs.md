# public.snapshot_runs

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：实时快照采集运行记录，用于追踪每轮快照任务。
- 估算行数：`-1`
- 存储大小：`32.0 KB`
- 主键：`run_id`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `run_id` | `uuid` | `NO` |
| `requested_trade_date` | `date` | `YES` |
| `started_at` | `timestamp with time zone` | `NO` |
| `finished_at` | `timestamp with time zone` | `YES` |
| `status` | `text` | `NO` |
| `mode` | `text` | `NO` |
| `source` | `text` | `NO` |
| `stock_count` | `integer` | `NO` |
| `elapsed_seconds` | `double precision` | `YES` |
| `interval_minutes` | `integer` | `YES` |
| `note` | `text` | `YES` |
| `market_phase` | `text` | `YES` |

## 使用建议

这是本地 QMT 行情/合约数据层；与 Tushare 外生数据按股票代码和日期/时间拼接使用。

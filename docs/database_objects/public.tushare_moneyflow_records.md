# public.tushare_moneyflow_records

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：旧版 Tushare 资金流原始记录，后续统一优先使用 tushare_raw_records。
- 估算行数：`-1`
- 存储大小：`32.0 KB`
- 主键：`endpoint, trade_date, ts_code, content_type, row_hash`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `endpoint` | `text` | `NO` |
| `trade_date` | `date` | `NO` |
| `ts_code` | `text` | `NO` |
| `content_type` | `text` | `NO` |
| `row_hash` | `text` | `NO` |
| `fetched_at` | `timestamp with time zone` | `NO` |
| `raw` | `jsonb` | `NO` |

## 使用建议

结合采集运行记录和字段含义使用；生产查询建议优先走业务视图或带过滤条件查询。

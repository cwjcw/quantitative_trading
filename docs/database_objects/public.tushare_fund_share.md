# public.tushare_fund_share

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：基金份额数据，来自 Tushare `fund_share`。
- 主键：`ts_code, trade_date`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `NO` |
| `trade_date` | `date` | `NO` |
| `fd_share` | `numeric` | `YES` |
| `fund_type` | `text` | `YES` |
| `market` | `text` | `YES` |
| `fetched_at` | `timestamp with time zone` | `NO` |

## 使用建议

用于分析基金规模和份额变化，尤其是 ETF 份额扩张和收缩。

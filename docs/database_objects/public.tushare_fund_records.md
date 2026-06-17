# public.tushare_fund_records

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：Tushare 基金接口统一原始层，保存基金接口完整 `raw jsonb`。
- 主键：`endpoint, date_key, ts_code, entity_key, market, content_type, row_hash`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `endpoint` | `text` | `NO` |
| `date_key` | `text` | `NO` |
| `date_type` | `text` | `NO` |
| `ts_code` | `text` | `NO` |
| `entity_key` | `text` | `NO` |
| `market` | `text` | `NO` |
| `content_type` | `text` | `NO` |
| `row_hash` | `text` | `NO` |
| `fetched_at` | `timestamp with time zone` | `NO` |
| `raw` | `jsonb` | `NO` |

## 使用建议

这是基金数据的 raw 入口。分析优先读取 `public.tushare_fund_*` 标准窄表；如需未映射字段，再回到本表查询 `raw`。

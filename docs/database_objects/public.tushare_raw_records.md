# public.tushare_raw_records

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：新版 Tushare 原始数据统一表，保存接口原始 JSON；当前按切片替换，保留最新版数据。
- 估算行数：`15531953`
- 存储大小：`10.4 GB`
- 主键：`endpoint, date_key, ts_code, content_type, row_hash`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `endpoint` | `text` | `NO` |
| `date_key` | `text` | `NO` |
| `date_type` | `text` | `NO` |
| `ts_code` | `text` | `NO` |
| `content_type` | `text` | `NO` |
| `row_hash` | `text` | `NO` |
| `fetched_at` | `timestamp with time zone` | `NO` |
| `raw` | `jsonb` | `NO` |

## 使用建议

这是 Tushare 原始层统一入口；新增接口应先落到这里，再按研究需求映射到 `analytics` 视图。

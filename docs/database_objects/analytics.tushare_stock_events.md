# analytics.tushare_stock_events

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：股票事件统一视图，整合 ST、停复牌、解禁、质押、股东变动、回购、公告、研报等事件。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `endpoint` | `text` | `YES` |
| `ts_code` | `text` | `YES` |
| `event_date` | `date` | `YES` |
| `raw` | `jsonb` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

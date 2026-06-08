# analytics.tushare_main_business

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：主营业务构成视图，包含业务项目、收入、成本和毛利等。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `YES` |
| `end_date` | `date` | `YES` |
| `business_item` | `text` | `YES` |
| `bz_sales` | `numeric` | `YES` |
| `bz_profit` | `numeric` | `YES` |
| `bz_cost` | `numeric` | `YES` |
| `raw` | `jsonb` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

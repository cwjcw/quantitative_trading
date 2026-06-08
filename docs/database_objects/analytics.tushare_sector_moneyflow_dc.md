# analytics.tushare_sector_moneyflow_dc

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：东方财富行业/概念板块资金流视图，支持板块轮动和强势板块筛选。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `sector_code` | `text` | `YES` |
| `sector_name` | `text` | `YES` |
| `trade_date` | `date` | `YES` |
| `leading_code` | `text` | `YES` |
| `leading_name` | `text` | `YES` |
| `pct_change` | `numeric` | `YES` |
| `leading_pct` | `numeric` | `YES` |
| `net_amount` | `numeric` | `YES` |
| `net_amount_rate` | `numeric` | `YES` |
| `total_mv` | `numeric` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

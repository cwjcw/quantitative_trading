# analytics.tushare_daily_price

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：Tushare A 股日线价格视图，包含开高低收、昨收、涨跌额、涨跌幅、成交量和成交额。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `YES` |
| `trade_date` | `date` | `YES` |
| `open` | `numeric` | `YES` |
| `high` | `numeric` | `YES` |
| `low` | `numeric` | `YES` |
| `close` | `numeric` | `YES` |
| `pre_close` | `numeric` | `YES` |
| `change` | `numeric` | `YES` |
| `pct_chg` | `numeric` | `YES` |
| `vol` | `numeric` | `YES` |
| `amount` | `numeric` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

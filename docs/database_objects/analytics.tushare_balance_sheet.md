# analytics.tushare_balance_sheet

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：资产负债表视图，包含货币资金、应收账款、存货、总资产、负债和股东权益等。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `YES` |
| `ann_date` | `date` | `YES` |
| `end_date` | `date` | `YES` |
| `report_type` | `text` | `YES` |
| `money_cap` | `numeric` | `YES` |
| `accounts_receiv` | `numeric` | `YES` |
| `inventories` | `numeric` | `YES` |
| `total_cur_assets` | `numeric` | `YES` |
| `total_assets` | `numeric` | `YES` |
| `total_cur_liab` | `numeric` | `YES` |
| `total_liab` | `numeric` | `YES` |
| `total_hldr_eqy_exc_min_int` | `numeric` | `YES` |
| `raw` | `jsonb` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

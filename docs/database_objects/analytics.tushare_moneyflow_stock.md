# analytics.tushare_moneyflow_stock

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：个股资金流视图，包含小单、中单、大单、特大单和主力净流入等字段。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `YES` |
| `trade_date` | `date` | `YES` |
| `buy_sm_amount` | `numeric` | `YES` |
| `sell_sm_amount` | `numeric` | `YES` |
| `buy_md_amount` | `numeric` | `YES` |
| `sell_md_amount` | `numeric` | `YES` |
| `buy_lg_amount` | `numeric` | `YES` |
| `sell_lg_amount` | `numeric` | `YES` |
| `buy_elg_amount` | `numeric` | `YES` |
| `sell_elg_amount` | `numeric` | `YES` |
| `net_mf_amount` | `numeric` | `YES` |
| `net_mf_vol` | `numeric` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

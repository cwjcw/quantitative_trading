# analytics.tushare_cashflow_statement

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：现金流量表视图，包含经营、投资、筹资现金流和自由现金流等。
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
| `c_fr_sale_sg` | `numeric` | `YES` |
| `c_paid_goods_s` | `numeric` | `YES` |
| `n_cashflow_act` | `numeric` | `YES` |
| `n_cashflow_inv_act` | `numeric` | `YES` |
| `n_cash_flows_fnc_act` | `numeric` | `YES` |
| `free_cashflow` | `numeric` | `YES` |
| `c_cash_equ_end_period` | `numeric` | `YES` |
| `raw` | `jsonb` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

# analytics.tushare_income_statement

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：利润表视图，包含营业收入、营业成本、利润总额、归母净利润等。
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
| `total_revenue` | `numeric` | `YES` |
| `revenue` | `numeric` | `YES` |
| `oper_cost` | `numeric` | `YES` |
| `sell_exp` | `numeric` | `YES` |
| `admin_exp` | `numeric` | `YES` |
| `fin_exp` | `numeric` | `YES` |
| `operate_profit` | `numeric` | `YES` |
| `total_profit` | `numeric` | `YES` |
| `n_income` | `numeric` | `YES` |
| `n_income_attr_p` | `numeric` | `YES` |
| `raw` | `jsonb` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

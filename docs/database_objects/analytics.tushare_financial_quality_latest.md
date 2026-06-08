# analytics.tushare_financial_quality_latest

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：最新一期财务质量宽表，合并财务指标、利润表、资产负债表和现金流量表。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `YES` |
| `ann_date` | `date` | `YES` |
| `end_date` | `date` | `YES` |
| `eps` | `numeric` | `YES` |
| `ocfps` | `numeric` | `YES` |
| `grossprofit_margin` | `numeric` | `YES` |
| `netprofit_margin` | `numeric` | `YES` |
| `roe` | `numeric` | `YES` |
| `roa` | `numeric` | `YES` |
| `roic` | `numeric` | `YES` |
| `debt_to_assets` | `numeric` | `YES` |
| `current_ratio` | `numeric` | `YES` |
| `total_revenue` | `numeric` | `YES` |
| `revenue` | `numeric` | `YES` |
| `n_income_attr_p` | `numeric` | `YES` |
| `total_assets` | `numeric` | `YES` |
| `total_liab` | `numeric` | `YES` |
| `n_cashflow_act` | `numeric` | `YES` |
| `free_cashflow` | `numeric` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

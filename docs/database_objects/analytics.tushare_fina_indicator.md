# analytics.tushare_fina_indicator

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：财务指标视图，包含 ROE、ROA、毛利率、净利率、偿债能力和运营效率等。
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
| `dt_eps` | `numeric` | `YES` |
| `bps` | `numeric` | `YES` |
| `ocfps` | `numeric` | `YES` |
| `grossprofit_margin` | `numeric` | `YES` |
| `netprofit_margin` | `numeric` | `YES` |
| `roe` | `numeric` | `YES` |
| `roe_waa` | `numeric` | `YES` |
| `roa` | `numeric` | `YES` |
| `roic` | `numeric` | `YES` |
| `debt_to_assets` | `numeric` | `YES` |
| `current_ratio` | `numeric` | `YES` |
| `quick_ratio` | `numeric` | `YES` |
| `assets_turn` | `numeric` | `YES` |
| `inv_turn` | `numeric` | `YES` |
| `ar_turn` | `numeric` | `YES` |
| `raw` | `jsonb` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

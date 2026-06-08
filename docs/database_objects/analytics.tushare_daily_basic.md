# analytics.tushare_daily_basic

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：Tushare 日频估值和流动性指标视图，包含 PE/PB、市值、换手率、量比等。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `YES` |
| `trade_date` | `date` | `YES` |
| `turnover_rate` | `numeric` | `YES` |
| `turnover_rate_f` | `numeric` | `YES` |
| `volume_ratio` | `numeric` | `YES` |
| `pe` | `numeric` | `YES` |
| `pe_ttm` | `numeric` | `YES` |
| `pb` | `numeric` | `YES` |
| `ps_ttm` | `numeric` | `YES` |
| `dv_ttm` | `numeric` | `YES` |
| `total_share` | `numeric` | `YES` |
| `float_share` | `numeric` | `YES` |
| `free_share` | `numeric` | `YES` |
| `total_mv` | `numeric` | `YES` |
| `circ_mv` | `numeric` | `YES` |
| `fetched_at` | `timestamp with time zone` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

# analytics.tushare_stock_feature_daily

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：个股日频特征宽表，整合日线价格、估值、资金流和筹码指标，供选股模型直接使用。
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
| `turnover_rate` | `numeric` | `YES` |
| `turnover_rate_f` | `numeric` | `YES` |
| `volume_ratio` | `numeric` | `YES` |
| `pe_ttm` | `numeric` | `YES` |
| `pb` | `numeric` | `YES` |
| `total_mv` | `numeric` | `YES` |
| `circ_mv` | `numeric` | `YES` |
| `net_mf_amount` | `numeric` | `YES` |
| `net_mf_vol` | `numeric` | `YES` |
| `winner_rate` | `numeric` | `YES` |
| `cost_50pct` | `numeric` | `YES` |
| `weight_avg` | `numeric` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

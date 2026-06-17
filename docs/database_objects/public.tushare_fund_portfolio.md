# public.tushare_fund_portfolio

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：基金持仓数据，来自 Tushare `fund_portfolio`。
- 主键：`ts_code, ann_date, symbol`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `NO` |
| `ann_date` | `date` | `NO` |
| `end_date` | `date` | `YES` |
| `symbol` | `text` | `NO` |
| `mkv` | `numeric` | `YES` |
| `amount` | `numeric` | `YES` |
| `stk_mkv_ratio` | `numeric` | `YES` |
| `stk_float_ratio` | `numeric` | `YES` |
| `fetched_at` | `timestamp with time zone` | `NO` |

## 使用建议

用于分析基金重仓股、行业暴露、持仓变化和个股机构持有情况。

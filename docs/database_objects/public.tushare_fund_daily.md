# public.tushare_fund_daily

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：场内基金日行情，来自 Tushare `fund_daily`。
- 主键：`ts_code, trade_date`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `NO` |
| `trade_date` | `date` | `NO` |
| `pre_close` | `numeric` | `YES` |
| `open` | `numeric` | `YES` |
| `high` | `numeric` | `YES` |
| `low` | `numeric` | `YES` |
| `close` | `numeric` | `YES` |
| `change` | `numeric` | `YES` |
| `pct_chg` | `numeric` | `YES` |
| `vol` | `numeric` | `YES` |
| `amount` | `numeric` | `YES` |
| `fetched_at` | `timestamp with time zone` | `NO` |

## 使用建议

用于 ETF、LOF 等场内基金行情分析。场外基金净值分析使用 `public.tushare_fund_nav`。

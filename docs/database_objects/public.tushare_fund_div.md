# public.tushare_fund_div

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：基金分红数据，来自 Tushare `fund_div`。
- 主键：`ts_code, ann_date, base_date, div_cash`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `NO` |
| `ann_date` | `date` | `NO` |
| `imp_anndate` | `date` | `YES` |
| `base_date` | `date` | `NO` |
| `div_proc` | `text` | `YES` |
| `record_date` | `date` | `YES` |
| `ex_date` | `date` | `YES` |
| `pay_date` | `date` | `YES` |
| `earpay_date` | `date` | `YES` |
| `net_ex_date` | `date` | `YES` |
| `div_cash` | `numeric` | `NO` |
| `base_unit` | `numeric` | `YES` |
| `ear_distr` | `numeric` | `YES` |
| `ear_amount` | `numeric` | `YES` |
| `account_date` | `date` | `YES` |
| `base_year` | `text` | `YES` |
| `fetched_at` | `timestamp with time zone` | `NO` |

## 使用建议

用于分析基金分红、除息和累计净值变化。

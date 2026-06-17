# public.tushare_fund_company

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：基金公司、资管和投资管理机构信息，来自 Tushare `fund_company`。
- 主键：`entity_key`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `entity_key` | `text` | `NO` |
| `name` | `text` | `YES` |
| `shortname` | `text` | `YES` |
| `province` | `text` | `YES` |
| `city` | `text` | `YES` |
| `address` | `text` | `YES` |
| `phone` | `text` | `YES` |
| `office` | `text` | `YES` |
| `website` | `text` | `YES` |
| `chairman` | `text` | `YES` |
| `manager` | `text` | `YES` |
| `reg_capital` | `numeric` | `YES` |
| `setup_date` | `date` | `YES` |
| `end_date` | `date` | `YES` |
| `employees` | `numeric` | `YES` |
| `main_business` | `text` | `YES` |
| `org_code` | `text` | `YES` |
| `credit_code` | `text` | `YES` |
| `fetched_at` | `timestamp with time zone` | `NO` |

## 使用建议

用于分析基金管理机构背景。注意本表是机构数据，不是私募产品明细。

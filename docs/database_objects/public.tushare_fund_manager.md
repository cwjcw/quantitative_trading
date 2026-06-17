# public.tushare_fund_manager

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：基金经理任职、离任和履历信息，来自 Tushare `fund_manager`。
- 主键：`manager_key`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `manager_key` | `text` | `NO` |
| `ts_code` | `text` | `NO` |
| `ann_date` | `date` | `YES` |
| `name` | `text` | `NO` |
| `gender` | `text` | `YES` |
| `birth_year` | `text` | `YES` |
| `edu` | `text` | `YES` |
| `nationality` | `text` | `YES` |
| `begin_date` | `date` | `YES` |
| `end_date` | `date` | `YES` |
| `resume` | `text` | `YES` |
| `fetched_at` | `timestamp with time zone` | `NO` |

## 使用建议

用于分析基金经理任职区间、变更事件和基金业绩的关系。部分 Tushare 记录缺少 `begin_date`，因此使用 `manager_key` 作为稳定主键。

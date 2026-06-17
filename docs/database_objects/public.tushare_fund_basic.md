# public.tushare_fund_basic

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：基金基础信息，来自 Tushare `fund_basic`，包含开放式基金和场内基金。
- 主键：`ts_code`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `ts_code` | `text` | `NO` |
| `name` | `text` | `YES` |
| `management` | `text` | `YES` |
| `custodian` | `text` | `YES` |
| `fund_type` | `text` | `YES` |
| `found_date` | `date` | `YES` |
| `due_date` | `date` | `YES` |
| `list_date` | `date` | `YES` |
| `issue_date` | `date` | `YES` |
| `delist_date` | `date` | `YES` |
| `issue_amount` | `numeric` | `YES` |
| `m_fee` | `numeric` | `YES` |
| `c_fee` | `numeric` | `YES` |
| `duration_year` | `numeric` | `YES` |
| `p_value` | `numeric` | `YES` |
| `min_amount` | `numeric` | `YES` |
| `exp_return` | `text` | `YES` |
| `benchmark` | `text` | `YES` |
| `status` | `text` | `YES` |
| `invest_type` | `text` | `YES` |
| `type` | `text` | `YES` |
| `trustee` | `text` | `YES` |
| `purc_startdate` | `date` | `YES` |
| `redm_startdate` | `date` | `YES` |
| `market` | `text` | `YES` |
| `fetched_at` | `timestamp with time zone` | `NO` |

## 使用建议

用于构建基金池、区分开放式基金 `O` 和场内基金 `E`、按基金类型和成立日期筛选。

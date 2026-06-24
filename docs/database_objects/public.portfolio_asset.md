# public.portfolio_asset

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：每个账户最新一条资金快照。
- 主键：`account_id`

## 字段

| 字段 | 类型 | 可空 | 含义 |
| --- | --- | --- | --- |
| `account_id` | `text` | `NO` | 账户编号。 |
| `cash` | `numeric` | `YES` | 可用资金。 |
| `frozen_cash` | `numeric` | `YES` | 冻结资金。 |
| `market_value` | `numeric` | `YES` | 持仓市值。 |
| `total_asset` | `numeric` | `YES` | 总资产，业务口径为 `cash + market_value`。 |
| `content_hash` | `text` | `NO` | 当前快照内容的 SHA256，用于去重和变更检测。 |
| `fetched_at` | `timestamp with time zone` | `NO` | 快照抓取时间。 |

## 索引

- `portfolio_asset_pkey`：唯一索引，字段为 `account_id`。

## 使用建议

该表只保存账户当前资金状态，不保留历史版本。查询资金曲线时使用
`public.portfolio_changes` 中 `kind = 'asset'` 的记录。

```sql
SELECT *
FROM public.portfolio_asset
WHERE account_id = 'your_account';
```

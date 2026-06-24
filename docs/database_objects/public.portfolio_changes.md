# public.portfolio_changes

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：账户资金和持仓的 append-only 变更审计日志。
- 主键：`change_id`

## 字段

| 字段 | 类型 | 可空 | 含义 |
| --- | --- | --- | --- |
| `change_id` | `bigint` | `NO` | 自增变更编号。 |
| `account_id` | `text` | `NO` | 账户编号。 |
| `kind` | `text` | `NO` | `asset` 或 `position`。 |
| `stock_code` | `text` | `YES` | 股票代码；资金变更时为空。 |
| `change_type` | `text` | `NO` | `insert`、`update` 或 `delete`。 |
| `old_hash` | `text` | `YES` | 变更前内容 hash。 |
| `new_hash` | `text` | `YES` | 变更后内容 hash。 |
| `snapshot` | `jsonb` | `YES` | 本次变更对应的完整快照。 |
| `detected_at` | `timestamp with time zone` | `NO` | 变更写入时间，默认 `NOW()`。 |

## 索引

- `portfolio_changes_pkey`：唯一索引，字段为 `change_id`。
- `idx_portfolio_changes_account_time`：字段为
  `account_id, detected_at DESC`。

## 使用建议

这是账户数据中具有历史时间序列语义的表，用于资金曲线、持仓变化和状态审计。
按股票查询持仓历史时同时限定 `kind = 'position'`；查询资金曲线时限定
`kind = 'asset'`。

```sql
SELECT detected_at, change_type, snapshot
FROM public.portfolio_changes
WHERE account_id = 'your_account'
  AND stock_code = '600519.SH'
  AND kind = 'position'
ORDER BY detected_at DESC;
```

```sql
SELECT
    detected_at,
    (snapshot->>'total_asset')::numeric AS total_asset
FROM public.portfolio_changes
WHERE account_id = 'your_account'
  AND kind = 'asset'
ORDER BY detected_at;
```

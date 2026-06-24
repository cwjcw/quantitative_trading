# public.portfolio_positions

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：每个账户、每只股票最新一条持仓快照。
- 主键：`account_id, stock_code`

## 字段

| 字段 | 类型 | 可空 | 含义 |
| --- | --- | --- | --- |
| `account_id` | `text` | `NO` | 账户编号。 |
| `stock_code` | `text` | `NO` | 股票代码，例如 `600519.SH`。 |
| `stock_name` | `text` | `YES` | 股票名称。 |
| `volume` | `bigint` | `YES` | 持仓数量。 |
| `can_use_volume` | `bigint` | `YES` | 当前可卖数量。 |
| `open_price` | `numeric` | `YES` | 开仓成本。 |
| `avg_price` | `numeric` | `YES` | 持仓均价。 |
| `market_value` | `numeric` | `YES` | 持仓市值。 |
| `frozen_volume` | `bigint` | `YES` | 冻结数量。 |
| `on_road_volume` | `bigint` | `YES` | 在途数量。 |
| `yesterday_volume` | `bigint` | `YES` | 昨日持仓数量。 |
| `content_hash` | `text` | `NO` | 当前快照内容的 SHA256。 |
| `fetched_at` | `timestamp with time zone` | `NO` | 快照抓取时间。 |

## 索引

- `portfolio_positions_pkey`：唯一索引，字段为 `account_id, stock_code`。
- `idx_portfolio_positions_account`：字段为 `account_id, fetched_at DESC`。

## 使用建议

该表用于读取当前持仓，不具有历史时间序列语义。股票被清仓后，当前状态行可被
同步脚本删除，对应删除事件应从 `public.portfolio_changes` 查询。

```sql
SELECT *
FROM public.portfolio_positions
WHERE account_id = 'your_account'
ORDER BY market_value DESC;
```

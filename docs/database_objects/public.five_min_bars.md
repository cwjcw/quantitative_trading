# public.five_min_bars

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：历史遗留 5 分钟 K 线表；新研究优先使用 stock_5m_bars。
- 估算行数：`-1`
- 存储大小：`32.0 KB`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `run_id` | `uuid` | `NO` |
| `trade_date` | `date` | `NO` |
| `captured_at` | `timestamp with time zone` | `NO` |
| `stock_code` | `text` | `NO` |
| `exchange` | `text` | `YES` |
| `bar_time` | `bigint` | `NO` |
| `period` | `text` | `NO` |
| `open` | `numeric` | `YES` |
| `high` | `numeric` | `YES` |
| `low` | `numeric` | `YES` |
| `close` | `numeric` | `YES` |
| `volume` | `bigint` | `YES` |
| `amount` | `numeric` | `YES` |
| `pre_close` | `numeric` | `YES` |

## 使用建议

这是本地 QMT 行情/合约数据层；与 Tushare 外生数据按股票代码和日期/时间拼接使用。

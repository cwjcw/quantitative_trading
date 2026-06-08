# public.stock_snapshots

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：QMT 实时快照表，保存采集时点的盘口/价格/成交等实时状态。
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
| `raw_time` | `bigint` | `YES` |
| `last_price` | `numeric` | `YES` |
| `open_price` | `numeric` | `YES` |
| `high_price` | `numeric` | `YES` |
| `low_price` | `numeric` | `YES` |
| `last_close` | `numeric` | `YES` |
| `volume` | `bigint` | `YES` |
| `amount` | `numeric` | `YES` |
| `ask_price` | `jsonb` | `YES` |
| `bid_price` | `jsonb` | `YES` |
| `ask_volume` | `jsonb` | `YES` |
| `bid_volume` | `jsonb` | `YES` |
| `raw` | `jsonb` | `NO` |
| `instrument_name` | `text` | `YES` |
| `change_amount` | `numeric` | `YES` |
| `change_percent` | `numeric` | `YES` |
| `amplitude_percent` | `numeric` | `YES` |
| `avg_price` | `numeric` | `YES` |
| `pvolume` | `bigint` | `YES` |
| `float_volume` | `numeric` | `YES` |
| `total_volume` | `numeric` | `YES` |
| `turnover_rate_float` | `numeric` | `YES` |
| `turnover_rate_total` | `numeric` | `YES` |
| `up_stop_price` | `numeric` | `YES` |
| `down_stop_price` | `numeric` | `YES` |
| `market_phase` | `text` | `NO` |

## 使用建议

这是本地 QMT 行情/合约数据层；与 Tushare 外生数据按股票代码和日期/时间拼接使用。

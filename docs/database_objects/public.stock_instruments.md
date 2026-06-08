# public.stock_instruments

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：本地股票合约/基础信息表，是本地可交易股票池的重要来源。
- 估算行数：`10416`
- 存储大小：`10.5 MB`
- 主键：`trade_date, stock_code`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `trade_date` | `date` | `NO` |
| `stock_code` | `text` | `NO` |
| `instrument_name` | `text` | `YES` |
| `exchange` | `text` | `YES` |
| `pre_close` | `numeric` | `YES` |
| `up_stop_price` | `numeric` | `YES` |
| `down_stop_price` | `numeric` | `YES` |
| `float_volume` | `numeric` | `YES` |
| `total_volume` | `numeric` | `YES` |
| `price_tick` | `numeric` | `YES` |
| `volume_multiple` | `integer` | `YES` |
| `raw` | `jsonb` | `NO` |

## 使用建议

这是本地 QMT 行情/合约数据层；与 Tushare 外生数据按股票代码和日期/时间拼接使用。

# public.stock_5m_bars

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：QMT 标准 5 分钟行情表，保存股票分钟级 OHLCV，是日内特征和回测的核心行情源。
- 估算行数：`-1`
- 存储大小：`40.0 KB`
- 主键：`stock_code, bar_time`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `stock_code` | `text` | `NO` |
| `trade_date` | `date` | `NO` |
| `bar_time` | `timestamp with time zone` | `NO` |
| `exchange` | `text` | `YES` |
| `open_price` | `numeric` | `YES` |
| `high_price` | `numeric` | `YES` |
| `low_price` | `numeric` | `YES` |
| `close_price` | `numeric` | `YES` |
| `pre_close` | `numeric` | `YES` |
| `volume` | `bigint` | `YES` |
| `pvolume` | `bigint` | `YES` |
| `amount` | `numeric` | `YES` |
| `source` | `text` | `NO` |
| `raw` | `jsonb` | `NO` |

## 使用建议

这是本地 QMT 行情/合约数据层；与 Tushare 外生数据按股票代码和日期/时间拼接使用。

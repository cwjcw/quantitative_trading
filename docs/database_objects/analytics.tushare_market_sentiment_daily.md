# analytics.tushare_market_sentiment_daily

- 类型：`VIEW`
- Schema：`analytics`
- 主要内容：市场情绪日频聚合视图，统计涨跌停、龙虎榜、大宗交易、公告和研报活跃度。
- 估算行数：`-1`
- 存储大小：`0 B`
- 主键：`无/视图不适用`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `trade_date` | `date` | `YES` |
| `limit_event_count` | `bigint` | `YES` |
| `top_list_count` | `bigint` | `YES` |
| `top_inst_count` | `bigint` | `YES` |
| `block_trade_count` | `bigint` | `YES` |
| `announcement_count` | `bigint` | `YES` |
| `research_report_count` | `bigint` | `YES` |

## 使用建议

优先从该分析视图读取标准化字段；如需未映射字段，再回到 `public.tushare_raw_records.raw` 查询原始 JSON。

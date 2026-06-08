# public.tushare_collection_checkpoints

- 类型：`BASE TABLE`
- Schema：`public`
- 主要内容：逐股或长任务采集 checkpoint 表，用于断点续跑、跳过已完成任务和记录失败原因。
- 估算行数：`26080`
- 存储大小：`8.6 MB`
- 主键：`checkpoint_key`

## 字段

| 字段 | 类型 | 可空 |
| --- | --- | --- |
| `checkpoint_key` | `text` | `NO` |
| `endpoint` | `text` | `NO` |
| `ts_code` | `text` | `NO` |
| `content_type` | `text` | `NO` |
| `range_start` | `date` | `YES` |
| `range_end` | `date` | `YES` |
| `status` | `text` | `NO` |
| `attempts` | `integer` | `NO` |
| `row_count` | `integer` | `NO` |
| `error_message` | `text` | `YES` |
| `updated_at` | `timestamp with time zone` | `NO` |

## 使用建议

结合采集运行记录和字段含义使用；生产查询建议优先走业务视图或带过滤条件查询。

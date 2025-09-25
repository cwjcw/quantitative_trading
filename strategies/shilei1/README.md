# 首波回撤选股器

该方案用于寻找经历“首波翻倍—深度回撤—波动收敛”的个股，并输出各项指标与是否通过筛选的结果。所有行情与历史K线均由 AkShare 获取。

## 筛选规则

1. **首波上涨**：在最近约 200 个交易日内，从波段起点到高点的涨幅不低于 100%（使用日内最高价确认）。
2. **深幅回撤**：高点后最低价距离波峰至少回撤 30%，用于捕捉洗盘阶段。
3. **波动收敛**：最近 20 个交易日的收盘波动率小于前 60 个交易日的波动率乘以阈值（默认 0.7）。
4. **价格保护线**：当前收盘价不低于首波起涨价的 35%。
5. **趋势保持**：在近 360 个交易日（或可用样本）中，至少 40% 的收盘价高于 30 日均线。
6. **健康过滤**：剔除 ST 股票，并默认要求总市值不低于 100 亿元（可在配置中修改）。

## 使用方式

```bash
python -m strategies.shilei1.src.screener --help
# 仅列出市值≥100亿元（可叠加 --max 控制数量）
python -m strategies.shilei1.src.screener --list-large-cap
# 运行筛选并导出全量结果到 CSV（默认写入 strategies/shilei1/data/shilei1.csv）
python -m strategies.shilei1.src.screener --show-progress --workers 10 --export-csv
python -m strategies.shilei1.src.screener --export-csv my_result.csv
```

程序默认会下载所有 A 股（含北交所）行情数据，并在本地根据上述规则完成评估。导出的 CSV 包含每个条件的数值、是否合格，以及通过条件计数，按通过数量降序排列便于查看。

## 目录结构

- `data/` AkShare 抓取的缓存数据或筛选结果，可在此查看 `shilei1.csv`。
- `notebooks/` Jupyter Notebook，便于做回测或参数敏感度分析。
- `src/` 主要代码，`screener.py` 提供命令行入口与核心逻辑，`config.py` 集中管理参数。

## 参数与扩展

- 所有可调参数归档在 `strategies/shilei1/src/config.py`，例如可将 `first_wave_gain` 调整为 `0.8` 捕捉弱势首波，或将 `market_cap_floor` 调成 `5e9` 放宽市值限制。
- `--workers` 控制并行线程数（默认 4），在本地网络带宽允许的情况下可提高到 8~12 以加速筛选。
- 举例：
  ```python
  from strategies.shilei1.src import config, screener
  cfg = config.ScreenConfig(first_wave_gain=0.8, market_cap_floor=5e9)
  results = screener.run_screen(config=cfg, workers=8, show_progress=True)
  ```
- 如需外部化参数可扩展 `config.py` 读取 JSON/YAML，再在 `screener.py` 中加载。

## 下一步建议

- 对入选股票做回测，评估不同市场环境下的命中率与收益表现。
- 在 `notebooks/` 中结合行业、基本面或因子数据，构建进一步的过滤或排序逻辑。
- 将脚本纳入定时任务或 CI，定期输出最新候选名单。

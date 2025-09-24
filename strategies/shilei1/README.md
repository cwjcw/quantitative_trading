# First Wave Pullback Screener

This screen searches for stocks that experienced a strong first rally (>100%) within the last ~200 trading days, followed by a deep pullback and subsequent volatility contraction.

## Selection rules

1. **First wave rally**: price doubled (≥100% gain) from a prior swing low within roughly the past 200 trading days, measured using the intraday high to confirm the magnitude.
2. **Deep pullback**: price retraced ≥30% from the first wave's peak, measured on the post-peak low price to capture intraday capitulation.
3. **Volatility contraction**: recent 20-day close-to-close volatility is lower than the preceding 60-day volatility window.
4. **Current price floor**: latest close ≥35% of the first wave starting price (keeps the structure intact).
5. **Trend persistence**: within the last 360 trading days (or what is available), at least 40% of the closes stayed above the 30-day moving average.
6. **Stock health filters**: exclude `ST` tickers and any stock with total market cap < 100亿元.

The logic is implemented in `src/screener.py`. Adjust parameters inside the script to fine-tune each rule.

### 参数亮点

- `first_wave_gain=1.0`：默认要求波段涨幅至少翻倍，可根据市场风格调至 0.8~1.2。
- `first_wave_lookback_days=200`：只在最近约200个交易日内寻找第一波，避免历史噪音。
- `pullback_threshold=0.3`：回撤阈值使用波段高点后的最低价，更贴近极端洗盘。
- 30日均线上方占比按 `144/360≈40%` 的比例缩放，即便样本不足 360 天也能评估。


## Usage

```bash
python -m strategies.shilei1.src.screener --help
# 仅列出市值≥100亿元（可叠加 --max 控制数量）
python -m strategies.shilei1.src.screener --list-large-cap
# 运行筛选并导出全量结果至 CSV（默认写入 strategies/shilei1/data/shilei1.csv）
python -m strategies.shilei1.src.screener --export-csv
python -m strategies.shilei1.src.screener --export-csv custom.csv
# 带进度条运行（需安装 tqdm）
python -m strategies.shilei1.src.screener --show-progress --export-csv
```

默认运行时使用 AkShare 获取最新 A 股行情快照，并通过东方财富 REST 接口抓取复权历史K线数据。

## Folder guide

- `data/` place cached raw data pulled from 东方财富或其他公开数据源。
- `notebooks/` exploratory analysis or backtesting notebooks.
- `src/` reusable screening code and helpers.

## 调整参数

- 所有可调节参数集中在 `strategies/shilei1/src/config.py`，直接修改 `ScreenConfig` 中的字段即可，例如将 `first_wave_gain` 调至 `0.8` 用于捕捉稍弱的第一波。
- 可使用 `python -m strategies.shilei1.src.screener --list-large-cap` 快速查看市值≥100亿元的股票名单。
- 支持 `--export-csv` 将所有筛选指标导出到 CSV（默认写入 `strategies/shilei1/data/shilei1.csv`），也可指定自定义文件名，导出内容按通过条件数量降序排序。
- 使用 `--show-progress` 可在筛选过程中显示进度（如安装 tqdm 会显示进度条）。
- 如果需要临时实验，可在运行脚本时加载自定义配置：
  ```python
  from strategies.shilei1.src import config, screener
  my_cfg = config.ScreenConfig(first_wave_gain=0.8)
  results = screener.run_screen(config=my_cfg)
  ```
- 若想外部化（YAML/JSON），可在 `config.py` 中新增解析函数并在 `screener.py` 调用。

## Next steps

- Backtest the screen to validate hit rates across sectors.
- Add factor exposures or risk controls as optional post-filters.
- Wrap the screener in a scheduled job under `scripts/` once parameters are stable.

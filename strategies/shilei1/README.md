# First Wave Pullback Screener

This screen searches for stocks that experienced a strong first rally (>100%) followed by a deep pullback and a subsequent volatility contraction.

## Selection rules

1. **First wave rally**: price doubled (≥100% gain) from a prior swing low during the last ~2 years.
2. **Deep pullback**: price retraced ≥30% from the first wave's peak.
3. **Volatility contraction**: recent 20-day close-to-close volatility is lower than the preceding 60-day volatility window.
4. **Current price floor**: latest close ≥35% of the first wave starting price (keeps the structure intact).
5. **Trend persistence**: within the last 360 trading days, at least 144 closes stayed above the 30-day moving average.
6. **Stock health filters**: exclude `ST` tickers and any stock with total market cap < 100 billion CNY.

The logic is implemented in `src/screener.py`. Adjust parameters inside the script to fine-tune each rule.

## Usage

```bash
python -m strategies.shilei1.src.screener --help
```

By default no network calls are executed until you provide ticker symbols. The script relies on AkShare endpoints to fetch daily price history and market snapshot data.

## Folder guide

- `data/` place cached raw data pulled from AkShare or other sources.
- `notebooks/` exploratory analysis or backtesting notebooks.
- `src/` reusable screening code and helpers.

## Next steps

- Backtest the screen to validate hit rates across sectors.
- Add factor exposures or risk controls as optional post-filters.
- Wrap the screener in a scheduled job under `scripts/` once parameters are stable.

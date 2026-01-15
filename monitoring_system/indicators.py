from __future__ import annotations

import pandas as pd


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(0)


def vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty or "amount" not in df.columns or "volume" not in df.columns:
        return pd.Series([], dtype=float)
    volume = df["volume"].replace(0, pd.NA) * 100
    vwap_val = (df["amount"].cumsum() / volume.cumsum()).fillna(method="ffill")
    return vwap_val

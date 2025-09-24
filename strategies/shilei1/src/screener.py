"""Screen for first-wave pullback opportunities using Eastmoney data."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
import requests

SESSION = requests.Session()
SESSION.trust_env = False

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

SPOT_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
HISTORY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

MAX_RESPONSE_PREVIEW = 200


def _request_json(url: str, params: dict) -> dict:
    """Call Eastmoney endpoint with standard headers and return parsed JSON."""

    response = SESSION.get(
        url,
        params=params,
        headers=HEADERS,
        proxies=None,
        timeout=10,
    )
    response.raise_for_status()
    if not response.encoding:
        response.encoding = 'utf-8'
    text = response.text.strip()
    if not text:
        raise ValueError(f'{url} 接口返回空响应')
    try:
        return response.json()
    except ValueError as exc:
        snippet = text[:MAX_RESPONSE_PREVIEW]
        raise ValueError(f'{url} 接口返回非 JSON 数据: {snippet}') from exc


@dataclass
class ScreenConfig:
    """Parameters controlling the pullback screen."""

    history_years: int = 3
    volatility_window_short: int = 20
    volatility_window_long: int = 60
    volatility_ratio_threshold: float = 0.7
    pullback_threshold: float = 0.3
    market_cap_floor: float = 1e10  # 100 亿元 (10 billion CNY)
    price_floor_ratio: float = 0.35
    trend_days_required: int = 144
    trend_window: int = 360


@dataclass
class ScreenResult:
    """Holds the evaluation outcome for a single ticker."""

    code: str
    name: str
    passed: bool
    reason: Optional[str]
    data_points: dict


def _fetch_spot_snapshot() -> pd.DataFrame:
    """Return the latest spot snapshot for all A-share equities."""

    params = {
        "pn": "1",
        "pz": "5000",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
    }
    payload = _request_json(SPOT_URL, params)
    diff = (payload.get("data") or {}).get("diff") or []
    if not diff:
        raise ValueError("行情列表为空")

    spot = pd.DataFrame(diff)
    spot = spot.rename(
        columns={
            "f12": "code",
            "f14": "name",
            "f2": "close",
            "f20": "market_cap",
        }
    )
    spot["code"] = spot["code"].astype(str)
    spot["close"] = pd.to_numeric(spot["close"], errors="coerce")
    spot["market_cap"] = pd.to_numeric(spot["market_cap"], errors="coerce")
    spot.dropna(subset=["code", "close", "market_cap"], inplace=True)
    return spot[["code", "name", "close", "market_cap"]]


def _filter_base_universe(spot: pd.DataFrame, config: ScreenConfig) -> pd.DataFrame:
    """Apply static filters (ST status and market cap)."""

    mask = (
        ~spot["name"].str.contains("ST", case=False, na=False)
        & (spot["market_cap"].astype(float) >= config.market_cap_floor)
    )
    return spot.loc[mask].copy()


def _fetch_history(code: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download daily price history via the Eastmoney REST API."""

    market_prefix = "1" if code.startswith(("5", "6", "9")) else "0"
    secid = f"{market_prefix}.{code}"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }
    payload = _request_json(HISTORY_URL, params)
    klines = (payload.get("data") or {}).get("klines") or []

    records = []
    for item in klines:
        parts = item.split(",")
        if len(parts) < 6:
            continue
        records.append(
            {
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
            }
        )

    history = pd.DataFrame(records)
    if history.empty:
        return history

    history["date"] = pd.to_datetime(history["date"])
    for column in ["open", "close", "high", "low", "volume"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history.dropna(subset=["close"], inplace=True)
    history.sort_values("date", inplace=True)
    history.reset_index(drop=True, inplace=True)
    return history


def _locate_first_wave(df: pd.DataFrame) -> Optional[dict]:
    """Find the first instance where price doubles from a prior swing low."""

    if df.empty:
        return None

    min_price = df.loc[0, "close"]
    min_idx = 0
    first_wave = None

    for idx, close in enumerate(df["close"]):
        if close < min_price:
            min_price = close
            min_idx = idx
            continue

        gain = (close - min_price) / min_price
        if gain >= 1.0:
            window = df.loc[min_idx : idx, "close"]
            peak_idx = window.idxmax()
            peak_price = window.max()
            first_wave = {
                "start_idx": min_idx,
                "start_price": float(min_price),
                "peak_idx": int(peak_idx),
                "peak_price": float(peak_price),
            }
            break

    return first_wave


def _check_pullback(df: pd.DataFrame, wave: dict, config: ScreenConfig) -> tuple[bool, Optional[float]]:
    """Verify price retraced the required percentage after the first wave."""

    if not wave:
        return False, None

    peak_idx = wave["peak_idx"]
    peak_price = wave["peak_price"]
    subsequent = df.loc[peak_idx + 1 :, "close"]
    if subsequent.empty:
        return False, None

    low_after_peak = subsequent.min()
    pullback = (peak_price - low_after_peak) / peak_price
    return pullback >= config.pullback_threshold, float(low_after_peak)


def _check_volatility_contraction(df: pd.DataFrame, config: ScreenConfig) -> bool:
    """Ensure recent realized volatility contracted versus the longer window."""

    if len(df) < config.volatility_window_short + config.volatility_window_long:
        return False

    returns = df["close"].pct_change().dropna()

    recent = returns.tail(config.volatility_window_short)
    prior = returns.iloc[-(config.volatility_window_short + config.volatility_window_long) : -config.volatility_window_short]

    if prior.empty or recent.empty:
        return False

    recent_std = recent.std(ddof=0)
    prior_std = prior.std(ddof=0)

    if prior_std == 0:
        return False

    return recent_std < prior_std * config.volatility_ratio_threshold


def _check_trend_persistence(df: pd.DataFrame, config: ScreenConfig) -> bool:
    """Count closes above the 30-day moving average within the specified window."""

    if len(df) < config.trend_window:
        return False

    window_df = df.tail(config.trend_window).copy()
    window_df["ma30"] = window_df["close"].rolling(window=30).mean()
    window_df.dropna(inplace=True)
    if window_df.empty:
        return False

    days_above = (window_df["close"] > window_df["ma30"]).sum()
    return int(days_above) >= config.trend_days_required


def evaluate_ticker(row: pd.Series, start: dt.date, end: dt.date, config: ScreenConfig) -> ScreenResult:
    """Evaluate a single ticker against all screening rules."""

    code = row["code"]
    name = row["name"]

    history = _fetch_history(code, start, end)
    if history.empty:
        return ScreenResult(code, name, False, "无历史数据", {})

    wave = _locate_first_wave(history)
    if not wave:
        return ScreenResult(code, name, False, "未找到第一波上涨", {})

    pullback_ok, pullback_low = _check_pullback(history, wave, config)
    if not pullback_ok:
        return ScreenResult(code, name, False, "回调幅度不足", {"wave": wave})

    last_close = float(history["close"].iloc[-1])
    price_floor = wave["start_price"] * config.price_floor_ratio
    if last_close < price_floor:
        return ScreenResult(
            code,
            name,
            False,
            "当前价格低于起涨价保护线",
            {"wave": wave, "last_close": last_close, "price_floor": price_floor},
        )

    if not _check_volatility_contraction(history, config):
        return ScreenResult(code, name, False, "波动率未收敛", {"wave": wave})

    if not _check_trend_persistence(history, config):
        return ScreenResult(code, name, False, "趋势保持不达标", {"wave": wave})

    return ScreenResult(
        code,
        name,
        True,
        None,
        {
            "wave": wave,
            "pullback_low": pullback_low,
            "last_close": last_close,
        },
    )


def run_screen(
    symbols: Optional[Iterable[str]] = None,
    as_of: Optional[dt.date] = None,
    config: Optional[ScreenConfig] = None,
) -> list[ScreenResult]:
    """Run the screen for selected symbols or the entire filtered universe."""

    config = config or ScreenConfig()
    as_of = as_of or dt.date.today()
    start_date = as_of - dt.timedelta(days=365 * config.history_years)

    spot = _fetch_spot_snapshot()
    base = _filter_base_universe(spot, config)

    if symbols:
        base = base[base["code"].isin(set(symbols))]

    results: list[ScreenResult] = []
    for _, row in base.iterrows():
        try:
            result = evaluate_ticker(row, start_date, as_of, config)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            results.append(
                ScreenResult(
                    code=row["code"],
                    name=row["name"],
                    passed=False,
                    reason=f"发生异常:{exc}",
                    data_points={},
                )
            )
    return results


def _format_result(result: ScreenResult) -> str:
    """Render a single result row for terminal output."""

    status = "PASS" if result.passed else "FAIL"
    summary = f"{result.code} {result.name} [{status}]"
    if result.reason:
        summary += f" 原因={result.reason}"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first-wave pullback screener")
    parser.add_argument("--symbols", nargs="*", help="Optional list of stock codes to evaluate")
    parser.add_argument("--date", help="As-of date (YYYY-MM-DD)")
    parser.add_argument(
        "--max", type=int, default=None, help="Limit evaluation to the first N tickers of the universe"
    )
    args = parser.parse_args()

    as_of = dt.datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None

    results = run_screen(args.symbols, as_of=as_of)

    if args.max is not None:
        results = results[: args.max]

    passes = [res for res in results if res.passed]

    for res in results:
        print(_format_result(res))

    print("-" * 60)
    print(f"Total evaluated: {len(results)} | Passed: {len(passes)}")


if __name__ == "__main__":
    main()

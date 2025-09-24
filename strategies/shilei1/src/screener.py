"""Screen for first-wave pullback opportunities using Eastmoney data."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Optional

import math

import pandas as pd
import akshare as ak
import requests
from pathlib import Path

from .config import ScreenConfig, get_default_config

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

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_FILE = BASE_DIR / "data" / "shilei1.csv"

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
class ScreenResult:
    """Holds the evaluation outcome for a single ticker."""

    code: str
    name: str
    passed: bool
    reason: Optional[str]
    data_points: dict


def _fetch_spot_snapshot() -> pd.DataFrame:
    """Return the latest spot snapshot for all A-share equities."""

    spot = ak.stock_zh_a_spot_em()
    spot = spot.rename(
        columns={
            "代码": "code",
            "名称": "name",
            "最新价": "close",
            "总市值": "market_cap",
        }
    )
    spot["code"] = spot["code"].astype(str)
    spot["close"] = pd.to_numeric(spot["close"], errors="coerce")
    spot["market_cap"] = pd.to_numeric(spot["market_cap"], errors="coerce")
    spot.dropna(subset=["code", "close", "market_cap"], inplace=True)
    return spot[["code", "name", "close", "market_cap"]]


def _export_dataframe(df: pd.DataFrame, destination: str | Path, total: int | None = None) -> Path:
    """Export dataframe to CSV, ensuring parent directories exist."""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    if total is not None:
        print(f"已导出 {total} 条记录到 {output_path}")
    else:
        print(f"已导出 {len(df)} 条记录到 {output_path}")
    return output_path


def list_large_cap_stocks(config: Optional[ScreenConfig] = None) -> pd.DataFrame:
    """列出满足市值下限且剔除 ST 的股票列表。"""

    config = config or get_default_config()
    spot = _fetch_spot_snapshot()
    universe = _filter_base_universe(spot, config)
    if universe.empty:
        return universe
    universe = universe.copy()
    universe.sort_values("market_cap", ascending=False, inplace=True)
    universe["market_cap_亿元"] = universe["market_cap"].astype(float) / 1e8
    universe.reset_index(drop=True, inplace=True)
    return universe


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


def _locate_first_wave(df: pd.DataFrame, config: ScreenConfig) -> Optional[dict]:
    """Locate the most recent 1st wave (≥100% gain) within the lookback window."""

    if df.empty:
        return None

    cutoff_date = df["date"].iloc[-1] - pd.Timedelta(days=config.first_wave_lookback_days)
    window_df = df[df["date"] >= cutoff_date].copy()
    if window_df.empty:
        window_df = df.copy()

    window_df = window_df.dropna(subset=["close", "high"]).copy()
    if window_df.empty:
        return None

    window_df.reset_index(inplace=True)

    min_price = math.inf
    min_idx = int(window_df.loc[0, "index"])
    candidate: Optional[dict] = None

    for _, row in window_df.iterrows():
        idx = int(row["index"])
        close = float(row["close"])
        if not math.isfinite(close) or close <= 0:
            continue
        if close < min_price:
            min_price = close
            min_idx = idx
            continue

        segment = df.loc[min_idx : idx]
        peak_idx = int(segment["high"].idxmax())
        peak_price = float(segment.loc[peak_idx, "high"])
        start_price = float(df.loc[min_idx, "close"])
        gain = (peak_price - start_price) / start_price

        if gain >= config.first_wave_gain:
            candidate = {
                "start_idx": min_idx,
                "start_price": start_price,
                "peak_idx": peak_idx,
                "peak_price": peak_price,
            }

    return candidate


def _check_pullback(df: pd.DataFrame, wave: dict, config: ScreenConfig) -> tuple[bool, Optional[float], Optional[float]]:
    """Verify price retraced the required percentage after the first wave."""

    if not wave:
        return False, None, None

    peak_idx = wave["peak_idx"]
    peak_price = wave["peak_price"]
    subsequent = df.iloc[peak_idx + 1 :]
    if subsequent.empty:
        return False, None, None

    low_after_peak = subsequent["low"].min()
    if pd.isna(low_after_peak) or peak_price == 0:
        return False, None, None

    pullback = (peak_price - low_after_peak) / peak_price
    return pullback >= config.pullback_threshold, float(low_after_peak), float(pullback)


def _check_volatility_contraction(df: pd.DataFrame, config: ScreenConfig) -> tuple[bool, Optional[float]]:
    """Ensure recent realized volatility contracted versus the longer window."""

    if len(df) < config.volatility_window_short + config.volatility_window_long:
        return False, None

    returns = df["close"].pct_change().dropna()

    recent = returns.tail(config.volatility_window_short)
    prior = returns.iloc[-(config.volatility_window_short + config.volatility_window_long) : -config.volatility_window_short]

    if prior.empty or recent.empty:
        return False, None

    recent_std = recent.std(ddof=0)
    prior_std = prior.std(ddof=0)

    if prior_std <= 0 or pd.isna(prior_std) or pd.isna(recent_std):
        return False, None

    ratio = float(recent_std / prior_std)
    if not math.isfinite(ratio):
        return False, None

    return ratio < config.volatility_ratio_threshold, ratio


def _check_trend_persistence(df: pd.DataFrame, config: ScreenConfig) -> tuple[bool, Optional[float]]:
    """Count closes above the 30-day moving average within the specified window."""

    window_df = df.tail(config.trend_window).copy()
    if len(window_df) < 30:
        return False, None

    window_df["ma30"] = window_df["close"].rolling(window=30).mean()
    window_df.dropna(inplace=True)
    if window_df.empty:
        return False, None

    days_above = int((window_df["close"] > window_df["ma30"]).sum())
    ratio_required = config.trend_days_required / config.trend_window
    required = math.ceil(ratio_required * len(window_df))
    ratio_actual = days_above / len(window_df) if len(window_df) else 0.0
    return days_above >= required, ratio_actual


def evaluate_ticker(row: pd.Series, start: dt.date, end: dt.date, config: ScreenConfig) -> ScreenResult:
    """Evaluate a single ticker against all screening rules."""

    code = row["code"]
    name = row["name"]
    market_cap = pd.to_numeric(row.get("market_cap"), errors="coerce")
    spot_close = pd.to_numeric(row.get("close"), errors="coerce")

    metrics: dict[str, object] = {
        "market_cap": float(market_cap) if pd.notna(market_cap) else None,
        "market_cap_亿元": float(market_cap) / 1e8 if pd.notna(market_cap) else None,
        "last_close_spot": float(spot_close) if pd.notna(spot_close) else None,
        "last_close": None,
        "as_of": end,
        "first_wave_gain": None,
        "first_wave_pass": False,
        "first_wave_threshold": config.first_wave_gain,
        "first_wave_start_price": None,
        "first_wave_peak_price": None,
        "first_wave_start_date": None,
        "first_wave_peak_date": None,
        "pullback_ratio": None,
        "pullback_pass": False,
        "pullback_low": None,
        "pullback_threshold": config.pullback_threshold,
        "price_floor": None,
        "price_floor_ratio": None,
        "price_floor_pass": False,
        "price_floor_threshold": config.price_floor_ratio,
        "volatility_ratio": None,
        "volatility_pass": False,
        "volatility_threshold": config.volatility_ratio_threshold,
        "trend_ratio": None,
        "trend_pass": False,
        "trend_required_ratio": config.trend_days_required / config.trend_window,
        "conditions_passed": 0,
    }

    history = _fetch_history(code, start, end)
    if history.empty:
        return ScreenResult(code, name, False, "无历史数据", metrics)

    wave = _locate_first_wave(history, config)
    if not wave:
        return ScreenResult(code, name, False, "未找到第一波上涨", metrics)

    start_idx = wave["start_idx"]
    peak_idx = wave["peak_idx"]
    start_price = wave["start_price"]
    peak_price = wave["peak_price"]

    metrics["first_wave_start_price"] = start_price
    metrics["first_wave_peak_price"] = peak_price
    metrics["first_wave_start_date"] = history.loc[start_idx, "date"].date()
    metrics["first_wave_peak_date"] = history.loc[peak_idx, "date"].date()

    if start_price > 0:
        gain = (peak_price - start_price) / start_price
        metrics["first_wave_gain"] = gain
        metrics["first_wave_pass"] = gain >= config.first_wave_gain
    else:
        metrics["first_wave_gain"] = None
        metrics["first_wave_pass"] = False

    pullback_pass, pullback_low, pullback_ratio = _check_pullback(history, wave, config)
    metrics["pullback_pass"] = pullback_pass
    metrics["pullback_low"] = pullback_low
    metrics["pullback_ratio"] = pullback_ratio

    last_close = float(history["close"].iloc[-1])
    metrics["last_close"] = last_close
    price_floor = start_price * config.price_floor_ratio
    metrics["price_floor"] = price_floor
    if start_price > 0:
        metrics["price_floor_ratio"] = last_close / start_price
    price_floor_pass = last_close >= price_floor
    metrics["price_floor_pass"] = price_floor_pass

    vol_pass, vol_ratio = _check_volatility_contraction(history, config)
    metrics["volatility_pass"] = vol_pass
    metrics["volatility_ratio"] = vol_ratio

    trend_pass, trend_ratio = _check_trend_persistence(history, config)
    metrics["trend_pass"] = trend_pass
    metrics["trend_ratio"] = trend_ratio

    condition_flags = [
        bool(metrics["first_wave_pass"]),
        bool(pullback_pass),
        bool(price_floor_pass),
        bool(vol_pass),
        bool(trend_pass),
    ]
    metrics["conditions_passed"] = sum(condition_flags)

    passed = all(condition_flags)

    reason = None
    if not metrics["first_wave_pass"]:
        reason = "未找到第一波上涨"
    elif not pullback_pass:
        reason = "回调幅度不足"
    elif not price_floor_pass:
        reason = "当前价格低于起涨价保护线"
    elif not vol_pass:
        reason = "波动率未收敛"
    elif not trend_pass:
        reason = "趋势保持不达标"

    return ScreenResult(code, name, passed, reason, metrics)


def run_screen(
    symbols: Optional[Iterable[str]] = None,
    as_of: Optional[dt.date] = None,
    config: Optional[ScreenConfig] = None,
    show_progress: bool = False,
) -> list[ScreenResult]:
    """Run the screen for selected symbols or the entire filtered universe."""

    config = config or get_default_config()
    as_of = as_of or dt.date.today()
    start_date = as_of - dt.timedelta(days=365 * config.history_years)

    spot = _fetch_spot_snapshot()
    base = _filter_base_universe(spot, config)

    if symbols:
        base = base[base["code"].isin(set(symbols))]

    results: list[ScreenResult] = []
    iterator = base.iterrows()
    total = len(base)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, total=total, desc="评估", unit="只")
    elif show_progress:
        print(f"开始评估 {total} 只股票...")
    for idx, (__, row) in enumerate(iterator):
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
                    data_points={"conditions_passed": 0},
                )
            )
        if show_progress and tqdm is None and (idx + 1) % 100 == 0:
            print(f"已评估 {idx + 1}/{total} 只股票", end="")
    if show_progress and tqdm is None:
        print(f"评估完成，共 {total} 只股票".ljust(40))
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
    parser.add_argument(
        "--list-large-cap",
        action="store_true",
        help="仅列出满足市值阈值的股票，不执行完整策略筛选",
    )
    parser.add_argument(
        "--export-csv",
        nargs="?",
        const=str(DEFAULT_EXPORT_FILE),
        help="将结果导出为CSV文件，默认保存到 data/shilei1.csv",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="显示评估进度（需安装 tqdm 才能显示进度条）",
    )
    args = parser.parse_args()

    if args.list_large_cap:
        cfg = get_default_config()
        universe_all = list_large_cap_stocks(cfg)
        total = len(universe_all)
        universe = universe_all.head(args.max) if args.max is not None else universe_all
        if universe.empty:
            print(f"当前暂无市值≥{cfg.market_cap_floor / 1e8:.0f}亿元且非ST的股票")
        else:
            for _, row in universe.iterrows():
                print(f"{row['code']} {row['name']} 市值≈{row['market_cap_亿元']:.2f}亿元")
            if args.max is not None and total > len(universe):
                print(f"...已截断，仅展示前 {len(universe)} 家")
            print(f"合计 {total} 家股票市值≥{cfg.market_cap_floor / 1e8:.0f}亿元")
        if args.export_csv and not universe_all.empty:
            _export_dataframe(universe_all, args.export_csv, total=total)
        return

    as_of = dt.datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None

    results = run_screen(args.symbols, as_of=as_of, show_progress=args.show_progress)

    if args.max is not None:
        results = results[: args.max]

    passes = [res for res in results if res.passed]

    for res in results:
        print(_format_result(res))

    print("-" * 60)
    print(f"Total evaluated: {len(results)} | Passed: {len(passes)}")

    if args.export_csv:
        records: list[dict[str, object]] = []
        for res in results:
            record = {"code": res.code, "name": res.name, "passed": res.passed, "reason": res.reason}
            record.update(res.data_points)
            records.append(record)
        df = pd.DataFrame(records)
        if not df.empty:
            sort_cols = [col for col in ["conditions_passed", "passed"] if col in df.columns]
            if sort_cols:
                df.sort_values(sort_cols, ascending=[False] * len(sort_cols), inplace=True)
            df.reset_index(drop=True, inplace=True)
        _export_dataframe(df, args.export_csv)


if __name__ == "__main__":
    main()

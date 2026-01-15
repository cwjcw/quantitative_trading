from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd

from db import get_conn, init_db, upsert_dataframe
from utils import get_symbols_and_market, load_config


MINUTE_PERIODS = ["5"]


def _format_date(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _rename_with(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rename_map = {k: v for k, v in mapping.items() if k in df.columns}
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _fetch_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="",
    )
    if df.empty:
        return df
    df = _rename_with(
        df,
        {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover",
        },
    )
    df["symbol"] = symbol
    for col in ["amount", "pct_chg", "change", "turnover"]:
        if col not in df.columns:
            df[col] = pd.NA
    cols = [
        "symbol",
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "pct_chg",
        "change",
        "turnover",
    ]
    return df[cols]


def _fetch_minute(symbol: str, start_dt: str, end_dt: str) -> Optional[pd.DataFrame]:
    for period in MINUTE_PERIODS:
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                start_date=start_dt,
                end_date=end_dt,
                adjust="",
            )
        except Exception:
            continue

        if df is None or df.empty:
            continue

        df = _rename_with(
            df,
            {
                "时间": "datetime",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "涨跌幅": "pct_chg",
                "涨跌额": "change",
                "换手率": "turnover",
            },
        )
        df["symbol"] = symbol
        df["period"] = period
        for col in ["pct_chg", "change", "turnover"]:
            if col not in df.columns:
                df[col] = pd.NA
        cols = [
            "symbol",
            "datetime",
            "period",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "pct_chg",
            "change",
            "turnover",
        ]
        return df[cols]
    return None


def _fetch_index_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    if symbol.startswith("3"):
        index_symbol = f"sz{symbol}"
    elif symbol.startswith("0"):
        index_symbol = f"sh{symbol}"
    else:
        index_symbol = symbol
    df = ak.stock_zh_index_daily(symbol=index_symbol)
    if df.empty:
        return df
    start_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end_str = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["date"] >= start_str) & (df["date"] <= end_str)]
    df["symbol"] = symbol
    df["amount"] = pd.NA
    df["pct_chg"] = pd.NA
    df["change"] = pd.NA
    df["turnover"] = pd.NA
    cols = [
        "symbol",
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "pct_chg",
        "change",
        "turnover",
    ]
    return df[cols]


def _fetch_index_minute(symbol: str, start_dt: str, end_dt: str, period: str) -> Optional[pd.DataFrame]:
    try:
        df = ak.index_zh_a_hist_min_em(
            symbol=symbol, period=period, start_date=start_dt, end_date=end_dt
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = _rename_with(
        df,
        {
            "时间": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover",
        },
    )
    df["symbol"] = symbol
    df["period"] = period
    for col in ["amount", "pct_chg", "change", "turnover"]:
        if col not in df.columns:
            df[col] = pd.NA
    cols = [
        "symbol",
        "datetime",
        "period",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "pct_chg",
        "change",
        "turnover",
    ]
    return df[cols]


def _get_trade_dates_between(start_date: str, end_date: str) -> List[str]:
    df = ak.tool_trade_date_hist_sina()
    df = df.dropna()
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    return df["trade_date"].tolist()


def _next_day_yyyymmdd(date_str: str) -> str:
    if "-" in date_str:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        dt = datetime.strptime(date_str, "%Y%m%d")
    return (dt + timedelta(days=1)).strftime("%Y%m%d")


def main() -> None:
    init_db()
    symbols, _ = get_symbols_and_market()
    config = load_config()
    index_symbols = config.get("index_symbols", [])

    today = datetime.now()
    start_daily = _format_date(today - timedelta(days=365))
    end_daily = _format_date(today)

    history_params = config.get("history_params", {})
    minute_start_date = history_params.get("minute_start_date", "2025-11-28")
    minute_end_date = history_params.get("minute_end_date", datetime.now().strftime("%Y-%m-%d"))
    minute_period = str(history_params.get("minute_period", "5"))

    if minute_period not in MINUTE_PERIODS:
        MINUTE_PERIODS.clear()
        MINUTE_PERIODS.append(minute_period)

    trade_dates = _get_trade_dates_between(
        minute_start_date.replace("-", ""), minute_end_date.replace("-", "")
    )
    print(f"分钟周期: {minute_period} | 日期范围: {minute_start_date} -> {minute_end_date}")
    print(f"交易日数量: {len(trade_dates)} | 股票数量: {len(symbols)} | 指数数量: {len(index_symbols)}")

    with get_conn() as conn:
        total_tasks = 0
        done_tasks = 0
        for symbol in symbols:
            max_daily_row = conn.execute(
                "SELECT MAX(date) FROM daily_bars WHERE symbol=?",
                (symbol,),
            ).fetchone()
            if max_daily_row and max_daily_row[0]:
                start_daily_sym = _next_day_yyyymmdd(str(max_daily_row[0]))
            else:
                start_daily_sym = start_daily

            existing_dates = set(
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT substr(datetime,1,10) FROM minute_bars WHERE symbol=?",
                    (symbol,),
                ).fetchall()
                if row[0]
            )
            if start_daily_sym <= end_daily:
                daily_df = _fetch_daily(symbol, start_daily_sym, end_daily)
                upsert_dataframe(conn, "daily_bars", daily_df)

            missing_dates = [
                d for d in trade_dates
                if f"{d[:4]}-{d[4:6]}-{d[6:]}" not in existing_dates
            ]
            total_tasks += len(missing_dates)
            for trade_date in missing_dates:
                date_fmt = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
                start_min = f"{date_fmt} 09:30:00"
                end_min = f"{date_fmt} 15:00:00"
                minute_df = _fetch_minute(symbol, start_min, end_min)
                if minute_df is not None and not minute_df.empty:
                    upsert_dataframe(conn, "minute_bars", minute_df)
                done_tasks += 1
                if done_tasks % 10 == 0 or done_tasks == total_tasks:
                    print(f"进度: {done_tasks}/{total_tasks} | {symbol}")

        for symbol in index_symbols:
            max_daily_row = conn.execute(
                "SELECT MAX(date) FROM daily_bars WHERE symbol=?",
                (symbol,),
            ).fetchone()
            if max_daily_row and max_daily_row[0]:
                start_daily_sym = _next_day_yyyymmdd(str(max_daily_row[0]))
            else:
                start_daily_sym = start_daily
            if start_daily_sym <= end_daily:
                daily_df = _fetch_index_daily(symbol, start_daily_sym, end_daily)
                upsert_dataframe(conn, "daily_bars", daily_df)
            existing_dates = set(
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT substr(datetime,1,10) FROM minute_bars WHERE symbol=?",
                    (symbol,),
                ).fetchall()
                if row[0]
            )
            missing_dates = [
                d for d in trade_dates
                if f"{d[:4]}-{d[4:6]}-{d[6:]}" not in existing_dates
            ]
            for trade_date in missing_dates:
                date_fmt = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
                start_min = f"{date_fmt} 09:30:00"
                end_min = f"{date_fmt} 15:00:00"
                minute_df = _fetch_index_minute(symbol, start_min, end_min, minute_period)
                if minute_df is not None and not minute_df.empty:
                    upsert_dataframe(conn, "minute_bars", minute_df)

    print("历史数据写入完成。")


if __name__ == "__main__":
    main()

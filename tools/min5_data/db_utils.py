from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import akshare as ak
import pandas as pd
import sqlite3

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "min5_data.db"
CODE_CACHE = DATA_DIR / "stock_codes.txt"


def get_trade_dates(count: int) -> List[str]:
    df = ak.tool_trade_date_hist_sina()
    df = df.dropna()
    df["trade_date"] = df["trade_date"].astype(str)
    today = datetime.now().strftime("%Y%m%d")
    df = df[df["trade_date"] <= today]
    return df["trade_date"].tail(count).tolist()


def get_stock_codes(use_cache: bool = True) -> List[str]:
    if use_cache and CODE_CACHE.exists():
        return [line.strip() for line in CODE_CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]

    df = ak.stock_zh_a_spot_em()
    codes = df["代码"].astype(str).tolist()
    CODE_CACHE.write_text("\n".join(codes), encoding="utf-8")
    return codes


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS min5_bars (
            symbol TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            amount REAL,
            PRIMARY KEY (symbol, datetime)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_min5_datetime ON min5_bars(datetime)"
    )
    return conn


def fetch_min5(symbol: str, start_dt: str, end_dt: str) -> pd.DataFrame:
    return ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        period="5",
        start_date=start_dt,
        end_date=end_dt,
        adjust="",
    )


def normalize_min5(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = df.rename(
        columns={
            "时间": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
    )
    df["symbol"] = symbol
    cols = ["symbol", "datetime", "open", "close", "high", "low", "volume", "amount"]
    df = df[cols]
    df["datetime"] = df["datetime"].astype(str)
    return df


def upsert_min5(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = df.to_records(index=False)
    conn.executemany(
        """
        INSERT OR REPLACE INTO min5_bars
        (symbol, datetime, open, close, high, low, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows.tolist(),
    )
    conn.commit()
    return len(df)

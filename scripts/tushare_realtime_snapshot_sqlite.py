from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts

from quantitative_trading.config import get_settings


DEFAULT_CODES = "300806.SZ,002938.SZ"
DEFAULT_DB = "tmp/tushare_realtime_snapshots.sqlite3"
LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Tushare realtime_quote snapshots into a local SQLite database."
    )
    parser.add_argument("--codes", default=DEFAULT_CODES, help="Comma-separated TS codes.")
    parser.add_argument("--all-stock", action="store_true", help="Fetch all listed A-share stocks.")
    parser.add_argument("--batch-size", type=int, default=800, help="Codes per realtime_quote request.")
    parser.add_argument("--src", default="sina", choices=["sina", "dc"], help="Tushare realtime source.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path.")
    parser.add_argument("--once", action="store_true", help="Fetch one snapshot batch and exit.")
    parser.add_argument("--loop", action="store_true", help="Fetch snapshots repeatedly.")
    parser.add_argument("--start-at", default="", help="Local start time HH:MM, e.g. 13:30.")
    parser.add_argument("--end-at", default="15:05", help="Local end time HH:MM for loop mode.")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval seconds.")
    return parser.parse_args()


def load_codes(all_stock: bool, codes: str) -> list[str]:
    if not all_stock:
        return [code.strip() for code in codes.split(",") if code.strip()]
    settings = get_settings()
    pro = ts.pro_api(settings.tushare_token)
    frame = pro.stock_basic(exchange="", list_status="L", fields="ts_code")
    return frame["ts_code"].dropna().astype(str).sort_values().tolist()


def connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma busy_timeout=5000")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists realtime_snapshots (
            id integer primary key autoincrement,
            captured_at text not null,
            quote_date text,
            quote_time text,
            ts_code text not null,
            name text,
            open_price real,
            pre_close real,
            last_price real,
            high_price real,
            low_price real,
            bid real,
            ask real,
            volume_shares integer,
            volume_lots real,
            amount real,
            change_amount real,
            change_percent real,
            avg_price real,
            amplitude_percent real,
            bid_price_json text,
            ask_price_json text,
            bid_volume_json text,
            ask_volume_json text,
            raw_json text not null,
            unique(captured_at, ts_code)
        )
        """
    )
    conn.execute(
        "create index if not exists idx_realtime_snapshots_code_time "
        "on realtime_snapshots(ts_code, quote_date, quote_time, captured_at)"
    )
    conn.commit()


def num(value):
    if value is None:
        return None
    try:
        if value != value:
            return None
        return float(value)
    except Exception:
        return None


def calc_change(last_price: float | None, pre_close: float | None) -> tuple[float | None, float | None]:
    if last_price is None or pre_close in (None, 0):
        return None, None
    change = last_price - pre_close
    return change, change / pre_close * 100


def calc_avg(amount: float | None, volume_shares: int | None) -> float | None:
    if amount is None or not volume_shares:
        return None
    return amount / volume_shares


def calc_amp(high_price: float | None, low_price: float | None, pre_close: float | None) -> float | None:
    if high_price is None or low_price is None or pre_close in (None, 0):
        return None
    return (high_price - low_price) / pre_close * 100


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


def row_payload(row, captured_at: datetime) -> dict:
    raw = row.to_dict()
    last_price = num(raw.get("PRICE"))
    pre_close = num(raw.get("PRE_CLOSE"))
    high_price = num(raw.get("HIGH"))
    low_price = num(raw.get("LOW"))
    amount = num(raw.get("AMOUNT"))
    volume_shares = int(raw["VOLUME"]) if raw.get("VOLUME") is not None else None
    change_amount, change_percent = calc_change(last_price, pre_close)

    bid_prices = [num(raw.get(f"B{i}_P")) for i in range(1, 6)]
    ask_prices = [num(raw.get(f"A{i}_P")) for i in range(1, 6)]
    bid_volumes = [int(raw.get(f"B{i}_V") or 0) for i in range(1, 6)]
    ask_volumes = [int(raw.get(f"A{i}_V") or 0) for i in range(1, 6)]

    return {
        "captured_at": captured_at.isoformat(timespec="seconds"),
        "quote_date": raw.get("DATE"),
        "quote_time": raw.get("TIME"),
        "ts_code": raw.get("TS_CODE"),
        "name": raw.get("NAME"),
        "open_price": num(raw.get("OPEN")),
        "pre_close": pre_close,
        "last_price": last_price,
        "high_price": high_price,
        "low_price": low_price,
        "bid": num(raw.get("BID")),
        "ask": num(raw.get("ASK")),
        "volume_shares": volume_shares,
        "volume_lots": volume_shares / 100 if volume_shares is not None else None,
        "amount": amount,
        "change_amount": change_amount,
        "change_percent": change_percent,
        "avg_price": calc_avg(amount, volume_shares),
        "amplitude_percent": calc_amp(high_price, low_price, pre_close),
        "bid_price_json": json.dumps(bid_prices, ensure_ascii=False),
        "ask_price_json": json.dumps(ask_prices, ensure_ascii=False),
        "bid_volume_json": json.dumps(bid_volumes, ensure_ascii=False),
        "ask_volume_json": json.dumps(ask_volumes, ensure_ascii=False),
        "raw_json": json.dumps(raw, ensure_ascii=False, default=str),
    }


def insert_payloads(conn: sqlite3.Connection, payloads: list[dict]) -> None:
    if not payloads:
        return
    columns = list(payloads[0].keys())
    placeholders = ", ".join(f":{column}" for column in columns)
    conn.executemany(
        f"""
        insert or ignore into realtime_snapshots ({", ".join(columns)})
        values ({placeholders})
        """,
        payloads,
    )
    conn.commit()


def chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_and_store(conn: sqlite3.Connection, codes: list[str], batch_size: int, src: str) -> int:
    captured_at = datetime.now(LOCAL_TZ)
    frames = []
    for batch in chunks(codes, batch_size):
        frame = ts.realtime_quote(ts_code=",".join(batch), src=src)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        print(f"{captured_at.isoformat(timespec='seconds')} no rows returned")
        return 0
    frame = pd.concat(frames, ignore_index=True)
    payloads = [row_payload(row, captured_at) for _, row in frame.iterrows()]
    insert_payloads(conn, payloads)
    for item in payloads[:20]:
        print(
            f"{item['captured_at']} {item['ts_code']} {item['name']} "
            f"quote={item['quote_date']} {item['quote_time']} "
            f"last={item['last_price']} pct={fmt(item['change_percent'])}% "
            f"avg={fmt(item['avg_price'])} vol_lots={fmt(item['volume_lots'], 0)}"
        )
    if len(payloads) > 20:
        print(f"{captured_at.isoformat(timespec='seconds')} stored {len(payloads)} rows")
    return len(payloads)


def parse_local_time(value: str) -> dt_time:
    hour, minute = value.split(":", 1)
    return dt_time(int(hour), int(minute), tzinfo=LOCAL_TZ)


def next_local_datetime(value: str) -> datetime:
    target_time = parse_local_time(value)
    now = datetime.now(LOCAL_TZ)
    target = datetime.combine(now.date(), target_time)
    if target < now:
        target += timedelta(days=1)
    return target


def sleep_until(start_at: str) -> None:
    if not start_at:
        return
    target = next_local_datetime(start_at)
    while True:
        now = datetime.now(LOCAL_TZ)
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30))


def loop(
    conn: sqlite3.Connection,
    codes: list[str],
    batch_size: int,
    src: str,
    start_at: str,
    end_at: str,
    interval: int,
) -> None:
    sleep_until(start_at)
    end_time = parse_local_time(end_at)
    while True:
        now = datetime.now(LOCAL_TZ)
        if now.timetz() > end_time:
            print(f"{now.isoformat(timespec='seconds')} reached end-at {end_at}, exiting.")
            return
        try:
            started = time.perf_counter()
            row_count = fetch_and_store(conn, codes, batch_size, src)
            elapsed = time.perf_counter() - started
            print(f"{datetime.now(LOCAL_TZ).isoformat(timespec='seconds')} batch done rows={row_count} elapsed={elapsed:.2f}s")
        except Exception as exc:
            print(f"{now.isoformat(timespec='seconds')} fetch failed: {exc!r}")
        time.sleep(interval)


def main() -> None:
    args = parse_args()
    get_settings()
    codes = load_codes(args.all_stock, args.codes)
    print(f"loaded {len(codes)} codes")
    conn = connect(args.db)
    if args.loop:
        loop(conn, codes, args.batch_size, args.src, args.start_at, args.end_at, args.interval)
        return
    started = time.perf_counter()
    row_count = fetch_and_store(conn, codes, args.batch_size, args.src)
    elapsed = time.perf_counter() - started
    print(f"done rows={row_count} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()

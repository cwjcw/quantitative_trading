from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from sqlalchemy import text

from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine


DEFAULT_SQLITE_DB = "tmp/tushare_realtime_all_snapshots.sqlite3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Tushare realtime SQLite snapshots with QMT snapshots.")
    parser.add_argument("--sqlite-db", default=DEFAULT_SQLITE_DB)
    parser.add_argument("--trade-date", default="2026-06-09")
    parser.add_argument("--quote-time", default="", help="Quote time like 11:30:00. Empty means latest per code.")
    parser.add_argument("--limit-samples", type=int, default=20)
    return parser.parse_args()


def load_tushare_rows(db_path: str, trade_date: str, quote_time: str) -> list[sqlite3.Row]:
    if not Path(db_path).exists():
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    quote_date = trade_date.replace("-", "")
    if quote_time:
        return list(
            conn.execute(
                """
                select *
                from realtime_snapshots
                where quote_date = ? and quote_time = ?
                order by ts_code, captured_at desc
                """,
                (quote_date, quote_time),
            )
        )
    return list(
        conn.execute(
            """
            with ranked as (
                select *,
                       row_number() over (
                           partition by ts_code
                           order by quote_date desc, quote_time desc, captured_at desc, id desc
                       ) as rn
                from realtime_snapshots
                where quote_date = ?
            )
            select * from ranked where rn = 1 order by ts_code
            """,
            (quote_date,),
        )
    )


def load_qmt_rows(trade_date: str, quote_time: str, codes: list[str]) -> dict[str, dict]:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    params = {"trade_date": trade_date, "codes": codes}
    time_filter = ""
    if quote_time:
        time_filter = "and raw->>'timetag' = replace(cast(:trade_date as text), '-', '') || ' ' || :quote_time"
        params["quote_time"] = quote_time
    query = text(
        f"""
        with ranked as (
            select *,
                   row_number() over (
                       partition by stock_code
                       order by raw_time desc nulls last, captured_at desc
                   ) as rn
            from public.stock_snapshots
            where trade_date = cast(:trade_date as date)
              and stock_code = any(:codes)
              {time_filter}
        )
        select stock_code, instrument_name, raw->>'timetag' as timetag, captured_at,
               open_price::float open_price, last_close::float pre_close,
               last_price::float last_price, high_price::float high_price, low_price::float low_price,
               volume::float volume_lots, pvolume::float volume_shares, amount::float amount,
               bid_price, ask_price, bid_volume, ask_volume
        from ranked
        where rn = 1
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()
    return {row["stock_code"]: dict(row) for row in rows}


def f(row, key):
    value = row[key]
    if value is None:
        return None
    return float(value)


def diff(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def max_abs_price_diff(ts_row, qmt_row) -> float:
    diffs = []
    pairs = [
        ("open_price", "open_price"),
        ("pre_close", "pre_close"),
        ("last_price", "last_price"),
        ("high_price", "high_price"),
        ("low_price", "low_price"),
        ("bid", None),
        ("ask", None),
    ]
    for ts_key, qmt_key in pairs:
        if qmt_key is None:
            continue
        value = diff(f(ts_row, ts_key), qmt_row[qmt_key])
        if value is not None:
            diffs.append(abs(value))
    return max(diffs) if diffs else 0.0


def parse_json(value):
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def list_close(left, right, tolerance: float = 1e-6) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    for left_value, right_value in zip(left, right):
        if left_value is None and right_value is None:
            continue
        if left_value is None or right_value is None:
            return False
        if abs(float(left_value) - float(right_value)) > tolerance:
            return False
    return True


def main() -> None:
    args = parse_args()
    ts_rows = load_tushare_rows(args.sqlite_db, args.trade_date, args.quote_time)
    codes = [row["ts_code"] for row in ts_rows]
    qmt = load_qmt_rows(args.trade_date, args.quote_time, codes)

    matched = []
    missing_qmt = []
    price_mismatches = []
    volume_mismatches = []
    amount_mismatches = []
    book_mismatches = []

    for row in ts_rows:
        code = row["ts_code"]
        q = qmt.get(code)
        if not q:
            missing_qmt.append(code)
            continue
        matched.append(code)

        price_abs = max_abs_price_diff(row, q)
        if price_abs > 0.001:
            price_mismatches.append((code, row["name"], row["quote_time"], q["timetag"], row["last_price"], q["last_price"], price_abs))

        vol_diff = diff(row["volume_shares"], q["volume_shares"])
        if vol_diff is not None and abs(vol_diff) > 1:
            volume_mismatches.append((code, row["name"], row["volume_shares"], q["volume_shares"], vol_diff))

        amt_diff = diff(row["amount"], q["amount"])
        if amt_diff is not None and abs(amt_diff) > 100:
            amount_mismatches.append((code, row["name"], row["amount"], q["amount"], amt_diff))

        ts_bid_prices = parse_json(row["bid_price_json"])
        ts_ask_prices = parse_json(row["ask_price_json"])
        if not list_close(ts_bid_prices, q["bid_price"]) or not list_close(ts_ask_prices, q["ask_price"]):
            book_mismatches.append((code, row["name"], ts_bid_prices, q["bid_price"], ts_ask_prices, q["ask_price"]))

    print("SUMMARY")
    print(f"tushare_rows={len(ts_rows)}")
    print(f"qmt_matched={len(matched)}")
    print(f"missing_qmt={len(missing_qmt)}")
    print(f"price_mismatches={len(price_mismatches)}")
    print(f"volume_mismatches={len(volume_mismatches)}")
    print(f"amount_mismatches={len(amount_mismatches)}")
    print(f"book_mismatches={len(book_mismatches)}")

    def show(title, values):
        print(f"\n{title}")
        if not values:
            print("none")
            return
        for item in values[: args.limit_samples]:
            print(item)

    show("missing_qmt samples", missing_qmt)
    show("price_mismatch samples", price_mismatches)
    show("volume_mismatch samples", volume_mismatches)
    show("amount_mismatch samples", amount_mismatches)
    show("book_mismatch samples", book_mismatches)


if __name__ == "__main__":
    main()

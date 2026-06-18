from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantitative_trading.config import build_database_url_from_parts, load_env_file
from scripts.tushare_fund_backfill import DDL, insert_frame


def fetch_fund_portfolio(pro, period: str, limit: int, offset: int, max_retries: int, retry_sleep: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            frame = pro.query("fund_portfolio", period=period, limit=limit, offset=offset)
            if frame is None:
                return pd.DataFrame()
            return frame
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait = retry_sleep * (attempt + 1)
            print(
                f"RETRY period={period} offset={offset} attempt={attempt + 1}/{max_retries} "
                f"sleep={wait:.1f}s error={exc}",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"fund_portfolio failed period={period} offset={offset}: {last_error}") from last_error


def period_count(engine, period: str) -> int:
    query = text(
        """
        SELECT count(*)
        FROM public.tushare_fund_portfolio
        WHERE end_date = to_date(:period, 'YYYYMMDD')
        """
    )
    with engine.connect() as conn:
        return int(conn.execute(query, {"period": period}).scalar_one())


def backfill_periods(periods: list[str], limit: int, sleep_seconds: float, max_retries: int, retry_sleep: float) -> None:
    load_env_file()
    token = (os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is missing.")

    ts.set_token(token)
    pro = ts.pro_api()
    url = os.getenv("DATABASE_URL") or os.getenv("SMART_STOCK_DATABASE_URL") or build_database_url_from_parts()
    engine = create_engine(url, pool_pre_ping=True, future=True)
    with engine.begin() as conn:
        conn.execute(text(DDL))

    for period in periods:
        existing = period_count(engine, period)
        if existing and existing % limit != 0:
            print(f"PERIOD_SKIP {period} existing_complete={existing}", flush=True)
            continue

        offset = existing
        total = existing
        print(f"PERIOD_START {period} existing={existing} offset={offset}", flush=True)
        while True:
            frame = fetch_fund_portfolio(pro, period, limit, offset, max_retries, retry_sleep)
            if frame is None or frame.empty:
                break
            inserted = insert_frame(engine, "fund_portfolio", frame)
            total += int(inserted)
            print(
                f"fund_portfolio period={period} offset={offset} "
                f"fetched={len(frame)} inserted={inserted} total={total}",
                flush=True,
            )
            if len(frame) < limit:
                break
            offset += limit
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        print(f"PERIOD_DONE {period} total={total}", flush=True)

    print("ALL_DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Tushare fund_portfolio for report periods.")
    parser.add_argument("periods", nargs="+", help="Report periods like 20200331.")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch periods from offset 0 even when rows already exist. Inserts are idempotent.",
    )
    args = parser.parse_args()
    if args.force:
        load_env_file()
        token = (os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN") or "").strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is missing.")
        ts.set_token(token)
        pro = ts.pro_api()
        url = os.getenv("DATABASE_URL") or os.getenv("SMART_STOCK_DATABASE_URL") or build_database_url_from_parts()
        engine = create_engine(url, pool_pre_ping=True, future=True)
        with engine.begin() as conn:
            conn.execute(text(DDL))
        for period in args.periods:
            print(f"PERIOD_START {period} force=true offset=0", flush=True)
            total = 0
            offset = 0
            while True:
                frame = fetch_fund_portfolio(
                    pro,
                    period,
                    args.limit,
                    offset,
                    args.max_retries,
                    args.retry_sleep,
                )
                if frame is None or frame.empty:
                    break
                inserted = insert_frame(engine, "fund_portfolio", frame)
                total += int(inserted)
                print(
                    f"fund_portfolio period={period} offset={offset} "
                    f"fetched={len(frame)} inserted={inserted} total={total}",
                    flush=True,
                )
                if len(frame) < args.limit:
                    break
                offset += args.limit
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            print(f"PERIOD_DONE {period} total={total}", flush=True)
        print("ALL_DONE", flush=True)
    else:
        backfill_periods(args.periods, args.limit, args.sleep_seconds, args.max_retries, args.retry_sleep)


if __name__ == "__main__":
    main()

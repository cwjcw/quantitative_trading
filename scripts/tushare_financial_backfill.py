from __future__ import annotations

import argparse
import time
from datetime import datetime

import tushare as ts

from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine
from quantitative_trading.tushare.collect_financials import collect_financials, load_codes
from quantitative_trading.tushare.collect_raw import parse_yyyymmdd
from quantitative_trading.tushare.schema import ensure_schema


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-market Tushare financial backfill in stock-code batches.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD format.")
    parser.add_argument("--batch-size", type=int, default=200, help="Stock codes per batch.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N stock codes before batching.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches in this run.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Sleep between API requests.")
    parser.add_argument("--batch-pause-seconds", type=float, default=5.0, help="Sleep between batches.")
    parser.add_argument("--include-delisted", action="store_true", help="Include D/P stocks when local codes are unavailable.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed checkpoints.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    engine = make_engine(settings.database_url)
    ensure_schema(engine)

    ts.set_token(settings.tushare_token)
    pro = ts.pro_api()
    codes = load_codes(engine, pro, args.include_delisted)[args.offset :]
    batches = chunks(codes, args.batch_size)
    if args.max_batches is not None:
        batches = batches[: args.max_batches]

    start_date = parse_yyyymmdd(args.start_date)
    end_date = parse_yyyymmdd(args.end_date)
    print(
        "FINANCIAL_BACKFILL "
        f"codes={len(codes)} batches={len(batches)} batch_size={args.batch_size} "
        f"start={args.start_date} end={args.end_date}"
    )

    for index, batch_codes in enumerate(batches, start=1):
        first = batch_codes[0]
        last = batch_codes[-1]
        print(f"BATCH {index}/{len(batches)} size={len(batch_codes)} first={first} last={last} at={datetime.now().isoformat()}")
        collect_financials(
            start_date=start_date,
            end_date=end_date,
            endpoint_names=None,
            ts_codes=set(batch_codes),
            all_codes=False,
            include_delisted=args.include_delisted,
            offset=0,
            max_codes=None,
            sleep_seconds=args.sleep_seconds,
            retry_failed=args.retry_failed,
        )
        if args.batch_pause_seconds > 0 and index < len(batches):
            time.sleep(args.batch_pause_seconds)


if __name__ == "__main__":
    main()

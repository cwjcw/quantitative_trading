from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import tushare as ts
from sqlalchemy import text
from sqlalchemy.engine import Engine

from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine
from quantitative_trading.tushare.collect_raw import (
    insert_rows,
    load_local_stock_codes,
    load_tushare_stock_codes,
    parse_yyyymmdd,
    to_tushare_date,
)
from quantitative_trading.tushare.endpoints import TushareEndpoint, select_endpoints
from quantitative_trading.tushare.schema import ensure_schema


FINANCIAL_ENDPOINTS = {
    "fina_indicator",
    "income",
    "balancesheet",
    "cashflow",
    "fina_mainbz",
}


def checkpoint_key(endpoint: TushareEndpoint, ts_code: str, start_date: date, end_date: date) -> str:
    return "|".join([endpoint.name, endpoint.content_type, ts_code, to_tushare_date(start_date), to_tushare_date(end_date)])


def checkpoint_status(engine: Engine, key: str) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT status FROM tushare_collection_checkpoints WHERE checkpoint_key = :key"),
            {"key": key},
        ).scalar_one_or_none()


def update_checkpoint(
    engine: Engine,
    endpoint: TushareEndpoint,
    ts_code: str,
    start_date: date,
    end_date: date,
    status: str,
    row_count: int,
    error_message: str | None = None,
) -> None:
    key = checkpoint_key(endpoint, ts_code, start_date, end_date)
    stmt = text(
        """
        INSERT INTO tushare_collection_checkpoints (
            checkpoint_key, endpoint, ts_code, content_type, range_start, range_end,
            status, attempts, row_count, error_message, updated_at
        )
        VALUES (
            :checkpoint_key, :endpoint, :ts_code, :content_type, :range_start, :range_end,
            :status, 1, :row_count, :error_message, :updated_at
        )
        ON CONFLICT (checkpoint_key)
        DO UPDATE SET
            status = EXCLUDED.status,
            attempts = tushare_collection_checkpoints.attempts + 1,
            row_count = EXCLUDED.row_count,
            error_message = EXCLUDED.error_message,
            updated_at = EXCLUDED.updated_at
        """
    )
    with engine.begin() as conn:
        conn.execute(
            stmt,
            {
                "checkpoint_key": key,
                "endpoint": endpoint.name,
                "ts_code": ts_code,
                "content_type": endpoint.content_type,
                "range_start": start_date,
                "range_end": end_date,
                "status": status,
                "row_count": row_count,
                "error_message": error_message,
                "updated_at": datetime.now(UTC),
            },
        )


def start_run(engine: Engine, endpoints: list[TushareEndpoint], start_date: date, end_date: date, codes: list[str]) -> uuid.UUID:
    run_id = uuid.uuid4()
    stmt = text(
        """
        INSERT INTO tushare_raw_runs (
            run_id, started_at, status, start_date, end_date, endpoints, trade_dates, note
        )
        VALUES (
            :run_id, :started_at, 'running', :start_date, :end_date,
            CAST(:endpoints AS jsonb), CAST(:trade_dates AS jsonb), :note
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": run_id,
                "started_at": datetime.now(UTC),
                "start_date": start_date,
                "end_date": end_date,
                "endpoints": json.dumps([asdict(item) for item in endpoints], ensure_ascii=True),
                "trade_dates": json.dumps([to_tushare_date(start_date), to_tushare_date(end_date)], ensure_ascii=True),
                "note": f"financial batch codes={len(codes)}",
            },
        )
    return run_id


def finish_run(
    engine: Engine,
    run_id: uuid.UUID,
    status: str,
    row_count: int,
    error_count: int,
    elapsed_seconds: float,
    note: str,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE tushare_raw_runs
                SET finished_at = :finished_at,
                    status = :status,
                    row_count = :row_count,
                    error_count = :error_count,
                    elapsed_seconds = :elapsed_seconds,
                    note = :note
                WHERE run_id = :run_id
                """
            ),
            {
                "run_id": run_id,
                "finished_at": datetime.now(UTC),
                "status": status,
                "row_count": row_count,
                "error_count": error_count,
                "elapsed_seconds": elapsed_seconds,
                "note": note,
            },
        )


def fetch_financial_endpoint(
    pro: Any,
    endpoint: TushareEndpoint,
    ts_code: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    params: dict[str, str] = {"ts_code": ts_code, **endpoint.default_params}
    if endpoint.name == "fina_mainbz":
        frames = []
        for period in quarter_periods(start_date, end_date):
            frame = pro.query(endpoint.name, **params, period=period)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    return pro.query(
        endpoint.name,
        **params,
        start_date=to_tushare_date(start_date),
        end_date=to_tushare_date(end_date),
    )


def quarter_periods(start_date: date, end_date: date) -> list[str]:
    periods: list[str] = []
    for year in range(start_date.year, end_date.year + 1):
        for month_day in ["0331", "0630", "0930", "1231"]:
            value = f"{year}{month_day}"
            parsed = datetime.strptime(value, "%Y%m%d").date()
            if start_date <= parsed <= end_date:
                periods.append(value)
    return periods


def load_codes(engine: Engine, pro: Any, include_delisted: bool) -> list[str]:
    codes = load_local_stock_codes(engine)
    if codes:
        return codes

    if include_delisted:
        return load_tushare_stock_codes(pro)

    frame = pro.query("stock_basic", exchange="", list_status="L")
    if frame.empty:
        return []
    return sorted(set(str(value) for value in frame["ts_code"].dropna().tolist()))


def collect_financials(
    start_date: date,
    end_date: date,
    endpoint_names: set[str] | None,
    ts_codes: set[str] | None,
    all_codes: bool,
    include_delisted: bool,
    offset: int,
    max_codes: int | None,
    sleep_seconds: float,
    retry_failed: bool,
) -> None:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    ensure_schema(engine)

    endpoints = [
        item
        for item in select_endpoints({"P2"}, endpoint_names or FINANCIAL_ENDPOINTS)
        if item.name in FINANCIAL_ENDPOINTS
    ]
    if not endpoints:
        raise RuntimeError("No financial endpoints selected.")

    ts.set_token(settings.tushare_token)
    pro = ts.pro_api()

    if ts_codes:
        codes = sorted(ts_codes)
    elif all_codes:
        codes = load_codes(engine, pro, include_delisted)
    else:
        raise RuntimeError("Financial collection requires --ts-code or --all-codes.")

    codes = codes[offset:]
    if max_codes is not None:
        codes = codes[:max_codes]
    if not codes:
        raise RuntimeError("No stock codes selected.")

    run_id = start_run(engine, endpoints, start_date, end_date, codes)
    started = time.monotonic()
    inserted_total = 0
    errors: list[str] = []

    for ts_code in codes:
        for endpoint in endpoints:
            key = checkpoint_key(endpoint, ts_code, start_date, end_date)
            previous_status = checkpoint_status(engine, key)
            if previous_status == "success" or (previous_status == "failed" and not retry_failed):
                print(f"SKIP {endpoint.name} {ts_code}: checkpoint={previous_status}")
                continue

            try:
                frame = fetch_financial_endpoint(pro, endpoint, ts_code, start_date, end_date)
                inserted = insert_rows(
                    engine,
                    endpoint,
                    frame,
                    end_date,
                    replace_ts_code=ts_code,
                    replace_start_date=start_date,
                    replace_end_date=end_date,
                )
                inserted_total += inserted
                update_checkpoint(engine, endpoint, ts_code, start_date, end_date, "success", inserted)
                print(f"{endpoint.name} {ts_code}: fetched={len(frame)} inserted={inserted}")
            except Exception as exc:
                message = f"{endpoint.name} {ts_code}: {exc}"
                errors.append(message)
                update_checkpoint(engine, endpoint, ts_code, start_date, end_date, "failed", 0, str(exc))
                print(f"ERROR {message}")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    elapsed = time.monotonic() - started
    status = "success" if not errors else "partial_success"
    finish_run(
        engine,
        run_id,
        status=status,
        row_count=inserted_total,
        error_count=len(errors),
        elapsed_seconds=elapsed,
        note="\n".join(errors[:100]),
    )
    print(f"run_id={run_id} status={status} inserted={inserted_total} errors={len(errors)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect stock-scoped Tushare financial statements into PostgreSQL.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD format.")
    parser.add_argument("--endpoint", action="append", default=None, help="Financial endpoint. Can be repeated.")
    parser.add_argument("--ts-code", action="append", default=None, help="Specific Tushare stock code. Can be repeated.")
    parser.add_argument("--all-codes", action="store_true", help="Collect for all local stock codes.")
    parser.add_argument("--include-delisted", action="store_true", help="When local codes are unavailable, include D/P stocks.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N selected stock codes.")
    parser.add_argument("--max-codes", type=int, default=None, help="Limit this run to N selected stock codes.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Sleep between API requests.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry checkpoints with failed status.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    collect_financials(
        start_date=parse_yyyymmdd(args.start_date),
        end_date=parse_yyyymmdd(args.end_date),
        endpoint_names=set(args.endpoint) if args.endpoint else None,
        ts_codes=set(args.ts_code) if args.ts_code else None,
        all_codes=args.all_codes,
        include_delisted=args.include_delisted,
        offset=args.offset,
        max_codes=args.max_codes,
        sleep_seconds=args.sleep_seconds,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()

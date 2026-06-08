from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import tushare as ts
from sqlalchemy import text
from sqlalchemy.engine import Engine

from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine
from quantitative_trading.tushare.endpoints import TushareEndpoint, select_endpoints
from quantitative_trading.tushare.schema import ensure_schema


def parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def date_range(start_date: date, end_date: date) -> list[date]:
    return [item.date() for item in pd.date_range(start=start_date, end=end_date, freq="D")]


def to_tushare_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {key: normalize_value(value) for key, value in row.to_dict().items()}


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def date_key_for(endpoint: TushareEndpoint, row: dict[str, Any], fallback: date | None) -> tuple[str, str]:
    if endpoint.date_field and row.get(endpoint.date_field):
        return str(row[endpoint.date_field]), endpoint.date_field
    if endpoint.date_param == "month" and fallback:
        return fallback.strftime("%Y%m"), "month"
    if endpoint.date_param and fallback:
        return to_tushare_date(fallback), endpoint.date_param
    return "", ""


def code_for(endpoint: TushareEndpoint, row: dict[str, Any]) -> str:
    if endpoint.code_field and row.get(endpoint.code_field):
        return str(row[endpoint.code_field])
    return ""


def fetch_endpoint(pro: Any, endpoint: TushareEndpoint, run_date: date) -> pd.DataFrame:
    params = dict(endpoint.default_params)
    if endpoint.date_param == "date_range":
        params["start_date"] = to_tushare_date(run_date)
        params["end_date"] = to_tushare_date(run_date)
    elif endpoint.date_param == "month":
        params["month"] = run_date.strftime("%Y%m")
    elif endpoint.date_param:
        params[endpoint.date_param] = to_tushare_date(run_date)
    return pro.query(endpoint.name, **params)


def fetch_endpoint_for_code(
    pro: Any,
    endpoint: TushareEndpoint,
    run_date: date,
    ts_code: str,
) -> pd.DataFrame:
    params = dict(endpoint.default_params)
    params["ts_code"] = ts_code
    if endpoint.date_param == "date_range":
        params["start_date"] = to_tushare_date(run_date)
        params["end_date"] = to_tushare_date(run_date)
    elif endpoint.date_param == "month":
        params["month"] = run_date.strftime("%Y%m")
    elif endpoint.date_param:
        params[endpoint.date_param] = to_tushare_date(run_date)
    return pro.query(endpoint.name, **params)


def to_ts_code(stock_code: str, exchange: str | None = None) -> str:
    value = stock_code.strip().upper()
    if "." in value:
        left, right = value.split(".", 1)
        if right in {"SH", "SZ", "BJ"}:
            return value
        if right in {"SSE", "SHSE"}:
            return f"{left}.SH"
        if right in {"SZSE"}:
            return f"{left}.SZ"
        if right in {"BSE"}:
            return f"{left}.BJ"

    exch = (exchange or "").strip().upper()
    if exch in {"SH", "SSE", "SHSE"}:
        return f"{value}.SH"
    if exch in {"SZ", "SZSE"}:
        return f"{value}.SZ"
    if exch in {"BJ", "BSE"}:
        return f"{value}.BJ"
    if value.startswith(("6", "9")):
        return f"{value}.SH"
    if value.startswith(("0", "2", "3")):
        return f"{value}.SZ"
    if value.startswith(("4", "8")):
        return f"{value}.BJ"
    return value


def load_local_stock_codes(engine: Engine) -> list[str]:
    queries = [
        """
        SELECT DISTINCT stock_code, exchange
        FROM stock_instruments
        WHERE stock_code IS NOT NULL AND stock_code <> ''
        """,
        """
        SELECT DISTINCT stock_code, exchange
        FROM stock_5m_bars
        WHERE stock_code IS NOT NULL AND stock_code <> ''
        """,
        """
        SELECT DISTINCT stock_code, exchange
        FROM stock_snapshots
        WHERE stock_code IS NOT NULL AND stock_code <> ''
        """,
    ]
    codes: set[str] = set()
    with engine.connect() as conn:
        for query in queries:
            try:
                rows = conn.execute(text(query)).mappings().all()
            except Exception:
                continue
            for row in rows:
                codes.add(to_ts_code(str(row["stock_code"]), row.get("exchange")))
    return sorted(codes)


def load_tushare_stock_codes(pro: Any) -> list[str]:
    frames = []
    for list_status in ["L", "D", "P"]:
        frame = pro.query("stock_basic", exchange="", list_status=list_status)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return []
    combined = pd.concat(frames, ignore_index=True)
    return sorted(set(str(value) for value in combined["ts_code"].dropna().tolist()))


def resolve_trade_dates(
    pro: Any,
    start_date: date,
    end_date: date,
) -> list[date]:
    try:
        frame = pro.query(
            "trade_cal",
            exchange="SSE",
            start_date=to_tushare_date(start_date),
            end_date=to_tushare_date(end_date),
            is_open="1",
        )
    except Exception as exc:
        print(f"WARN trade_cal failed, fallback to calendar days: {exc}")
        return date_range(start_date, end_date)

    if frame.empty or "cal_date" not in frame.columns:
        return date_range(start_date, end_date)
    return [parse_yyyymmdd(str(value)) for value in frame["cal_date"].tolist()]


def unique_month_dates(start_date: date, end_date: date) -> list[date]:
    months = pd.date_range(start=start_date, end=end_date, freq="MS")
    if not months.empty and months[0].date() == start_date.replace(day=1):
        return [item.date() for item in months]
    first = start_date.replace(day=1)
    values = [first] + [item.date() for item in months if item.date() != first]
    return sorted(set(values))


def dates_for_endpoint(
    endpoint: TushareEndpoint,
    start_date: date,
    end_date: date,
    calendar_dates: list[date],
    trade_dates: list[date],
) -> list[date]:
    if endpoint.date_param == "trade_date":
        return trade_dates
    if endpoint.date_param == "month":
        return unique_month_dates(start_date, end_date)
    if endpoint.date_param is None:
        return [end_date]
    return calendar_dates


def insert_rows(
    engine: Engine,
    endpoint: TushareEndpoint,
    frame: pd.DataFrame,
    run_date: date,
    replace_ts_code: str | None = None,
    replace_start_date: date | None = None,
    replace_end_date: date | None = None,
) -> int:
    fetched_at = datetime.now(UTC)
    rows = []
    for _, row in frame.iterrows():
        payload = row_to_dict(row)
        date_key, date_type = date_key_for(endpoint, payload, run_date)
        rows.append(
            {
                "endpoint": endpoint.name,
                "date_key": date_key,
                "date_type": date_type,
                "ts_code": code_for(endpoint, payload),
                "content_type": endpoint.content_type,
                "row_hash": stable_hash(payload),
                "fetched_at": fetched_at,
                "raw": json.dumps(payload, ensure_ascii=True, sort_keys=True),
            }
        )

    delete_stmt, delete_params = latest_slice_delete(endpoint, rows, run_date, replace_ts_code, replace_start_date, replace_end_date)
    insert_stmt = text(
        """
        INSERT INTO tushare_raw_records (
            endpoint, date_key, date_type, ts_code, content_type,
            row_hash, fetched_at, raw
        )
        VALUES (
            :endpoint, :date_key, :date_type, :ts_code, :content_type,
            :row_hash, :fetched_at, CAST(:raw AS jsonb)
        )
        ON CONFLICT (endpoint, date_key, ts_code, content_type, row_hash)
        DO UPDATE SET
            date_type = EXCLUDED.date_type,
            fetched_at = EXCLUDED.fetched_at,
            raw = EXCLUDED.raw
        """
    )
    with engine.begin() as conn:
        conn.execute(delete_stmt, delete_params)
        if not rows:
            return 0
        result = conn.execute(insert_stmt, rows)
    return int(result.rowcount or 0)


def latest_slice_delete(
    endpoint: TushareEndpoint,
    rows: list[dict[str, Any]],
    run_date: date,
    replace_ts_code: str | None,
    replace_start_date: date | None,
    replace_end_date: date | None,
) -> tuple[Any, dict[str, Any]]:
    base_params: dict[str, Any] = {
        "endpoint": endpoint.name,
        "content_type": endpoint.content_type,
    }

    if replace_ts_code and replace_start_date and replace_end_date:
        return (
            text(
                """
                DELETE FROM tushare_raw_records
                WHERE endpoint = :endpoint
                  AND content_type = :content_type
                  AND ts_code = :ts_code
                  AND date_key BETWEEN :start_key AND :end_key
                """
            ),
            {
                **base_params,
                "ts_code": replace_ts_code,
                "start_key": to_tushare_date(replace_start_date),
                "end_key": to_tushare_date(replace_end_date),
            },
        )

    if endpoint.date_param is None:
        return (
            text(
                """
                DELETE FROM tushare_raw_records
                WHERE endpoint = :endpoint
                  AND content_type = :content_type
                """
            ),
            base_params,
        )

    date_keys = sorted({row["date_key"] for row in rows if row["date_key"]})
    if not date_keys:
        date_keys = [run_date.strftime("%Y%m") if endpoint.date_param == "month" else to_tushare_date(run_date)]

    return (
        text(
            """
            DELETE FROM tushare_raw_records
            WHERE endpoint = :endpoint
              AND content_type = :content_type
              AND date_key = ANY(:date_keys)
            """
        ),
        {**base_params, "date_keys": date_keys},
    )


def start_run(engine: Engine, endpoints: list[TushareEndpoint], dates: list[date]) -> uuid.UUID:
    run_id = uuid.uuid4()
    stmt = text(
        """
        INSERT INTO tushare_raw_runs (
            run_id, started_at, status, start_date, end_date, endpoints, trade_dates
        )
        VALUES (
            :run_id, :started_at, 'running', :start_date, :end_date,
            CAST(:endpoints AS jsonb), CAST(:trade_dates AS jsonb)
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(
            stmt,
            {
                "run_id": run_id,
                "started_at": datetime.now(UTC),
                "start_date": min(dates) if dates else None,
                "end_date": max(dates) if dates else None,
                "endpoints": json.dumps([asdict(item) for item in endpoints], ensure_ascii=True),
                "trade_dates": json.dumps([to_tushare_date(item) for item in dates], ensure_ascii=True),
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
    stmt = text(
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
    )
    with engine.begin() as conn:
        conn.execute(
            stmt,
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


def collect(
    start_date: date,
    end_date: date,
    priorities: set[str],
    endpoint_names: set[str] | None,
    ts_codes: set[str] | None,
    all_codes: bool,
    max_codes: int | None,
    sleep_seconds: float,
) -> None:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    ensure_schema(engine)

    endpoints = select_endpoints(priorities=priorities, names=endpoint_names)
    if not endpoints:
        raise RuntimeError("No endpoints selected.")

    ts.set_token(settings.tushare_token)
    pro = ts.pro_api()
    calendar_dates = date_range(start_date, end_date)
    trade_dates = resolve_trade_dates(pro, start_date, end_date)
    run_dates = sorted(set(calendar_dates + trade_dates))
    selected_codes = sorted({to_ts_code(item) for item in ts_codes}) if ts_codes else []
    if all_codes:
        selected_codes = load_local_stock_codes(engine)
        if not selected_codes:
            selected_codes = load_tushare_stock_codes(pro)
    if max_codes is not None:
        selected_codes = selected_codes[:max_codes]

    run_id = start_run(engine, endpoints, run_dates)
    started = time.monotonic()
    row_count = 0
    errors: list[str] = []

    for endpoint in endpoints:
        endpoint_dates = dates_for_endpoint(endpoint, start_date, end_date, calendar_dates, trade_dates)
        for run_date in endpoint_dates:
            if endpoint.requires_ts_code and not selected_codes:
                message = (
                    f"{to_tushare_date(run_date)} {endpoint.name}: skipped, "
                    "requires --ts-code or --all-codes"
                )
                errors.append(message)
                print(f"WARN {message}")
                continue
            try:
                if endpoint.requires_ts_code:
                    fetched_total = 0
                    inserted_total = 0
                    for ts_code in selected_codes:
                        frame = fetch_endpoint_for_code(pro, endpoint, run_date, ts_code)
                        inserted = insert_rows(engine, endpoint, frame, run_date)
                        fetched_total += len(frame)
                        inserted_total += inserted
                        if sleep_seconds > 0:
                            time.sleep(sleep_seconds)
                    row_count += inserted_total
                    print(
                        f"{to_tushare_date(run_date)} {endpoint.name}: "
                        f"codes={len(selected_codes)} fetched={fetched_total} inserted={inserted_total}"
                    )
                else:
                    frame = fetch_endpoint(pro, endpoint, run_date)
                    inserted = insert_rows(engine, endpoint, frame, run_date)
                    row_count += inserted
                    print(f"{to_tushare_date(run_date)} {endpoint.name}: fetched={len(frame)} inserted={inserted}")
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
            except Exception as exc:  # Keep other endpoints moving when one API has no permission.
                message = f"{to_tushare_date(run_date)} {endpoint.name}: {exc}"
                errors.append(message)
                print(f"ERROR {message}")

    elapsed = time.monotonic() - started
    status = "success" if not errors else "partial_success"
    finish_run(
        engine,
        run_id,
        status=status,
        row_count=row_count,
        error_count=len(errors),
        elapsed_seconds=elapsed,
        note="\n".join(errors[:100]),
    )
    print(f"run_id={run_id} status={status} inserted={row_count} errors={len(errors)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Tushare raw data into PostgreSQL.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD format.")
    parser.add_argument(
        "--priority",
        action="append",
        choices=["P0", "P1", "P2", "P3"],
        default=None,
        help="Priority group to collect. Can be repeated. Defaults to P0.",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        default=None,
        help="Specific endpoint to collect. Can be repeated.",
    )
    parser.add_argument(
        "--ts-code",
        action="append",
        default=None,
        help="Specific Tushare stock code for stock-scoped endpoints. Can be repeated.",
    )
    parser.add_argument(
        "--all-codes",
        action="store_true",
        help="Collect stock-scoped endpoints for all local stock codes, falling back to Tushare stock_basic.",
    )
    parser.add_argument(
        "--max-codes",
        type=int,
        default=None,
        help="Limit stock-scoped collection to the first N selected codes.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between API requests.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    collect(
        start_date=parse_yyyymmdd(args.start_date),
        end_date=parse_yyyymmdd(args.end_date),
        priorities=set(args.priority or ["P0"]),
        endpoint_names=set(args.endpoint) if args.endpoint else None,
        ts_codes=set(args.ts_code) if args.ts_code else None,
        all_codes=args.all_codes,
        max_codes=args.max_codes,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()

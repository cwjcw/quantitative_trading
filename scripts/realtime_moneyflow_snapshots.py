from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from datetime import datetime, time as dt_time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

import requests
import tushare as ts
from sqlalchemy import text

from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
EASTMONEY_URL = "https://push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_UT = "b2884a393a59ad64002292a3e90d46a5"
DEFAULT_SCOPES = "stock,industry,concept"

DDL = """
CREATE TABLE IF NOT EXISTS public.realtime_moneyflow_runs (
    run_id uuid PRIMARY KEY,
    source text NOT NULL,
    started_at timestamptz NOT NULL,
    finished_at timestamptz,
    trade_date date NOT NULL,
    status text NOT NULL,
    scope_list text[] NOT NULL,
    requested_pages integer NOT NULL DEFAULT 0,
    returned_rows integer NOT NULL DEFAULT 0,
    inserted_rows integer NOT NULL DEFAULT 0,
    elapsed_seconds numeric,
    error text
);

CREATE TABLE IF NOT EXISTS public.realtime_moneyflow_snapshots (
    run_id uuid NOT NULL REFERENCES public.realtime_moneyflow_runs(run_id) ON DELETE CASCADE,
    trade_date date NOT NULL,
    captured_at timestamptz NOT NULL,
    source text NOT NULL,
    scope text NOT NULL,
    code text NOT NULL,
    ts_code text,
    name text,
    latest_price numeric,
    pct_change numeric,
    main_net_amount numeric,
    main_net_rate numeric,
    super_large_net_amount numeric,
    super_large_net_rate numeric,
    large_net_amount numeric,
    large_net_rate numeric,
    medium_net_amount numeric,
    medium_net_rate numeric,
    small_net_amount numeric,
    small_net_rate numeric,
    leading_code text,
    leading_name text,
    leading_pct numeric,
    quote_timestamp timestamptz,
    rank_no integer,
    raw jsonb NOT NULL,
    PRIMARY KEY (run_id, scope, code)
);

CREATE INDEX IF NOT EXISTS idx_realtime_moneyflow_snapshots_scope_time
    ON public.realtime_moneyflow_snapshots (scope, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_realtime_moneyflow_snapshots_ts_code_time
    ON public.realtime_moneyflow_snapshots (ts_code, captured_at DESC)
    WHERE ts_code IS NOT NULL;

CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.realtime_moneyflow_latest AS
SELECT DISTINCT ON (scope, code)
       run_id,
       trade_date,
       captured_at,
       source,
       scope,
       code,
       ts_code,
       name,
       latest_price,
       pct_change,
       main_net_amount,
       main_net_rate,
       super_large_net_amount,
       super_large_net_rate,
       large_net_amount,
       large_net_rate,
       medium_net_amount,
       medium_net_rate,
       small_net_amount,
       small_net_rate,
       leading_code,
       leading_name,
       leading_pct,
       quote_timestamp,
       rank_no,
       raw
FROM public.realtime_moneyflow_snapshots
ORDER BY scope, code, captured_at DESC;

CREATE OR REPLACE VIEW analytics.realtime_sector_moneyflow_latest AS
SELECT *
FROM analytics.realtime_moneyflow_latest
WHERE scope IN ('industry', 'concept', 'region');

CREATE OR REPLACE VIEW analytics.realtime_dc_member_moneyflow_latest AS
WITH latest_stock AS (
    SELECT *
    FROM analytics.realtime_moneyflow_latest
    WHERE scope = 'stock'
),
latest_members AS (
    SELECT DISTINCT ON (
           replace(ts_code, '.DC', ''),
           raw->>'con_code'
       )
       replace(ts_code, '.DC', '') AS sector_code,
       raw->>'con_code' AS con_code
    FROM public.tushare_raw_records
    WHERE endpoint = 'dc_member'
      AND raw ? 'con_code'
      AND ts_code IS NOT NULL
    ORDER BY replace(ts_code, '.DC', ''), raw->>'con_code', date_key DESC, fetched_at DESC
),
latest_sectors AS (
    SELECT DISTINCT ON (replace(ts_code, '.DC', ''))
       replace(ts_code, '.DC', '') AS sector_code,
       raw->>'name' AS sector_name,
       raw->>'idx_type' AS sector_type
    FROM public.tushare_raw_records
    WHERE endpoint = 'dc_index'
      AND ts_code IS NOT NULL
    ORDER BY replace(ts_code, '.DC', ''), date_key DESC, fetched_at DESC
)
SELECT
    m.sector_code,
    s.sector_name,
    s.sector_type,
    max(st.trade_date) AS trade_date,
    max(st.captured_at) AS captured_at,
    count(*) AS member_count,
    count(*) FILTER (WHERE st.main_net_amount > 0) AS positive_member_count,
    sum(st.main_net_amount) AS member_main_net_amount,
    avg(st.main_net_rate) AS avg_main_net_rate,
    sum(st.super_large_net_amount) AS member_super_large_net_amount,
    sum(st.large_net_amount) AS member_large_net_amount,
    sum(st.medium_net_amount) AS member_medium_net_amount,
    sum(st.small_net_amount) AS member_small_net_amount
FROM latest_members m
JOIN latest_stock st ON st.ts_code = m.con_code
LEFT JOIN latest_sectors s ON s.sector_code = m.sector_code
GROUP BY m.sector_code, s.sector_name, s.sector_type;
"""

SCOPE_CONFIG = {
    "stock": {
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "referer": "https://data.eastmoney.com/zjlx/detail.html",
    },
    "industry": {
        "fs": "m:90+t:2",
        "referer": "https://data.eastmoney.com/bkzj/hy.html",
    },
    "concept": {
        "fs": "m:90+t:3",
        "referer": "https://data.eastmoney.com/bkzj/gn.html",
    },
    "region": {
        "fs": "m:90+t:1",
        "referer": "https://data.eastmoney.com/bkzj/dy.html",
    },
}

FIELDS = ",".join(
    [
        "f2",
        "f3",
        "f12",
        "f14",
        "f62",
        "f66",
        "f69",
        "f72",
        "f75",
        "f78",
        "f81",
        "f84",
        "f87",
        "f124",
        "f128",
        "f136",
        "f140",
        "f184",
        "f207",
        "f208",
        "f209",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect intraday Eastmoney moneyflow snapshots.")
    parser.add_argument("--once", action="store_true", help="Collect one snapshot and exit.")
    parser.add_argument("--loop", action="store_true", help="Collect snapshots repeatedly.")
    parser.add_argument("--interval", type=int, default=600, help="Loop interval seconds.")
    parser.add_argument("--start-at", default="09:30", help="Local start time HH:MM for loop mode.")
    parser.add_argument("--end-at", default="15:05", help="Local end time HH:MM for loop mode.")
    parser.add_argument("--pause-start", default="11:35", help="Local pause start HH:MM for loop mode.")
    parser.add_argument("--pause-end", default="12:59", help="Local pause end HH:MM for loop mode.")
    parser.add_argument("--page-size", type=int, default=100, help="Eastmoney rows per request.")
    parser.add_argument("--timeout", type=float, default=12.0, help="HTTP request timeout seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed HTTP page.")
    parser.add_argument("--retry-sleep", type=float, default=1.5, help="Seconds to sleep between retries.")
    parser.add_argument(
        "--scopes",
        default=DEFAULT_SCOPES,
        help=f"Comma-separated scopes. Available: {','.join(SCOPE_CONFIG)}.",
    )
    return parser.parse_args()


def ensure_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(DDL))


def to_tushare_date(value) -> str:
    return value.strftime("%Y%m%d")


def is_trading_day(token: str) -> bool:
    today = datetime.now(LOCAL_TZ).date()
    trade_date = to_tushare_date(today)
    pro = ts.pro_api(token)
    try:
        frame = pro.query(
            "trade_cal",
            exchange="SSE",
            start_date=trade_date,
            end_date=trade_date,
            fields="cal_date,is_open",
        )
    except Exception as exc:
        print(f"WARN trade_cal check failed for {trade_date}, continuing collection: {exc}")
        return True

    if frame is None or frame.empty or "is_open" not in frame.columns:
        print(f"WARN trade_cal returned no usable row for {trade_date}, continuing collection.")
        return True

    is_open = str(frame.iloc[0].get("is_open", "")).strip()
    if is_open == "1":
        return True
    if is_open == "0":
        print(f"{datetime.now(LOCAL_TZ).isoformat(timespec='seconds')} {trade_date} is not a trading day, exiting.")
        return False

    print(f"WARN trade_cal returned unexpected is_open={is_open!r} for {trade_date}, continuing collection.")
    return True


def parse_local_time(value: str) -> dt_time:
    hour, minute = value.split(":", 1)
    return dt_time(int(hour), int(minute), tzinfo=LOCAL_TZ)


def next_eligible_time(
    now: datetime,
    start_time: dt_time,
    end_time: dt_time,
    pause_start: dt_time,
    pause_end: dt_time,
) -> datetime | None:
    today = now.date()
    start_dt = datetime.combine(today, start_time.replace(tzinfo=None), tzinfo=LOCAL_TZ)
    end_dt = datetime.combine(today, end_time.replace(tzinfo=None), tzinfo=LOCAL_TZ)
    pause_start_dt = datetime.combine(today, pause_start.replace(tzinfo=None), tzinfo=LOCAL_TZ)
    pause_end_dt = datetime.combine(today, pause_end.replace(tzinfo=None), tzinfo=LOCAL_TZ)
    resume_dt = pause_end_dt + timedelta(minutes=1)

    if now < start_dt:
        return start_dt
    if pause_start_dt <= now < resume_dt:
        return resume_dt
    if now > end_dt:
        return None
    return now


def sleep_until(target: datetime) -> None:
    while True:
        remaining = (target - datetime.now(LOCAL_TZ)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30))


def parse_scopes(value: str) -> list[str]:
    scopes = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in scopes if item not in SCOPE_CONFIG]
    if invalid:
        raise ValueError(f"Unsupported scopes: {','.join(invalid)}")
    return scopes


def to_decimal(value: Any) -> Decimal | None:
    if value in (None, "", "-"):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def normalize_raw(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return value


def stock_ts_code(code: str) -> str | None:
    if not code or len(code) != 6 or not code.isdigit():
        return None
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "2", "3")):
        return f"{code}.SZ"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return None


def quote_timestamp(value: Any) -> datetime | None:
    if value in (None, "", "-", 0):
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=LOCAL_TZ)
    except (TypeError, ValueError, OSError):
        return None


def request_page(scope: str, page: int, page_size: int, timeout: float) -> dict[str, Any]:
    config = SCOPE_CONFIG[scope]
    params = {
        "fid": "f62",
        "po": "1",
        "pz": str(page_size),
        "pn": str(page),
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "ut": EASTMONEY_UT,
        "fs": config["fs"],
        "fields": FIELDS,
    }
    session = requests.Session()
    session.trust_env = False
    response = session.get(
        EASTMONEY_URL,
        params=params,
        timeout=timeout,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Referer": config["referer"],
            "User-Agent": "Mozilla/5.0",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("rc") != 0:
        raise RuntimeError(f"Eastmoney returned rc={payload.get('rc')} for scope={scope} page={page}")
    return payload


def fetch_page_with_retry(
    scope: str,
    page: int,
    page_size: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return request_page(scope, page, page_size, timeout)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep)
    raise RuntimeError(f"failed scope={scope} page={page}: {last_error}") from last_error


def fetch_scope(
    scope: str,
    page_size: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> tuple[list[dict[str, Any]], int]:
    payload = fetch_page_with_retry(scope, 1, page_size, timeout, retries, retry_sleep)
    data = payload.get("data") or {}
    total = int(data.get("total") or 0)
    rows = list(data.get("diff") or [])
    actual_page_size = len(rows) or page_size
    pages = max(1, math.ceil(total / actual_page_size)) if total else 1
    for page in range(2, pages + 1):
        payload = fetch_page_with_retry(scope, page, page_size, timeout, retries, retry_sleep)
        page_data = payload.get("data") or {}
        rows.extend(page_data.get("diff") or [])
    return rows, pages


def build_row(raw: dict[str, Any], scope: str, run_id: uuid.UUID, captured_at: datetime, rank_no: int) -> dict[str, Any]:
    code = str(raw.get("f12") or "").strip()
    raw_with_source = {key: normalize_raw(value) for key, value in raw.items()}
    raw_with_source["source"] = "eastmoney.push2.clist"
    raw_with_source["scope"] = scope
    return {
        "run_id": run_id,
        "trade_date": captured_at.date(),
        "captured_at": captured_at,
        "source": "eastmoney",
        "scope": scope,
        "code": code,
        "ts_code": stock_ts_code(code) if scope == "stock" else None,
        "name": raw.get("f14"),
        "latest_price": to_decimal(raw.get("f2")),
        "pct_change": to_decimal(raw.get("f3")),
        "main_net_amount": to_decimal(raw.get("f62")),
        "main_net_rate": to_decimal(raw.get("f184")),
        "super_large_net_amount": to_decimal(raw.get("f66")),
        "super_large_net_rate": to_decimal(raw.get("f69")),
        "large_net_amount": to_decimal(raw.get("f72")),
        "large_net_rate": to_decimal(raw.get("f75")),
        "medium_net_amount": to_decimal(raw.get("f78")),
        "medium_net_rate": to_decimal(raw.get("f81")),
        "small_net_amount": to_decimal(raw.get("f84")),
        "small_net_rate": to_decimal(raw.get("f87")),
        "leading_code": raw.get("f140") or raw.get("f208"),
        "leading_name": raw.get("f128") or raw.get("f207"),
        "leading_pct": to_decimal(raw.get("f136") or raw.get("f209")),
        "quote_timestamp": quote_timestamp(raw.get("f124")),
        "rank_no": rank_no,
        "raw": json.dumps(raw_with_source, ensure_ascii=False, default=str),
    }


def start_run(engine, run_id: uuid.UUID, captured_at: datetime, scopes: list[str]) -> None:
    query = text(
        """
        INSERT INTO public.realtime_moneyflow_runs (
            run_id, source, started_at, trade_date, status, scope_list
        ) VALUES (
            :run_id, 'eastmoney', :started_at, :trade_date, 'running', :scope_list
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "run_id": run_id,
                "started_at": captured_at,
                "trade_date": captured_at.date(),
                "scope_list": scopes,
            },
        )


def finish_run(
    engine,
    run_id: uuid.UUID,
    status: str,
    requested_pages: int,
    returned_rows: int,
    inserted_rows: int,
    elapsed_seconds: float,
    error: str | None = None,
) -> None:
    query = text(
        """
        UPDATE public.realtime_moneyflow_runs
        SET finished_at = :finished_at,
            status = :status,
            requested_pages = :requested_pages,
            returned_rows = :returned_rows,
            inserted_rows = :inserted_rows,
            elapsed_seconds = :elapsed_seconds,
            error = :error
        WHERE run_id = :run_id
        """
    )
    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "run_id": run_id,
                "finished_at": datetime.now(LOCAL_TZ),
                "status": status,
                "requested_pages": requested_pages,
                "returned_rows": returned_rows,
                "inserted_rows": inserted_rows,
                "elapsed_seconds": Decimal(str(round(elapsed_seconds, 4))),
                "error": error,
            },
        )


def insert_snapshots(engine, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    query = text(
        """
        INSERT INTO public.realtime_moneyflow_snapshots (
            run_id, trade_date, captured_at, source, scope, code, ts_code, name,
            latest_price, pct_change, main_net_amount, main_net_rate,
            super_large_net_amount, super_large_net_rate, large_net_amount, large_net_rate,
            medium_net_amount, medium_net_rate, small_net_amount, small_net_rate,
            leading_code, leading_name, leading_pct, quote_timestamp, rank_no, raw
        ) VALUES (
            :run_id, :trade_date, :captured_at, :source, :scope, :code, :ts_code, :name,
            :latest_price, :pct_change, :main_net_amount, :main_net_rate,
            :super_large_net_amount, :super_large_net_rate, :large_net_amount, :large_net_rate,
            :medium_net_amount, :medium_net_rate, :small_net_amount, :small_net_rate,
            :leading_code, :leading_name, :leading_pct, :quote_timestamp, :rank_no, CAST(:raw AS jsonb)
        )
        ON CONFLICT (run_id, scope, code) DO UPDATE SET
            raw = EXCLUDED.raw
        """
    )
    with engine.begin() as conn:
        conn.execute(query, rows)
    return len(rows)


def collect_once(
    engine,
    scopes: list[str],
    page_size: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> int:
    run_id = uuid.uuid4()
    captured_at = datetime.now(LOCAL_TZ)
    started = time.perf_counter()
    requested_pages = 0
    returned_rows = 0
    inserted_rows = 0
    start_run(engine, run_id, captured_at, scopes)
    try:
        rows: list[dict[str, Any]] = []
        for scope in scopes:
            raw_rows, pages = fetch_scope(scope, page_size, timeout, retries, retry_sleep)
            requested_pages += pages
            returned_rows += len(raw_rows)
            rows.extend(
                build_row(raw, scope, run_id, captured_at, rank_no)
                for rank_no, raw in enumerate(raw_rows, start=1)
            )
        inserted_rows = insert_snapshots(engine, rows)
        elapsed = time.perf_counter() - started
        finish_run(engine, run_id, "success", requested_pages, returned_rows, inserted_rows, elapsed)
        print(
            f"{captured_at.isoformat(timespec='seconds')} run_id={run_id} "
            f"scopes={','.join(scopes)} pages={requested_pages} returned={returned_rows} "
            f"inserted={inserted_rows} elapsed={elapsed:.2f}s"
        )
        return inserted_rows
    except Exception as exc:
        elapsed = time.perf_counter() - started
        finish_run(
            engine,
            run_id,
            "failed",
            requested_pages,
            returned_rows,
            inserted_rows,
            elapsed,
            error=str(exc),
        )
        print(f"{captured_at.isoformat(timespec='seconds')} run_id={run_id} failed elapsed={elapsed:.2f}s error={exc}")
        raise


def run_loop(
    engine,
    scopes: list[str],
    page_size: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
    interval: int,
    start_at: str,
    end_at: str,
    pause_start_at: str,
    pause_end_at: str,
) -> None:
    start_time = parse_local_time(start_at)
    end_time = parse_local_time(end_at)
    pause_start = parse_local_time(pause_start_at)
    pause_end = parse_local_time(pause_end_at)
    while True:
        now = datetime.now(LOCAL_TZ)
        target = next_eligible_time(now, start_time, end_time, pause_start, pause_end)
        if target is None:
            print(f"{now.isoformat(timespec='seconds')} reached end-at {end_at}, exiting.")
            return
        if target > now:
            print(f"{now.isoformat(timespec='seconds')} sleeping until {target.isoformat(timespec='seconds')}")
            sleep_until(target)
        started = datetime.now(LOCAL_TZ)
        try:
            collect_once(engine, scopes, page_size, timeout, retries, retry_sleep)
        except Exception as exc:
            print(f"{started.isoformat(timespec='seconds')} collection failed, will retry next interval: {exc}")
        time.sleep(interval)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not args.once and not args.loop:
        args.loop = True
    if not is_trading_day(settings.tushare_token):
        return
    scopes = parse_scopes(args.scopes)
    engine = make_engine(settings.database_url)
    ensure_schema(engine)
    if args.once:
        collect_once(engine, scopes, args.page_size, args.timeout, args.retries, args.retry_sleep)
        return
    run_loop(
        engine,
        scopes,
        args.page_size,
        args.timeout,
        args.retries,
        args.retry_sleep,
        args.interval,
        args.start_at,
        args.end_at,
        args.pause_start,
        args.pause_end,
    )


if __name__ == "__main__":
    main()

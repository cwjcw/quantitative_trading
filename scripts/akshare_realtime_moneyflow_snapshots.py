from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, time as dt_time, timedelta
from decimal import Decimal, InvalidOperation
from io import StringIO
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import requests
from akshare.stock_feature import stock_fund_flow as ak_ths_fund_flow
from bs4 import BeautifulSoup
from sqlalchemy import text

from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_SCOPES = "stock,industry,concept"
THS_STOCK_MIN_ROWS = 4500
THS_REQUEST_TIMEOUT = 15

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
WHERE scope IN ('industry', 'concept');

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


def ensure_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(DDL))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect intraday AkShare/THS moneyflow snapshots.")
    parser.add_argument("--once", action="store_true", help="Collect one snapshot and exit.")
    parser.add_argument("--loop", action="store_true", help="Collect snapshots repeatedly.")
    parser.add_argument("--interval", type=int, default=600, help="Loop interval seconds.")
    parser.add_argument("--start-at", default="09:30", help="Local start time HH:MM for loop mode.")
    parser.add_argument("--end-at", default="15:05", help="Local end time HH:MM for loop mode.")
    parser.add_argument("--pause-start", default="11:35", help="Local pause start HH:MM for loop mode.")
    parser.add_argument("--pause-end", default="12:59", help="Local pause end HH:MM for loop mode.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed scope.")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Seconds to sleep between retries.")
    parser.add_argument(
        "--scopes",
        default=DEFAULT_SCOPES,
        help="Comma-separated scopes. Available: stock,industry,concept.",
    )
    return parser.parse_args()


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
    invalid = [item for item in scopes if item not in {"stock", "industry", "concept"}]
    if invalid:
        raise ValueError(f"Unsupported scopes: {','.join(invalid)}")
    return scopes


def to_decimal(value: Any) -> Decimal | None:
    if value in (None, "", "-", "--"):
        return None
    if isinstance(value, Decimal):
        return value
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    multiplier = Decimal("1")
    if text.endswith("万"):
        multiplier = Decimal("10000")
        text = text[:-1]
    elif text.endswith("亿"):
        multiplier = Decimal("100000000")
        text = text[:-1]
    try:
        return Decimal(text) * multiplier
    except (InvalidOperation, ValueError):
        return None


def stock_ts_code(code: Any) -> str | None:
    symbol = str(code or "").strip().zfill(6)
    if len(symbol) != 6 or not symbol.isdigit():
        return None
    if symbol.startswith(("6", "9")):
        return f"{symbol}.SH"
    if symbol.startswith(("0", "2", "3")):
        return f"{symbol}.SZ"
    if symbol.startswith(("4", "8")):
        return f"{symbol}.BJ"
    return None


def ths_headers() -> dict[str, str]:
    js_code = ak_ths_fund_flow.py_mini_racer.MiniRacer()
    js_code.eval(ak_ths_fund_flow._get_file_content_ths("ths.js"))
    return {
        "Accept": "text/html, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "hexin-v": js_code.call("v"),
        "Pragma": "no-cache",
        "Referer": "https://data.10jqka.com.cn/funds/ggzjl/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }


def normalize_ths_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [
            "-".join(str(part) for part in column if str(part) != "nan").strip("-")
            for column in normalized.columns
        ]
    normalized.columns = [
        str(column)
        .strip()
        .replace("(元)", "")
        .replace("（元）", "")
        .replace("资金流入", "流入资金")
        .replace("资金流出", "流出资金")
        for column in normalized.columns
    ]
    return normalized


def parse_ths_stock_page(html: str, page: int) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"THS stock page {page} has no data table") from exc
    if not tables:
        raise RuntimeError(f"THS stock page {page} has no data table")
    frame = normalize_ths_columns(tables[0])
    required = {"股票代码", "股票简称", "最新价", "涨跌幅", "净额"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"THS stock page {page} missing columns {missing}; actual={list(frame.columns)}"
        )
    return frame


def request_ths_stock_page(
    session: requests.Session,
    page: int,
    retries: int,
    retry_sleep: float,
) -> tuple[pd.DataFrame, int]:
    url = (
        "https://data.10jqka.com.cn/funds/ggzjl/"
        f"field/zdf/order/desc/page/{page}/ajax/1/"
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(
                url,
                headers=ths_headers(),
                timeout=THS_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            frame = parse_ths_stock_page(response.text, page)
            soup = BeautifulSoup(response.text, features="lxml")
            page_info = soup.find(name="span", attrs={"class": "page_info"})
            if page_info is None:
                raise RuntimeError(f"THS stock page {page} is missing page_info")
            total_pages = int(page_info.text.strip().split("/", 1)[1])
            return frame, total_pages
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"THS stock page {page} failed after retries: {last_error}") from last_error


def fetch_ths_stock_full(retries: int, retry_sleep: float) -> pd.DataFrame:
    session = requests.Session()
    first_frame, total_pages = request_ths_stock_page(session, 1, retries, retry_sleep)
    if total_pages < 80:
        raise RuntimeError(
            f"THS stock pagination is truncated: total_pages={total_pages}, expected at least 80"
        )

    frames = [first_frame]
    for page in range(2, total_pages + 1):
        frame, reported_total_pages = request_ths_stock_page(
            session,
            page,
            retries,
            retry_sleep,
        )
        if reported_total_pages != total_pages:
            raise RuntimeError(
                "THS stock page count changed during collection: "
                f"page=1 reported {total_pages}, page={page} reported {reported_total_pages}"
            )
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["股票代码"] = (
        result["股票代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    result = result.drop_duplicates(subset=["股票代码"], keep="first").reset_index(drop=True)
    if len(result) < THS_STOCK_MIN_ROWS:
        raise RuntimeError(
            f"THS stock result is incomplete: unique_rows={len(result)}, "
            f"minimum={THS_STOCK_MIN_ROWS}, pages={total_pages}"
        )
    result.attrs["requested_pages"] = total_pages
    return result


def fetch_scope(scope: str):
    if scope == "stock":
        raise ValueError("stock scope requires fetch_ths_stock_full")
    if scope == "industry":
        return ak.stock_fund_flow_industry(symbol="即时")
    if scope == "concept":
        return ak.stock_fund_flow_concept(symbol="即时")
    raise ValueError(f"Unsupported scope: {scope}")


def fetch_scope_with_retry(scope: str, retries: int, retry_sleep: float):
    if scope == "stock":
        return fetch_ths_stock_full(retries, retry_sleep)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fetch_scope(scope)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"{scope} failed after retries: {last_error}") from last_error


def start_run(engine, run_id: uuid.UUID, captured_at: datetime, scopes: list[str]) -> None:
    query = text(
        """
        INSERT INTO public.realtime_moneyflow_runs (
            run_id, source, started_at, trade_date, status, scope_list
        ) VALUES (
            :run_id, 'akshare_ths', :started_at, :trade_date, 'running', :scope_list
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


def build_stock_row(raw: dict[str, Any], run_id: uuid.UUID, captured_at: datetime, rank_no: int) -> dict[str, Any]:
    code = str(raw.get("股票代码") or "").strip().zfill(6)
    return {
        "run_id": run_id,
        "trade_date": captured_at.date(),
        "captured_at": captured_at,
        "source": "akshare_ths",
        "scope": "stock",
        "code": code,
        "ts_code": stock_ts_code(code),
        "name": raw.get("股票简称"),
        "latest_price": to_decimal(raw.get("最新价")),
        "pct_change": to_decimal(raw.get("涨跌幅")),
        "main_net_amount": to_decimal(raw.get("净额")),
        "main_net_rate": None,
        "super_large_net_amount": None,
        "super_large_net_rate": None,
        "large_net_amount": None,
        "large_net_rate": None,
        "medium_net_amount": None,
        "medium_net_rate": None,
        "small_net_amount": None,
        "small_net_rate": None,
        "leading_code": None,
        "leading_name": None,
        "leading_pct": None,
        "quote_timestamp": captured_at,
        "rank_no": rank_no,
        "raw": json.dumps(raw, ensure_ascii=False, default=str),
    }


def build_sector_row(
    raw: dict[str, Any],
    scope: str,
    run_id: uuid.UUID,
    captured_at: datetime,
    rank_no: int,
) -> dict[str, Any]:
    name = str(raw.get("行业") or "").strip()
    return {
        "run_id": run_id,
        "trade_date": captured_at.date(),
        "captured_at": captured_at,
        "source": "akshare_ths",
        "scope": scope,
        "code": f"AK:{scope}:{name}",
        "ts_code": None,
        "name": name,
        "latest_price": to_decimal(raw.get("行业指数")),
        "pct_change": to_decimal(raw.get("行业-涨跌幅")),
        "main_net_amount": to_decimal(raw.get("净额")),
        "main_net_rate": None,
        "super_large_net_amount": None,
        "super_large_net_rate": None,
        "large_net_amount": None,
        "large_net_rate": None,
        "medium_net_amount": None,
        "medium_net_rate": None,
        "small_net_amount": None,
        "small_net_rate": None,
        "leading_code": None,
        "leading_name": raw.get("领涨股"),
        "leading_pct": to_decimal(raw.get("领涨股-涨跌幅")),
        "quote_timestamp": captured_at,
        "rank_no": rank_no,
        "raw": json.dumps(raw, ensure_ascii=False, default=str),
    }


def collect_once(engine, scopes: list[str], retries: int, retry_sleep: float) -> int:
    run_id = uuid.uuid4()
    captured_at = datetime.now(LOCAL_TZ)
    started = time.perf_counter()
    start_run(engine, run_id, captured_at, scopes)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    requested_pages = 0
    returned_rows = 0
    inserted_rows = 0
    try:
        for scope in scopes:
            try:
                frame = fetch_scope_with_retry(scope, retries, retry_sleep)
                records = frame.to_dict("records")
                requested_pages += int(frame.attrs.get("requested_pages", 1))
                returned_rows += len(records)
                for rank_no, raw in enumerate(records, start=1):
                    if scope == "stock":
                        rows.append(build_stock_row(raw, run_id, captured_at, rank_no))
                    else:
                        rows.append(build_sector_row(raw, scope, run_id, captured_at, rank_no))
            except Exception as exc:
                message = f"{scope} failed: {exc}"
                errors.append(message)
                print(f"WARN {message}")

        inserted_rows = insert_snapshots(engine, rows)
        elapsed = time.perf_counter() - started
        status = "success"
        if errors and inserted_rows:
            status = "partial_success"
        elif errors:
            status = "failed"
        finish_run(
            engine,
            run_id,
            status,
            requested_pages,
            returned_rows,
            inserted_rows,
            elapsed,
            error="\n".join(errors)[:5000] if errors else None,
        )
        print(
            f"{captured_at.isoformat(timespec='seconds')} run_id={run_id} "
            f"status={status} source=akshare_ths scopes={','.join(scopes)} "
            f"returned={returned_rows} inserted={inserted_rows} elapsed={elapsed:.2f}s"
        )
        if status == "failed":
            raise RuntimeError("; ".join(errors))
        return inserted_rows
    except Exception as exc:
        elapsed = time.perf_counter() - started
        if not errors:
            errors.append(str(exc))
        finish_run(engine, run_id, "failed", requested_pages, returned_rows, inserted_rows, elapsed, "\n".join(errors)[:5000])
        raise


def run_loop(
    engine,
    scopes: list[str],
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
            collect_once(engine, scopes, retries, retry_sleep)
        except Exception as exc:
            print(f"{started.isoformat(timespec='seconds')} collection failed, will retry next interval: {exc}")
        time.sleep(interval)


def main() -> None:
    args = parse_args()
    if not args.once and not args.loop:
        args.loop = True
    settings = get_settings()
    scopes = parse_scopes(args.scopes)
    engine = make_engine(settings.database_url)
    ensure_schema(engine)
    if args.once:
        collect_once(engine, scopes, args.retries, args.retry_sleep)
        return
    run_loop(
        engine,
        scopes,
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

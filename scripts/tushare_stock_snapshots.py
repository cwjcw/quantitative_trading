from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, time as dt_time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts
from sqlalchemy import text

from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_CODES = "300806.SZ,002938.SZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Tushare realtime_quote snapshots into public.stock_snapshots.")
    parser.add_argument("--codes", default=DEFAULT_CODES, help="Comma-separated TS codes.")
    parser.add_argument("--all-stock", action="store_true", help="Fetch all listed A-share stocks.")
    parser.add_argument("--batch-size", type=int, default=800, help="Codes per Tushare realtime_quote request.")
    parser.add_argument("--src", default="sina", choices=["sina", "dc"], help="Tushare realtime quote source.")
    parser.add_argument("--once", action="store_true", help="Fetch one snapshot batch and exit.")
    parser.add_argument("--loop", action="store_true", help="Fetch snapshots repeatedly.")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval seconds.")
    parser.add_argument("--start-at", default="09:10", help="Local start time HH:MM for loop mode.")
    parser.add_argument("--end-at", default="15:05", help="Local end time HH:MM for loop mode.")
    parser.add_argument("--pause-start", default="11:35", help="Local pause start HH:MM for loop mode.")
    parser.add_argument("--pause-end", default="12:59", help="Local pause end HH:MM for loop mode.")
    parser.add_argument(
        "--cleanup-history-after",
        default="09:40",
        help="On startup, delete snapshots from dates before today whose quote time is after HH:MM. Empty disables cleanup.",
    )
    return parser.parse_args()


def normalize(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, Decimal):
        return str(value)
    return value


def row_to_raw(row: pd.Series) -> dict[str, Any]:
    return {key: normalize(value) for key, value in row.to_dict().items()}


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    numeric = to_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def to_decimal(value: Any) -> Decimal | None:
    numeric = to_float(value)
    if numeric is None:
        return None
    try:
        return Decimal(str(numeric))
    except InvalidOperation:
        return None


def calc_change(last_price: float | None, pre_close: float | None) -> tuple[Decimal | None, Decimal | None]:
    if last_price is None or pre_close in (None, 0):
        return None, None
    change = last_price - pre_close
    return Decimal(str(change)), Decimal(str(change / pre_close * 100))


def calc_percent(numerator: float | None, denominator: float | None) -> Decimal | None:
    if numerator is None or denominator in (None, 0):
        return None
    return Decimal(str(numerator / denominator * 100))


def parse_quote_datetime(quote_date: str | None, quote_time: str | None) -> datetime | None:
    if not quote_date or not quote_time:
        return None
    try:
        return datetime.strptime(f"{quote_date} {quote_time}", "%Y%m%d %H:%M:%S").replace(tzinfo=LOCAL_TZ)
    except ValueError:
        return None


def market_phase_at(value: datetime) -> str:
    local_time = value.timetz().replace(tzinfo=None)
    if dt_time(9, 15) <= local_time < dt_time(9, 25):
        return "集合竞价"
    if dt_time(9, 30) <= local_time <= dt_time(11, 30):
        return "盘中"
    if dt_time(11, 30) < local_time < dt_time(13, 0):
        return "午间休市"
    if dt_time(13, 0) <= local_time < dt_time(14, 57):
        return "盘中"
    if dt_time(14, 57) <= local_time <= dt_time(15, 0):
        return "收盘集合竞价"
    if local_time > dt_time(15, 0):
        return "收盘"
    return "盘前"


def exchange_for(ts_code: str) -> str | None:
    if "." not in ts_code:
        return None
    return ts_code.rsplit(".", 1)[1]


def load_codes(all_stock: bool, codes: str, token: str) -> list[str]:
    if not all_stock:
        return [code.strip().upper() for code in codes.split(",") if code.strip()]
    pro = ts.pro_api(token)
    frame = pro.stock_basic(exchange="", list_status="L", fields="ts_code")
    return frame["ts_code"].dropna().astype(str).sort_values().tolist()


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

    row = frame.iloc[0]
    is_open = str(row.get("is_open", "")).strip()
    if is_open == "1":
        return True
    if is_open == "0":
        print(f"{datetime.now(LOCAL_TZ).isoformat(timespec='seconds')} {trade_date} is not a trading day, exiting.")
        return False

    print(f"WARN trade_cal returned unexpected is_open={is_open!r} for {trade_date}, continuing collection.")
    return True


def chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_share_map(engine) -> dict[str, dict[str, float | None]]:
    query = text(
        """
        select distinct on (ts_code)
               ts_code,
               (float_share::float * 10000.0) as float_volume,
               (total_share::float * 10000.0) as total_volume
        from analytics.tushare_daily_basic
        where ts_code is not null
        order by ts_code, trade_date desc
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return {
        row["ts_code"]: {
            "float_volume": row["float_volume"],
            "total_volume": row["total_volume"],
        }
        for row in rows
    }


def fetch_realtime(codes: list[str], batch_size: int, src: str) -> pd.DataFrame:
    frames = []
    for batch in chunks(codes, batch_size):
        frame = ts.realtime_quote(ts_code=",".join(batch), src=src)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_snapshot(row: pd.Series, run_id: uuid.UUID, captured_at: datetime, share_map: dict[str, dict[str, float | None]]) -> dict:
    raw = row_to_raw(row)
    ts_code = str(raw.get("TS_CODE") or "").upper()
    quote_dt = parse_quote_datetime(str(raw.get("DATE") or ""), str(raw.get("TIME") or ""))
    quote_dt = quote_dt or captured_at
    trade_date = quote_dt.date()

    last_price = to_float(raw.get("PRICE"))
    open_price = to_float(raw.get("OPEN"))
    high_price = to_float(raw.get("HIGH"))
    low_price = to_float(raw.get("LOW"))
    pre_close = to_float(raw.get("PRE_CLOSE"))
    volume_shares = to_int(raw.get("VOLUME"))
    amount = to_float(raw.get("AMOUNT"))
    volume_lots = int(round(volume_shares / 100)) if volume_shares is not None else None
    change_amount, change_percent = calc_change(last_price, pre_close)
    amplitude_percent = None
    if high_price is not None and low_price is not None:
        amplitude_percent = calc_percent(high_price - low_price, pre_close)
    avg_price = Decimal(str(amount / volume_shares)) if amount is not None and volume_shares else None

    shares = share_map.get(ts_code, {})
    float_volume = shares.get("float_volume")
    total_volume = shares.get("total_volume")
    turnover_rate_float = calc_percent(volume_shares, float_volume)
    turnover_rate_total = calc_percent(volume_shares, total_volume)

    bid_prices = [to_float(raw.get(f"B{i}_P")) for i in range(1, 6)]
    ask_prices = [to_float(raw.get(f"A{i}_P")) for i in range(1, 6)]
    bid_volumes = [to_int(raw.get(f"B{i}_V")) or 0 for i in range(1, 6)]
    ask_volumes = [to_int(raw.get(f"A{i}_V")) or 0 for i in range(1, 6)]

    raw["source"] = "tushare.realtime_quote"
    return {
        "run_id": run_id,
        "trade_date": trade_date,
        "captured_at": captured_at,
        "stock_code": ts_code,
        "exchange": exchange_for(ts_code),
        "raw_time": int(quote_dt.timestamp() * 1000),
        "last_price": to_decimal(last_price),
        "open_price": to_decimal(open_price),
        "high_price": to_decimal(high_price),
        "low_price": to_decimal(low_price),
        "last_close": to_decimal(pre_close),
        "volume": volume_lots,
        "amount": to_decimal(amount),
        "ask_price": json.dumps(ask_prices, ensure_ascii=False),
        "bid_price": json.dumps(bid_prices, ensure_ascii=False),
        "ask_volume": json.dumps(ask_volumes, ensure_ascii=False),
        "bid_volume": json.dumps(bid_volumes, ensure_ascii=False),
        "raw": json.dumps(raw, ensure_ascii=False, default=str),
        "instrument_name": raw.get("NAME"),
        "change_amount": change_amount,
        "change_percent": change_percent,
        "amplitude_percent": amplitude_percent,
        "avg_price": avg_price,
        "pvolume": volume_shares,
        "float_volume": to_decimal(float_volume),
        "total_volume": to_decimal(total_volume),
        "turnover_rate_float": turnover_rate_float,
        "turnover_rate_total": turnover_rate_total,
        "up_stop_price": None,
        "down_stop_price": None,
        "market_phase": market_phase_at(quote_dt),
    }


def insert_snapshots(engine, rows: list[dict]) -> int:
    if not rows:
        return 0
    query = text(
        """
        insert into public.stock_snapshots (
            run_id, trade_date, captured_at, stock_code, exchange, raw_time,
            last_price, open_price, high_price, low_price, last_close,
            volume, amount, ask_price, bid_price, ask_volume, bid_volume, raw,
            instrument_name, change_amount, change_percent, amplitude_percent, avg_price,
            pvolume, float_volume, total_volume, turnover_rate_float, turnover_rate_total,
            up_stop_price, down_stop_price, market_phase
        ) values (
            :run_id, :trade_date, :captured_at, :stock_code, :exchange, :raw_time,
            :last_price, :open_price, :high_price, :low_price, :last_close,
            :volume, :amount, cast(:ask_price as jsonb), cast(:bid_price as jsonb),
            cast(:ask_volume as jsonb), cast(:bid_volume as jsonb), cast(:raw as jsonb),
            :instrument_name, :change_amount, :change_percent, :amplitude_percent, :avg_price,
            :pvolume, :float_volume, :total_volume, :turnover_rate_float, :turnover_rate_total,
            :up_stop_price, :down_stop_price, :market_phase
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(query, rows)
    return len(rows)


def collect_once(engine, codes: list[str], batch_size: int, src: str, share_map: dict[str, dict[str, float | None]]) -> int:
    run_id = uuid.uuid4()
    captured_at = datetime.now(LOCAL_TZ)
    frame = fetch_realtime(codes, batch_size, src)
    rows = [build_snapshot(row, run_id, captured_at, share_map) for _, row in frame.iterrows()]
    inserted = insert_snapshots(engine, rows)
    print(
        f"{captured_at.isoformat(timespec='seconds')} run_id={run_id} "
        f"requested={len(codes)} returned={len(frame)} inserted={inserted}"
    )
    return inserted


def parse_local_time(value: str) -> dt_time:
    hour, minute = value.split(":", 1)
    return dt_time(int(hour), int(minute), tzinfo=LOCAL_TZ)


def cleanup_history_snapshots(engine, cleanup_after: str) -> int:
    if not cleanup_after:
        return 0
    cutoff = parse_local_time(cleanup_after)
    today = datetime.now(LOCAL_TZ).date()
    query = text(
        """
        delete from public.stock_snapshots
        where trade_date < :today
          and raw_time is not null
          and (to_timestamp(raw_time / 1000.0) at time zone 'Asia/Shanghai')::time > :cutoff
        """
    )
    with engine.begin() as conn:
        result = conn.execute(query, {"today": today, "cutoff": cutoff.replace(tzinfo=None)})
    deleted = result.rowcount or 0
    print(f"startup cleanup deleted={deleted} before_date={today} after_time={cleanup_after}")
    return deleted


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


def run_loop(
    engine,
    codes: list[str],
    batch_size: int,
    src: str,
    interval: int,
    start_at: str,
    end_at: str,
    pause_start_at: str,
    pause_end_at: str,
) -> None:
    share_map = load_share_map(engine)
    start_time = parse_local_time(start_at)
    end_time = parse_local_time(end_at)
    pause_start = parse_local_time(pause_start_at)
    pause_end = parse_local_time(pause_end_at)
    final_collected = False
    while True:
        now = datetime.now(LOCAL_TZ)
        target = next_eligible_time(now, start_time, end_time, pause_start, pause_end)
        if target is None:
            if not final_collected:
                started = time.perf_counter()
                print(f"{now.isoformat(timespec='seconds')} reached end-at {end_at}, collecting final snapshot before exit.")
                collect_once(engine, codes, batch_size, src, share_map)
                elapsed = time.perf_counter() - started
                print(f"{datetime.now(LOCAL_TZ).isoformat(timespec='seconds')} final elapsed={elapsed:.2f}s")
                final_collected = True
            print(f"{now.isoformat(timespec='seconds')} reached end-at {end_at}, exiting.")
            return
        if target > now:
            print(f"{now.isoformat(timespec='seconds')} sleeping until {target.isoformat(timespec='seconds')}")
            sleep_until(target)
            now = datetime.now(LOCAL_TZ)

        started = time.perf_counter()
        collect_once(engine, codes, batch_size, src, share_map)
        elapsed = time.perf_counter() - started
        print(f"{datetime.now(LOCAL_TZ).isoformat(timespec='seconds')} elapsed={elapsed:.2f}s")
        time.sleep(interval)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not is_trading_day(settings.tushare_token):
        return
    engine = make_engine(settings.database_url)
    codes = load_codes(args.all_stock, args.codes, settings.tushare_token)
    print(f"loaded {len(codes)} codes")
    if args.loop:
        cleanup_history_snapshots(engine, args.cleanup_history_after)
        run_loop(
            engine,
            codes,
            args.batch_size,
            args.src,
            args.interval,
            args.start_at,
            args.end_at,
            args.pause_start,
            args.pause_end,
        )
        return
    share_map = load_share_map(engine)
    started = time.perf_counter()
    collect_once(engine, codes, args.batch_size, args.src, share_map)
    print(f"done elapsed={time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()

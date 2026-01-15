from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

from db_utils import fetch_min5, get_conn, get_stock_codes, get_trade_dates, normalize_min5, upsert_min5

MAX_WORKERS = 4
MAX_CODES = None  # 设为整数可限制股票数量


def _date_to_dt(date_str: str, time_str: str) -> str:
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str}"


def _fetch_one(symbol: str, start_dt: str, end_dt: str) -> int:
    df = fetch_min5(symbol, start_dt, end_dt)
    df = normalize_min5(df, symbol)
    with get_conn() as conn:
        return upsert_min5(conn, df)


def main() -> None:
    trade_dates = get_trade_dates(20)
    if not trade_dates:
        print("未获取到交易日列表。")
        return

    start_dt = _date_to_dt(trade_dates[0], "09:30:00")
    end_dt = _date_to_dt(trade_dates[-1], "15:00:00")
    print(f"初始化区间: {start_dt} -> {end_dt}")

    codes = get_stock_codes()
    if MAX_CODES:
        codes = codes[:MAX_CODES]
    print(f"股票数量: {len(codes)}")

    total = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_one, code, start_dt, end_dt): code for code in codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                inserted = future.result()
                total += inserted
            except Exception as exc:
                print(f"{code} 失败: {exc}")

    print(f"写入完成，总记录数: {total}")


if __name__ == "__main__":
    main()

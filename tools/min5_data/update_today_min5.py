from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from db_utils import fetch_min5, get_conn, get_stock_codes, normalize_min5, upsert_min5

MAX_WORKERS = 4
MAX_CODES = None  # 设为整数可限制股票数量


def _fetch_one(symbol: str, start_dt: str, end_dt: str) -> int:
    df = fetch_min5(symbol, start_dt, end_dt)
    df = normalize_min5(df, symbol)
    with get_conn() as conn:
        return upsert_min5(conn, df)


def main() -> None:
    today = datetime.now()
    start_dt = today.strftime("%Y-%m-%d 09:30:00")
    end_dt = today.strftime("%Y-%m-%d %H:%M:%S")

    codes = get_stock_codes()
    if MAX_CODES:
        codes = codes[:MAX_CODES]
    print(f"更新日期: {today.strftime('%Y-%m-%d')} | 股票数量: {len(codes)}")

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

    print(f"更新完成，总记录数: {total}")


if __name__ == "__main__":
    main()

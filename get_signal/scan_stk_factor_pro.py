import argparse
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import tushare as ts


DEFAULT_FIELDS = (
    "ts_code,trade_date,close_qfq,"
    "ma_qfq_5,ma_qfq_10,ma_qfq_20,"
    "macd_dif_qfq,macd_dea_qfq,macd_qfq,"
    "rsi_qfq_6,rsi_qfq_12,"
    "kdj_k_qfq,kdj_d_qfq,"
    "boll_mid_qfq"
)


def _yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def get_latest_trade_date(pro, today: Optional[datetime] = None, exchange: str = "SSE") -> str:
    """
    Get the latest open trading date (YYYYMMDD) <= today for the given exchange.
    """
    if today is None:
        today = datetime.now()

    # Look back a bit to cover long holidays.
    start = (today.replace(day=1) - pd.Timedelta(days=45)).to_pydatetime()
    df = pro.trade_cal(exchange=exchange, start_date=_yyyymmdd(start), end_date=_yyyymmdd(today), is_open=1)
    if df is None or df.empty:
        raise RuntimeError("trade_cal returned empty; cannot determine latest trade_date")

    # tushare may return strings; keep as YYYYMMDD string.
    df = df.sort_values("cal_date")
    return str(df["cal_date"].iloc[-1])


def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path, dtype={"ts_code": "string", "trade_date": "string"})
    return None


def fetch_stock_basic(pro, cache_path: Optional[str], refresh: bool) -> pd.DataFrame:
    """
    Fetch stock_basic and optionally cache to CSV.
    """
    if cache_path and not refresh:
        cached = _read_csv_if_exists(cache_path)
        if cached is not None and not cached.empty:
            return cached

    df = pro.stock_basic(list_status="L", fields="ts_code,name,market,exchange")
    if df is None or df.empty:
        raise RuntimeError("stock_basic returned empty")

    df["ts_code"] = df["ts_code"].astype("string")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return df


def fetch_stk_factor_pro(
    pro, trade_date: str, fields: str, cache_path: Optional[str], refresh: bool
) -> pd.DataFrame:
    """
    Fetch stk_factor_pro for one trade_date and optionally cache to CSV.
    """
    if cache_path and not refresh:
        cached = _read_csv_if_exists(cache_path)
        if cached is not None and not cached.empty and str(cached["trade_date"].iloc[0]) == str(trade_date):
            return cached

    df = pro.stk_factor_pro(trade_date=trade_date, fields=fields)
    if df is None or df.empty:
        raise RuntimeError(f"stk_factor_pro returned empty for trade_date={trade_date}")

    df["ts_code"] = df["ts_code"].astype("string")
    df["trade_date"] = df["trade_date"].astype("string")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return df


def add_buy_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Use qfq columns to align with common A-share analysis.
    close = df["close_qfq"]
    ma5 = df["ma_qfq_5"]
    ma10 = df["ma_qfq_10"]
    ma20 = df["ma_qfq_20"]

    dif = df["macd_dif_qfq"]
    dea = df["macd_dea_qfq"]
    hist = df["macd_qfq"]

    rsi6 = df["rsi_qfq_6"]
    rsi12 = df["rsi_qfq_12"]

    k = df["kdj_k_qfq"]
    d = df["kdj_d_qfq"]

    boll_mid = df["boll_mid_qfq"]

    c_ma = (ma5 > ma10) & (ma10 > ma20) & (close > ma20)
    c_macd = (dif > dea) & (hist > 0)
    c_rsi = (rsi6 > 50) & (rsi6 > rsi12)
    c_kdj = (k > d) & (k < 80)
    c_boll = close > boll_mid

    df = df.copy()
    df["c_ma"] = c_ma.fillna(False)
    df["c_macd"] = c_macd.fillna(False)
    df["c_rsi"] = c_rsi.fillna(False)
    df["c_kdj"] = c_kdj.fillna(False)
    df["c_boll"] = c_boll.fillna(False)
    df["buy_score"] = (
        df["c_ma"].astype(int)
        + df["c_macd"].astype(int)
        + df["c_rsi"].astype(int)
        + df["c_kdj"].astype(int)
        + df["c_boll"].astype(int)
    )

    # Simple tie-breakers for same buy_score.
    df["ma20_dist"] = (df["close_qfq"] / df["ma_qfq_20"]) - 1.0
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Scan A-shares using Tushare stk_factor_pro and rank strongest buy signals.")
    p.add_argument("--trade-date", default=None, help="Trade date in YYYYMMDD; default = latest open date.")
    p.add_argument("--top", type=int, default=20, help="Top N to output.")
    p.add_argument("--out-dir", default="data", help="Output directory (ignored by git in this repo).")
    p.add_argument("--refresh", action="store_true", help="Ignore cached CSVs and refetch from Tushare.")
    p.add_argument("--include-kcb", action="store_true", help="Include STAR market (科创板). Default excludes it.")
    p.add_argument("--include-st", action="store_true", help="Include ST stocks. Default excludes them.")
    p.add_argument("--fields", default=DEFAULT_FIELDS, help="Fields to request from stk_factor_pro.")
    args = p.parse_args()

    pro = ts.pro_api()
    trade_date = args.trade_date or get_latest_trade_date(pro)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    cache_basic = os.path.join(out_dir, "tmp_stock_basic.csv")
    cache_factor = os.path.join(out_dir, f"tmp_stk_factor_pro_{trade_date}.csv")

    basic = fetch_stock_basic(pro, cache_basic, refresh=args.refresh)
    factor = fetch_stk_factor_pro(pro, trade_date=trade_date, fields=args.fields, cache_path=cache_factor, refresh=args.refresh)

    merged = factor.merge(basic[["ts_code", "name", "market", "exchange"]], on="ts_code", how="left")

    if not args.include_kcb:
        merged = merged[merged["market"] != "科创板"]

    if not args.include_st:
        # Covers ST and *ST.
        merged = merged[~merged["name"].fillna("").str.upper().str.contains("ST")]

    merged = add_buy_signal_columns(merged)

    ranked = merged.sort_values(
        by=["buy_score", "macd_qfq", "rsi_qfq_6", "ma20_dist", "ts_code"],
        ascending=[False, False, False, False, True],
    )

    # Keep only reasonable columns for the user.
    cols = [
        "ts_code",
        "name",
        "market",
        "exchange",
        "buy_score",
        "close_qfq",
        "ma_qfq_5",
        "ma_qfq_10",
        "ma_qfq_20",
        "macd_dif_qfq",
        "macd_dea_qfq",
        "macd_qfq",
        "rsi_qfq_6",
        "rsi_qfq_12",
        "kdj_k_qfq",
        "kdj_d_qfq",
        "boll_mid_qfq",
        "c_ma",
        "c_macd",
        "c_rsi",
        "c_kdj",
        "c_boll",
    ]
    ranked_out = ranked[cols]
    top = ranked_out.head(args.top).reset_index(drop=True)

    full_path = os.path.join(out_dir, f"stk_factor_pro_buy_score_{trade_date}.csv")
    top_path = os.path.join(out_dir, f"top{args.top}_buy_signals_{trade_date}.csv")
    ranked_out.to_csv(full_path, index=False, encoding="utf-8-sig")
    top.to_csv(top_path, index=False, encoding="utf-8-sig")

    print(f"trade_date={trade_date}  universe_rows={len(ranked_out)}  top={args.top}")
    print(top.to_string(index=False))
    print(f"\nWrote:\n- {full_path}\n- {top_path}\n- {cache_basic}\n- {cache_factor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

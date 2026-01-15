from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List

import akshare as ak
import pandas as pd

from db import get_conn, init_db, upsert_dataframe
from utils import get_symbols_and_market, load_config


FUND_FLOW_INTERVAL_MIN = 10
MINUTE_FETCH_INTERVAL_MIN = 5


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _rename_with(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rename_map = {k: v for k, v in mapping.items() if k in df.columns}
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _fetch_spot(symbols: List[str]) -> pd.DataFrame:
    spot = ak.stock_zh_a_spot_em()
    spot["代码"] = spot["代码"].astype(str)
    spot = spot[spot["代码"].isin(symbols)]
    spot = _rename_with(
        spot,
        {
            "代码": "symbol",
            "名称": "name",
            "最新价": "latest",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "最高": "high",
            "最低": "low",
            "今开": "open",
            "昨收": "prev_close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "市盈率-动态": "pe_dynamic",
            "市盈率": "pe_static",
            "总市值": "total_mv",
            "流通市值": "float_mv",
            "量比": "volume_ratio",
            "内盘": "inner_vol",
            "外盘": "outer_vol",
        },
    )
    for col in ["pe_dynamic", "pe_static", "inner_vol", "outer_vol"]:
        if col not in spot.columns:
            spot[col] = pd.NA
    spot["ts"] = _now_ts()
    cols = [
        "ts",
        "symbol",
        "name",
        "latest",
        "pct_chg",
        "change",
        "high",
        "low",
        "open",
        "prev_close",
        "volume",
        "amount",
        "turnover",
        "pe_dynamic",
        "pe_static",
        "total_mv",
        "float_mv",
        "volume_ratio",
        "inner_vol",
        "outer_vol",
    ]
    for col in cols:
        if col not in spot.columns:
            spot[col] = pd.NA
    return spot[cols]


def _fetch_index_spot(index_symbols: List[str]) -> pd.DataFrame:
    if not index_symbols:
        return pd.DataFrame()
    df = ak.stock_zh_index_spot_em()
    df["代码"] = df["代码"].astype(str)
    df = df[df["代码"].isin(index_symbols)]
    df = _rename_with(
        df,
        {
            "代码": "symbol",
            "名称": "name",
            "最新价": "latest",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "最高": "high",
            "最低": "low",
            "今开": "open",
            "昨收": "prev_close",
            "成交量": "volume",
            "成交额": "amount",
            "量比": "volume_ratio",
        },
    )
    df["ts"] = _now_ts()
    for col in [
        "turnover",
        "pe_dynamic",
        "pe_static",
        "total_mv",
        "float_mv",
        "inner_vol",
        "outer_vol",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    cols = [
        "ts",
        "symbol",
        "name",
        "latest",
        "pct_chg",
        "change",
        "high",
        "low",
        "open",
        "prev_close",
        "volume",
        "amount",
        "turnover",
        "pe_dynamic",
        "pe_static",
        "total_mv",
        "float_mv",
        "volume_ratio",
        "inner_vol",
        "outer_vol",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols]


def _fetch_bid_ask(symbol: str) -> Dict[str, float]:
    df = ak.stock_bid_ask_em(symbol=symbol)
    if df.empty:
        return {}
    return dict(zip(df["item"], df["value"]))


def _fetch_intraday_ticks(symbol: str) -> pd.DataFrame:
    df = ak.stock_intraday_em(symbol=symbol)
    if df.empty:
        return df
    df = _rename_with(
        df,
        {
            "时间": "time",
            "成交价": "price",
            "成交量": "volume",
            "买卖盘性质": "side",
        },
    )
    date_str = datetime.now().strftime("%Y-%m-%d")
    df["datetime"] = date_str + " " + df["time"].astype(str)
    df["symbol"] = symbol
    return df[["symbol", "datetime", "price", "volume", "side"]]


def _fetch_fund_flow(symbol: str, market: str) -> pd.DataFrame:
    df = ak.stock_individual_fund_flow(stock=symbol, market=market)
    if df.empty:
        return df
    latest = df.tail(1).copy()
    latest = _rename_with(
        latest,
        {
            "日期": "date",
            "主力净流入-净额": "main_net",
            "主力净流入-净占比": "main_ratio",
            "超大单净流入-净额": "super_net",
            "超大单净流入-净占比": "super_ratio",
            "大单净流入-净额": "big_net",
            "大单净流入-净占比": "big_ratio",
            "中单净流入-净额": "mid_net",
            "中单净流入-净占比": "mid_ratio",
            "小单净流入-净额": "small_net",
            "小单净流入-净占比": "small_ratio",
        },
    )
    latest["symbol"] = symbol
    cols = [
        "symbol",
        "date",
        "main_net",
        "main_ratio",
        "super_net",
        "super_ratio",
        "big_net",
        "big_ratio",
        "mid_net",
        "mid_ratio",
        "small_net",
        "small_ratio",
    ]
    for col in cols:
        if col not in latest.columns:
            latest[col] = pd.NA
    return latest[cols]


def _fetch_recent_minute(symbol: str, start_dt: str, end_dt: str, period: str = "5") -> pd.DataFrame:
    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol,
            period=period,
            start_date=start_dt,
            end_date=end_dt,
            adjust="",
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = _rename_with(
        df,
        {
            "时间": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover",
        },
    )
    df["symbol"] = symbol
    df["period"] = period
    for col in ["pct_chg", "change", "turnover"]:
        if col not in df.columns:
            df[col] = pd.NA
    cols = [
        "symbol",
        "datetime",
        "period",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "pct_chg",
        "change",
        "turnover",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols]


def _fetch_recent_index_minute(symbol: str, start_dt: str, end_dt: str, period: str = "5") -> pd.DataFrame:
    try:
        df = ak.index_zh_a_hist_min_em(
            symbol=symbol,
            period=period,
            start_date=start_dt,
            end_date=end_dt,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = _rename_with(
        df,
        {
            "时间": "datetime",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover",
        },
    )
    df["symbol"] = symbol
    df["period"] = period
    for col in ["amount", "pct_chg", "change", "turnover"]:
        if col not in df.columns:
            df[col] = pd.NA
    cols = [
        "symbol",
        "datetime",
        "period",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "pct_chg",
        "change",
        "turnover",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols]


def main() -> None:
    init_db()
    symbols, market_map = get_symbols_and_market()
    config = load_config()
    index_symbols = config.get("index_symbols", [])

    last_fund_flow_ts = None
    last_minute_key = None

    while True:
        ts_now = _now_ts()
        with get_conn() as conn:
            try:
                spot_df = _fetch_spot(symbols)
                index_df = _fetch_index_spot(index_symbols)
                if not index_df.empty:
                    spot_df = pd.concat([spot_df, index_df], ignore_index=True)
                upsert_dataframe(conn, "spot_snapshot", spot_df)
            except Exception as exc:
                print(f"实时快照失败: {exc}")

            for symbol in symbols:
                try:
                    ba = _fetch_bid_ask(symbol)
                    if ba:
                        row = {
                            "ts": ts_now,
                            "symbol": symbol,
                            "buy_1": ba.get("buy_1"),
                            "buy_1_vol": ba.get("buy_1_vol"),
                            "buy_2": ba.get("buy_2"),
                            "buy_2_vol": ba.get("buy_2_vol"),
                            "buy_3": ba.get("buy_3"),
                            "buy_3_vol": ba.get("buy_3_vol"),
                            "buy_4": ba.get("buy_4"),
                            "buy_4_vol": ba.get("buy_4_vol"),
                            "buy_5": ba.get("buy_5"),
                            "buy_5_vol": ba.get("buy_5_vol"),
                            "sell_1": ba.get("sell_1"),
                            "sell_1_vol": ba.get("sell_1_vol"),
                            "sell_2": ba.get("sell_2"),
                            "sell_2_vol": ba.get("sell_2_vol"),
                            "sell_3": ba.get("sell_3"),
                            "sell_3_vol": ba.get("sell_3_vol"),
                            "sell_4": ba.get("sell_4"),
                            "sell_4_vol": ba.get("sell_4_vol"),
                            "sell_5": ba.get("sell_5"),
                            "sell_5_vol": ba.get("sell_5_vol"),
                        }
                        upsert_dataframe(conn, "bid_ask", pd.DataFrame([row]))
                except Exception as exc:
                    print(f"{symbol} 买卖盘失败: {exc}")

                try:
                    ticks = _fetch_intraday_ticks(symbol)
                    upsert_dataframe(conn, "ticks", ticks)
                except Exception as exc:
                    print(f"{symbol} 逐笔失败: {exc}")

        now = datetime.now()
        minute_key = now.strftime("%Y-%m-%d %H:%M")
        if now.minute % MINUTE_FETCH_INTERVAL_MIN == 0 and minute_key != last_minute_key:
            start_dt = (now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
            end_dt = now.strftime("%Y-%m-%d %H:%M:%S")
            with get_conn() as conn:
                for symbol in symbols:
                    minute_df = _fetch_recent_minute(symbol, start_dt, end_dt, period="5")
                    if not minute_df.empty:
                        upsert_dataframe(conn, "minute_bars", minute_df)
                for symbol in index_symbols:
                    minute_df = _fetch_recent_index_minute(symbol, start_dt, end_dt, period="5")
                    if not minute_df.empty:
                        upsert_dataframe(conn, "minute_bars", minute_df)
            last_minute_key = minute_key
        if (
            last_fund_flow_ts is None
            or (now - last_fund_flow_ts).total_seconds() >= FUND_FLOW_INTERVAL_MIN * 60
        ):
            with get_conn() as conn:
                for symbol in symbols:
                    market = market_map.get(symbol, "sz")
                    try:
                        fund_df = _fetch_fund_flow(symbol, market)
                        upsert_dataframe(conn, "fund_flow", fund_df)
                    except Exception as exc:
                        print(f"{symbol} 资金流失败: {exc}")
            last_fund_flow_ts = now

        time.sleep(60)


if __name__ == "__main__":
    main()

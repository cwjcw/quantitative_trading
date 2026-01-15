import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from indicators import bollinger_bands, rsi, vwap
from utils import get_symbols_and_market, load_config


ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "monitoring.db"
LOT_SIZE = 100


def _load_strategy_params() -> dict:
    config = load_config()
    params = config.get("strategy_params", {}).copy()
    mode = config.get("market_mode", "range")
    mode_params = config.get("market_mode_params", {}).get(mode, {})
    params.update(mode_params)
    return {
        "market_mode": mode,
        "band_tol": float(params.get("band_tol", 0.005)),
        "vwap_dev_sell": float(params.get("vwap_dev_sell", 0.015)),
        "volume_mult": float(params.get("volume_mult", 1.5)),
        "start_time": params.get("start_time"),
        "end_time": params.get("end_time"),
        "vwap_cross_k": int(params.get("vwap_cross_k", 2)),
        "buy_rsi_threshold": float(params.get("buy_rsi_threshold", 30)),
        "sell_rsi_threshold": float(params.get("sell_rsi_threshold", 70)),
        "trend_rsi_threshold": float(params.get("trend_rsi_threshold", 50)),
        "trend_use_mid": bool(params.get("trend_use_mid", True)),
        "trend_use_vwap": bool(params.get("trend_use_vwap", True)),
        "trend_require_up": bool(params.get("trend_require_up", True)),
        "sell_require_min": int(params.get("sell_require_min", 3)),
        "sell_use_upper": bool(params.get("sell_use_upper", True)),
        "sell_use_rsi": bool(params.get("sell_use_rsi", True)),
        "sell_use_vwap_dev": bool(params.get("sell_use_vwap_dev", True)),
        "sell_use_volume": bool(params.get("sell_use_volume", True)),
        "sell_must_upper": bool(params.get("sell_must_upper", False)),
        "sell_must_rsi": bool(params.get("sell_must_rsi", False)),
        "sell_must_vwap_dev": bool(params.get("sell_must_vwap_dev", False)),
        "sell_must_volume": bool(params.get("sell_must_volume", False)),
    }


def _vwap_cross_up(df: pd.DataFrame, k: int) -> bool:
    if k < 2 or len(df) < k:
        return False
    window = df.iloc[-k:]
    if (window["close"] >= window["vwap"]).any():
        return False
    gaps = (window["vwap"] - window["close"]).values
    closes = window["close"].values
    return all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)) and all(
        closes[i] <= closes[i + 1] for i in range(len(closes) - 1)
    )


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _latest_snapshot(symbols):
    with _get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT * FROM spot_snapshot
            WHERE symbol IN ({})
            AND ts = (SELECT MAX(ts) FROM spot_snapshot)
            """.format(",".join(["?"] * len(symbols))),
            conn,
            params=symbols,
        )
    return df


def _latest_bid_ask(symbols):
    with _get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT * FROM bid_ask
            WHERE symbol IN ({})
            AND ts = (SELECT MAX(ts) FROM bid_ask)
            """.format(",".join(["?"] * len(symbols))),
            conn,
            params=symbols,
        )
    return df


def _load_minute(symbol: str) -> pd.DataFrame:
    with _get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT * FROM minute_bars
            WHERE symbol = ?
            ORDER BY datetime ASC
            """,
            conn,
            params=(symbol,),
        )
    if df.empty:
        return df
    for period in ["1", "5", "15", "30", "60"]:
        if (df["period"] == period).any():
            return df[df["period"] == period].copy()
    return df


def _build_signal(min_df: pd.DataFrame, enable_sell: bool) -> tuple[str, str, str]:
    if min_df.empty or len(min_df) < 30:
        buy_signal = "否"
        sell_signal = "否" if enable_sell else "-"
        reason = "数据不足"
        if not enable_sell:
            reason = "非持仓不监控卖出"
        return buy_signal, sell_signal, reason

    min_df["close"] = pd.to_numeric(min_df["close"], errors="coerce")
    min_df["amount"] = pd.to_numeric(min_df["amount"], errors="coerce")
    min_df["volume"] = pd.to_numeric(min_df["volume"], errors="coerce")
    min_df = min_df.dropna(subset=["close"])

    bands = bollinger_bands(min_df["close"])
    min_df = pd.concat([min_df, bands], axis=1)
    min_df["rsi"] = rsi(min_df["close"], 14)
    min_df["vwap"] = vwap(min_df)
    min_df["vol_ma"] = min_df["volume"].rolling(20).mean()

    if len(min_df) < 2:
        buy_signal = "否"
        sell_signal = "否" if enable_sell else "-"
        reason = "数据不足"
        if not enable_sell:
            reason = "非持仓不监控卖出"
        return buy_signal, sell_signal, reason

    row = min_df.iloc[-1]
    params = _load_strategy_params()
    market_mode = params["market_mode"]
    band_tol = params["band_tol"]
    vwap_dev_sell = params["vwap_dev_sell"]
    volume_mult = params["volume_mult"]
    start_time = params["start_time"]
    end_time = params["end_time"]
    vwap_cross_k = params["vwap_cross_k"]
    buy_rsi_threshold = params["buy_rsi_threshold"]
    sell_rsi_threshold = params["sell_rsi_threshold"]
    trend_rsi_threshold = params["trend_rsi_threshold"]
    trend_use_mid = params["trend_use_mid"]
    trend_use_vwap = params["trend_use_vwap"]
    trend_require_up = params["trend_require_up"]
    sell_require_min = params["sell_require_min"]
    sell_use_upper = params["sell_use_upper"]
    sell_use_rsi = params["sell_use_rsi"]
    sell_use_vwap_dev = params["sell_use_vwap_dev"]
    sell_use_volume = params["sell_use_volume"]
    sell_must_upper = params["sell_must_upper"]
    sell_must_rsi = params["sell_must_rsi"]
    sell_must_vwap_dev = params["sell_must_vwap_dev"]
    sell_must_volume = params["sell_must_volume"]

    time_str = str(row.get("datetime", ""))[-8:]
    if start_time and time_str < start_time:
        return "否", "否" if enable_sell else "-", "未到信号时间"
    if end_time and time_str > end_time:
        return "否", "否" if enable_sell else "-", "已过信号时间"

    near_lower = row["close"] <= row["lower"] * (1 + band_tol)
    rsi_buy = row["rsi"] < buy_rsi_threshold
    vwap_cross = _vwap_cross_up(min_df, vwap_cross_k)
    buy_reasons = []
    if not near_lower:
        buy_reasons.append("未触及下轨")
    if not rsi_buy:
        buy_reasons.append("RSI未超卖")
    if not vwap_cross:
        buy_reasons.append("VWAP未确认上穿")
    buy_signal = near_lower and rsi_buy and vwap_cross

    if market_mode == "bull":
        prev_close = float(min_df.iloc[-2]["close"])
        trend_checks = []
        if trend_use_mid:
            cond_mid = row["close"] >= row["mid"] * (1 - band_tol)
            trend_checks.append(cond_mid)
            if not cond_mid:
                buy_reasons.append("未站上中轨")
        if trend_use_vwap:
            cond_vwap = row["close"] >= row["vwap"]
            trend_checks.append(cond_vwap)
            if not cond_vwap:
                buy_reasons.append("未站上VWAP")
        if trend_require_up:
            cond_up = row["close"] >= prev_close
            trend_checks.append(cond_up)
            if not cond_up:
                buy_reasons.append("未出现抬升")
        cond_trend_rsi = row["rsi"] >= trend_rsi_threshold
        trend_checks.append(cond_trend_rsi)
        if not cond_trend_rsi:
            buy_reasons.append("趋势RSI不足")
        trend_buy = all(trend_checks)
        buy_signal = buy_signal or trend_buy

    sell_signal = False
    sell_reasons = []
    if enable_sell:
        sell_checks = []
        cond_upper = row["close"] >= row["upper"] * (1 - band_tol)
        cond_rsi = row["rsi"] > sell_rsi_threshold
        cond_vwap = row["close"] >= row["vwap"] * (1 + vwap_dev_sell)
        cond_vol = row["volume"] > (row["vol_ma"] * volume_mult) if pd.notna(row["vol_ma"]) else False

        if sell_use_upper:
            sell_checks.append(cond_upper)
        if sell_use_rsi:
            sell_checks.append(cond_rsi)
        if sell_use_vwap_dev:
            sell_checks.append(cond_vwap)
        if sell_use_volume:
            sell_checks.append(cond_vol)

        must_ok = True
        if sell_must_upper:
            must_ok = must_ok and cond_upper
        if sell_must_rsi:
            must_ok = must_ok and cond_rsi
        if sell_must_vwap_dev:
            must_ok = must_ok and cond_vwap
        if sell_must_volume:
            must_ok = must_ok and cond_vol

        if not must_ok:
            sell_reasons.append("未满足必选条件")
        if sell_checks and sum(1 for ok in sell_checks if ok) < sell_require_min:
            sell_reasons.append("卖出条件不足")
        if not sell_checks:
            sell_reasons.append("未启用卖出条件")

        sell_signal = sell_checks and must_ok and sum(1 for ok in sell_checks if ok) >= sell_require_min
    else:
        sell_reasons.append("非持仓不监控卖出")

    reason = "；".join([*buy_reasons, *sell_reasons]) if (not buy_signal and not sell_signal) else ""

    return (
        "是" if buy_signal else "否",
        "是" if sell_signal else ("否" if enable_sell else "-"),
        reason,
    )


def _load_daily(symbol: str, limit: int = 200) -> pd.DataFrame:
    with _get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT * FROM daily_bars
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            conn,
            params=(symbol, limit),
        )
    return df


def _load_intraday_by_date(symbol: str, date_str: str) -> pd.DataFrame:
    with _get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT * FROM minute_bars
            WHERE symbol = ?
              AND datetime LIKE ?
            ORDER BY datetime ASC
            """,
            conn,
            params=(symbol, f"{date_str}%"),
        )
    return df


st.set_page_config(page_title="量化监控系统", layout="wide")
st.title("量化交易监控系统")

config = load_config()
mode_options = list((config.get("market_mode_params") or {}).keys())
mode_default = config.get("market_mode", "range")
if mode_default not in mode_options and mode_options:
    mode_default = mode_options[0]
mode_labels = {
    "bull": "牛市",
    "bear": "熊市",
    "range": "震荡",
}
mode_display = [mode_labels.get(m, m) for m in mode_options]
mode_index = mode_options.index(mode_default) if mode_default in mode_options else 0
pick_label = st.selectbox("市场模式", mode_display, index=mode_index)
label_to_mode = {v: k for k, v in mode_labels.items()}
pick_mode = label_to_mode.get(pick_label, mode_default)
config["market_mode"] = pick_mode
symbols, _ = get_symbols_and_market()
positions = set(config.get("positions", []) or [])

snapshot = _latest_snapshot(symbols)
bid_ask = _latest_bid_ask(symbols)

if snapshot.empty:
    st.warning("暂无实时快照数据，请先运行实时采集脚本。")
    st.stop()

df = snapshot.merge(bid_ask, on=["symbol"], how="left", suffixes=("", "_ba"))
df["买入信号"] = ""
df["卖出信号"] = ""
df["未触发原因"] = ""
for i, row in df.iterrows():
    min_df = _load_minute(row["symbol"])
    enable_sell = row["symbol"] in positions
    buy_sig, sell_sig, reason = _build_signal(min_df, enable_sell)
    df.at[i, "买入信号"] = buy_sig
    df.at[i, "卖出信号"] = sell_sig
    df.at[i, "未触发原因"] = reason

display_cols = {
    "symbol": "代码",
    "name": "名称",
    "latest": "最新价",
    "pct_chg": "涨跌幅",
    "change": "涨跌额",
    "high": "最高",
    "low": "最低",
    "open": "今开",
    "prev_close": "昨收",
    "volume": "成交量(手)",
    "amount": "成交额(亿元)",
    "turnover": "换手率",
    "volume_ratio": "量比",
    "buy_1": "买一价",
    "sell_1": "卖一价",
    "buy_1_vol": "买一量",
    "sell_1_vol": "卖一量",
    "买入信号": "买入信号",
    "卖出信号": "卖出信号",
    "未触发原因": "未触发原因",
}

show_df = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)
if "成交额(亿元)" in show_df.columns:
    show_df["成交额(亿元)"] = pd.to_numeric(show_df["成交额(亿元)"], errors="coerce") / 1e8
st.dataframe(show_df, use_container_width=True, height=700)

st.markdown("---")
st.subheader("历史日线数据")
pick_symbol = st.selectbox("选择股票", symbols, index=0)
daily_df = _load_daily(pick_symbol, limit=200)
if daily_df.empty:
    st.warning("日线数据为空，请先抓取历史数据。")
else:
    daily_show = daily_df.rename(
        columns={
            "date": "日期",
            "open": "开盘",
            "close": "收盘",
            "high": "最高",
            "low": "最低",
            "volume": "成交量(手)",
            "amount": "成交额(亿元)",
            "pct_chg": "涨跌幅",
            "change": "涨跌额",
            "turnover": "换手率",
        }
    )
    daily_show["成交额(亿元)"] = pd.to_numeric(daily_show["成交额(亿元)"], errors="coerce") / 1e8
    st.dataframe(daily_show, use_container_width=True, height=300)

st.subheader("指定日期分钟线折线图")
default_date = datetime.now() - timedelta(days=1)
selected_date = st.date_input("选择日期", value=default_date.date())
date_str = selected_date.strftime("%Y-%m-%d")
minute_df = _load_intraday_by_date(pick_symbol, date_str)
if minute_df.empty:
    st.warning("分钟线数据为空，请先抓取历史数据或更换日期。")
else:
    minute_df["datetime"] = pd.to_datetime(minute_df["datetime"])
    minute_df["close"] = pd.to_numeric(minute_df["close"], errors="coerce")
    st.line_chart(minute_df.set_index("datetime")[["close"]])

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import sqlite3
import akshare as ak

from indicators import bollinger_bands, rsi, vwap
from utils import get_symbols_and_market, load_config


ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "monitoring.db"
LOT_SIZE = 100


@dataclass
class Trade:
    symbol: str
    name: str
    action: str
    datetime: str
    price: float
    shares: int
    cash: float
    fee: float


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


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
            df = df[df["period"] == period].copy()
            break

    return df


def _load_daily(symbol: str) -> pd.DataFrame:
    with _get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT * FROM daily_bars
            WHERE symbol = ?
            ORDER BY date ASC
            """,
            conn,
            params=(symbol,),
        )
    return df


def _limit_rate(symbol: str) -> float:
    if symbol.startswith(("300", "688", "301", "689")):
        return 0.2
    return 0.1


def _build_daily_map(daily_df: pd.DataFrame) -> Dict[str, float]:
    if daily_df.empty:
        return {}
    daily_df = daily_df.copy()
    daily_df["prev_close"] = daily_df["close"].shift(1)
    return dict(zip(daily_df["date"], daily_df["prev_close"]))


def _apply_fee(price: float, shares: int, commission_rate: float, stamp_rate: float, sell: bool) -> float:
    amount = price * shares * LOT_SIZE
    fee = amount * commission_rate
    if sell:
        fee += amount * stamp_rate
    return fee


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
        "max_pos_pct": float(params.get("max_pos_pct", 0.2)),
        "cash_reserve_pct": float(params.get("cash_reserve_pct", 0.2)),
        "start_time": params.get("start_time"),
        "end_time": params.get("end_time"),
        "vwap_cross_k": int(params.get("vwap_cross_k", 2)),
        "buy_rsi_threshold": float(params.get("buy_rsi_threshold", 30)),
        "sell_rsi_threshold": float(params.get("sell_rsi_threshold", 70)),
        "trend_rsi_threshold": float(params.get("trend_rsi_threshold", 50)),
        "trend_use_mid": bool(params.get("trend_use_mid", True)),
        "trend_use_vwap": bool(params.get("trend_use_vwap", True)),
        "trend_require_up": bool(params.get("trend_require_up", True)),
        "buy_position_ratio": float(params.get("buy_position_ratio", 0.5)),
        "add_position_ratio": float(params.get("add_position_ratio", 0.5)),
        "sell_require_min": int(params.get("sell_require_min", 2)),
        "sell_use_upper": bool(params.get("sell_use_upper", True)),
        "sell_use_rsi": bool(params.get("sell_use_rsi", True)),
        "sell_use_vwap_dev": bool(params.get("sell_use_vwap_dev", True)),
        "sell_use_volume": bool(params.get("sell_use_volume", True)),
        "sell_must_upper": bool(params.get("sell_must_upper", False)),
        "sell_must_rsi": bool(params.get("sell_must_rsi", False)),
        "sell_must_vwap_dev": bool(params.get("sell_must_vwap_dev", False)),
        "sell_must_volume": bool(params.get("sell_must_volume", False)),
        "take_profit_pct": float(params.get("take_profit_pct", 0.05)),
        "stop_loss_pct": float(params.get("stop_loss_pct", 0.03)),
    }


def _vwap_cross_up(df: pd.DataFrame, idx: int, k: int) -> bool:
    if idx - k + 1 < 0 or k < 2:
        return False
    window = df.iloc[idx - k + 1 : idx + 1]
    if (window["close"] >= window["vwap"]).any():
        return False
    gaps = (window["vwap"] - window["close"]).values
    closes = window["close"].values
    return all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)) and all(
        closes[i] <= closes[i + 1] for i in range(len(closes) - 1)
    )


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max
    return float(drawdown.min())


def _load_symbol_name_map(symbols: List[str]) -> dict:
    try:
        spot = ak.stock_zh_a_spot_em()
        spot["代码"] = spot["代码"].astype(str)
        mapping = dict(zip(spot["代码"], spot["名称"]))
        return {s: mapping.get(s, s) for s in symbols}
    except Exception:
        return {s: s for s in symbols}


def backtest_portfolio(
    symbols: List[str],
    max_pos_pct_override: float | None = None,
    cash_reserve_pct_override: float | None = None,
    buy_ratio_override: float | None = None,
) -> Tuple[List[Trade], pd.DataFrame, List[float], str, str]:
    config = load_config()
    init_cash = float(config["initial_cash"])
    commission_rate = float(config["commission_rate"])
    stamp_rate = float(config["stamp_duty_rate"])
    stamp_on_sell = bool(config.get("stamp_duty_on_sell", True))
    priority_map = config.get("priority_map", {})
    name_map = _load_symbol_name_map(symbols)

    params = _load_strategy_params()
    market_mode = params["market_mode"]
    band_tol = params["band_tol"]
    vwap_dev_sell = params["vwap_dev_sell"]
    volume_mult = params["volume_mult"]
    buy_rsi_threshold = params["buy_rsi_threshold"]
    sell_rsi_threshold = params["sell_rsi_threshold"]
    trend_rsi_threshold = params["trend_rsi_threshold"]
    trend_use_mid = params["trend_use_mid"]
    trend_use_vwap = params["trend_use_vwap"]
    trend_require_up = params["trend_require_up"]
    buy_position_ratio = params["buy_position_ratio"]
    add_position_ratio = params["add_position_ratio"]
    max_pos_pct = max_pos_pct_override if max_pos_pct_override is not None else params["max_pos_pct"]
    cash_reserve_pct = (
        cash_reserve_pct_override if cash_reserve_pct_override is not None else params["cash_reserve_pct"]
    )
    buy_ratio = buy_ratio_override if buy_ratio_override is not None else 0.5
    start_time = params["start_time"]
    end_time = params["end_time"]
    vwap_cross_k = params["vwap_cross_k"]
    sell_require_min = params["sell_require_min"]
    sell_use_upper = params["sell_use_upper"]
    sell_use_rsi = params["sell_use_rsi"]
    sell_use_vwap_dev = params["sell_use_vwap_dev"]
    sell_use_volume = params["sell_use_volume"]
    sell_must_upper = params["sell_must_upper"]
    sell_must_rsi = params["sell_must_rsi"]
    sell_must_vwap_dev = params["sell_must_vwap_dev"]
    sell_must_volume = params["sell_must_volume"]
    take_profit_pct = params["take_profit_pct"]
    stop_loss_pct = params["stop_loss_pct"]

    data_map: Dict[str, pd.DataFrame] = {}
    daily_map: Dict[str, Dict[str, float]] = {}
    index_map: Dict[str, Dict[str, int]] = {}
    for symbol in symbols:
        minute_df = _load_minute(symbol)
        daily_df = _load_daily(symbol)
        if minute_df.empty or daily_df.empty:
            continue
        daily_map[symbol] = _build_daily_map(daily_df)
        minute_df["date"] = minute_df["datetime"].str.slice(0, 10)
        minute_df["close"] = pd.to_numeric(minute_df["close"], errors="coerce")
        minute_df["amount"] = pd.to_numeric(minute_df["amount"], errors="coerce")
        minute_df["volume"] = pd.to_numeric(minute_df["volume"], errors="coerce")
        minute_df = minute_df.dropna(subset=["close"])
        bands = bollinger_bands(minute_df["close"])
        minute_df = pd.concat([minute_df, bands], axis=1)
        minute_df["rsi"] = rsi(minute_df["close"], 14)
        minute_df["vwap"] = vwap(minute_df)
        minute_df["vol_ma"] = minute_df["volume"].rolling(20).mean()
        minute_df = minute_df.reset_index(drop=True)
        data_map[symbol] = minute_df
        index_map[symbol] = {dt: i for i, dt in enumerate(minute_df["datetime"])}

    if not data_map:
        return [], pd.DataFrame(), [], "", ""

    all_dates = sorted({d for df in data_map.values() for d in df["date"].unique()})
    start_day = all_dates[0]
    last_day = all_dates[-1]
    all_times = sorted({t for df in data_map.values() for t in df["datetime"].unique()})

    cash = init_cash
    positions = {s: 0 for s in symbols}
    last_buy_date = {s: None for s in symbols}
    open_lots = {s: [] for s in symbols}
    last_price = {s: None for s in symbols}
    trades: List[Trade] = []
    pnl_rows: List[float] = []
    equity_rows: List[dict] = []

    for current_time in all_times:
        # sell first
        for symbol, df in data_map.items():
            idx = index_map[symbol].get(current_time)
            if idx is None or idx == 0:
                continue
            row = df.iloc[idx]
            price = float(row["close"])
            last_price[symbol] = price
            date_key = row["date"]
            time_str = row["datetime"][-8:]

            if start_time and time_str < start_time:
                continue
            if end_time and time_str > end_time:
                continue
            if positions[symbol] <= 0:
                continue
            if last_buy_date[symbol] == date_key:
                continue

            prev_close = daily_map[symbol].get(date_key)
            limit_down = prev_close * (1 - _limit_rate(symbol)) if prev_close else None

            cond_upper = price >= row["upper"] * (1 - band_tol)
            cond_rsi = row["rsi"] > sell_rsi_threshold
            cond_vwap = price >= row["vwap"] * (1 + vwap_dev_sell)
            cond_vol = row["volume"] > (row["vol_ma"] * volume_mult) if pd.notna(row["vol_ma"]) else False

            sell_checks = []
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

            sell_signal = (
                must_ok and (sum(1 for ok in sell_checks if ok) >= sell_require_min)
                if sell_checks
                else False
            )

            take_profit = False
            stop_loss = False
            if open_lots[symbol]:
                entry_price = open_lots[symbol][0]["price"]
                take_profit = price >= entry_price * (1 + take_profit_pct)
                stop_loss = price <= entry_price * (1 - stop_loss_pct)

            if sell_signal or take_profit or stop_loss:
                if limit_down and price <= limit_down:
                    continue
                sell_shares = positions[symbol]
                fee = _apply_fee(price, sell_shares, commission_rate, stamp_rate, stamp_on_sell)
                cash += price * sell_shares * LOT_SIZE - fee
                trades.append(Trade(symbol, name_map.get(symbol, symbol), "SELL", row["datetime"], price, sell_shares, cash, fee))
                positions[symbol] = 0

                left = sell_shares
                while left > 0 and open_lots[symbol]:
                    lot = open_lots[symbol][0]
                    take = min(lot["shares"], left)
                    pnl_rows.append((price - lot["price"]) * take * LOT_SIZE - lot["fee"] - fee)
                    lot["shares"] -= take
                    left -= take
                    if lot["shares"] == 0:
                        open_lots[symbol].pop(0)

        # buy next
        buy_candidates = []
        for symbol, df in data_map.items():
            idx = index_map[symbol].get(current_time)
            if idx is None or idx == 0:
                continue
            row = df.iloc[idx]
            price = float(row["close"])
            last_price[symbol] = price
            date_key = row["date"]
            time_str = row["datetime"][-8:]

            if date_key == last_day:
                continue
            if start_time and time_str < start_time:
                continue
            if end_time and time_str > end_time:
                continue

            prev_close = daily_map[symbol].get(date_key)
            limit_up = prev_close * (1 + _limit_rate(symbol)) if prev_close else None

            near_lower = price <= row["lower"] * (1 + band_tol)
            rsi_buy = row["rsi"] < buy_rsi_threshold
            vwap_cross = _vwap_cross_up(df, idx, vwap_cross_k)
            buy_signal = near_lower and rsi_buy and vwap_cross
            trend_buy = False

            if market_mode == "bull":
                prev_close_min = float(df.iloc[idx - 1]["close"])
                trend_checks = []
                if trend_use_mid:
                    trend_checks.append(price >= row["mid"] * (1 - band_tol))
                if trend_use_vwap:
                    trend_checks.append(price >= row["vwap"])
                if trend_require_up:
                    trend_checks.append(price >= prev_close_min)
                trend_checks.append(row["rsi"] >= trend_rsi_threshold)
                trend_buy = all(trend_checks)
                buy_signal = buy_signal or trend_buy

            if positions[symbol] > 0:
                if market_mode != "bull":
                    continue
                if last_buy_date[symbol] == date_key:
                    continue
                if not trend_buy:
                    continue
                if limit_up and price >= limit_up:
                    continue
                max_position_value = init_cash * max_pos_pct
                min_cash_reserve = init_cash * cash_reserve_pct
                current_value = positions[symbol] * price * LOT_SIZE
                if current_value >= max_position_value * 0.99:
                    continue
                add_target = max_position_value * add_position_ratio
                remaining_value = max_position_value - current_value
                add_value = min(add_target, remaining_value)
                shares = int(add_value // (price * LOT_SIZE))
                if shares <= 0:
                    continue
                fee = _apply_fee(price, shares, commission_rate, stamp_rate, False)
                if (price * shares * LOT_SIZE) + fee > cash - min_cash_reserve:
                    continue
                cost = price * shares * LOT_SIZE + fee
                cash -= cost
                positions[symbol] += shares
                last_buy_date[symbol] = date_key
                open_lots[symbol].append({"price": price, "shares": shares, "fee": fee})
                trades.append(Trade(symbol, name_map.get(symbol, symbol), "BUY_ADD", row["datetime"], price, shares, cash, fee))
                continue
            if buy_signal and (not limit_up or price < limit_up):
                prio = int(priority_map.get(symbol, 1))
                buy_candidates.append((prio, symbol, price, date_key, row["datetime"]))

        buy_candidates.sort(key=lambda x: x[0])
        for prio, symbol, price, date_key, dt_str in buy_candidates:
            max_position_value = init_cash * max_pos_pct
            min_cash_reserve = init_cash * cash_reserve_pct
            if cash <= min_cash_reserve:
                continue
            shares = int((max_position_value * buy_ratio * buy_position_ratio) // (price * LOT_SIZE))
            if shares <= 0:
                continue
            fee = _apply_fee(price, shares, commission_rate, stamp_rate, False)
            if (price * shares * LOT_SIZE) + fee > cash - min_cash_reserve:
                continue
            cost = price * shares * LOT_SIZE + fee
            cash -= cost
            positions[symbol] = shares
            last_buy_date[symbol] = date_key
            open_lots[symbol].append({"price": price, "shares": shares, "fee": fee})
            trades.append(Trade(symbol, name_map.get(symbol, symbol), "BUY", dt_str, price, shares, cash, fee))

        equity = cash
        for symbol in symbols:
            if positions[symbol] <= 0:
                continue
            price = last_price.get(symbol)
            if price is None:
                continue
            equity += positions[symbol] * price * LOT_SIZE
        equity_rows.append({"datetime": current_time, "equity": equity})

    # force close on last day
    for symbol, df in data_map.items():
        if positions[symbol] <= 0:
            continue
        last_rows = df[df["date"] == last_day]
        if last_rows.empty:
            continue
        row = last_rows.iloc[-1]
        price = float(row["close"])
        fee = _apply_fee(price, positions[symbol], commission_rate, stamp_rate, stamp_on_sell)
        cash += price * positions[symbol] * LOT_SIZE - fee
        trades.append(Trade(symbol, name_map.get(symbol, symbol), "SELL_FORCE", row["datetime"], price, positions[symbol], cash, fee))
        positions[symbol] = 0
        open_lots[symbol].clear()

    equity_df = pd.DataFrame(equity_rows)
    return trades, equity_df, pnl_rows, start_day, last_day


def _build_pnl_summary(trades: List[Trade], start_day: str, last_day: str) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    pnl_by_symbol = []
    for symbol in trades_df["symbol"].unique():
        sym_df = trades_df[trades_df["symbol"] == symbol].sort_values("datetime")
        open_lots = []
        pnl_rows_sym = []
        buy_cost_total = 0.0
        for _, row in sym_df.iterrows():
            if row["action"] in ("BUY", "BUY_ADD"):
                buy_cost_total += float(row["price"]) * int(row["shares"]) * LOT_SIZE + float(row["fee"])
                open_lots.append(row)
            else:
                left = int(row["shares"])
                sell_price = float(row["price"])
                sell_fee = float(row["fee"])
                while left > 0 and open_lots:
                    buy = open_lots[0]
                    lot_shares = int(buy["shares"])
                    take = min(lot_shares, left)
                    buy_price = float(buy["price"])
                    buy_fee = float(buy["fee"])
                    pnl_rows_sym.append((sell_price - buy_price) * take * LOT_SIZE - buy_fee - sell_fee)
                    if take == lot_shares:
                        open_lots.pop(0)
                    else:
                        buy["shares"] = lot_shares - take
                    left -= take
        total_pnl = float(pd.Series(pnl_rows_sym).sum()) if pnl_rows_sym else 0.0
        pnl_pct = (total_pnl / buy_cost_total) if buy_cost_total > 0 else 0.0
        name = sym_df["name"].iloc[0] if not sym_df["name"].isna().all() else symbol

        with _get_conn() as conn:
            daily_df = pd.read_sql_query(
                """
                SELECT date, open, close FROM daily_bars
                WHERE symbol = ? AND date IN (?, ?)
                """,
                conn,
                params=(symbol, start_day, last_day),
            )
        start_open = None
        end_close = None
        if not daily_df.empty:
            start_open_row = daily_df[daily_df["date"] == start_day]
            end_close_row = daily_df[daily_df["date"] == last_day]
            if not start_open_row.empty:
                start_open = float(start_open_row.iloc[0]["open"])
            if not end_close_row.empty:
                end_close = float(end_close_row.iloc[0]["close"])
        price_pct = ((end_close - start_open) / start_open) if start_open and end_close else 0.0

        pnl_by_symbol.append(
            {
                "symbol": symbol,
                "name": name,
                "total_pnl": total_pnl,
                "pnl_pct": pnl_pct * 100,
                "start_open": start_open,
                "end_close": end_close,
                "price_pct": price_pct * 100,
            }
        )

    metrics_df = pd.DataFrame(pnl_by_symbol)

    config = load_config()
    index_symbols = config.get("index_symbols", ["000001"])
    index_symbol = index_symbols[0] if index_symbols else "000001"
    with _get_conn() as conn:
        idx_df = pd.read_sql_query(
            """
            SELECT date, open, close FROM daily_bars
            WHERE symbol = ? AND date IN (?, ?)
            """,
            conn,
            params=(index_symbol, start_day, last_day),
        )
    idx_start_open = None
    idx_end_close = None
    if not idx_df.empty:
        idx_start_row = idx_df[idx_df["date"] == start_day]
        idx_end_row = idx_df[idx_df["date"] == last_day]
        if not idx_start_row.empty:
            idx_start_open = float(idx_start_row.iloc[0]["open"])
        if not idx_end_row.empty:
            idx_end_close = float(idx_end_row.iloc[0]["close"])
    idx_pct = ((idx_end_close - idx_start_open) / idx_start_open) if idx_start_open and idx_end_close else 0.0

    metrics_df["index_start_open"] = idx_start_open
    metrics_df["index_end_close"] = idx_end_close
    metrics_df["index_price_pct"] = idx_pct * 100

    for col in [
        "total_pnl",
        "pnl_pct",
        "start_open",
        "end_close",
        "price_pct",
        "index_start_open",
        "index_end_close",
        "index_price_pct",
    ]:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].astype(float).round(2)

    total_row = pd.DataFrame(
        [
            {
                "symbol": "TOTAL",
                "name": "TOTAL",
                "total_pnl": float(metrics_df["total_pnl"].sum()),
                "pnl_pct": None,
                "start_open": None,
                "end_close": None,
                "price_pct": None,
                "index_start_open": idx_start_open,
                "index_end_close": idx_end_close,
                "index_price_pct": idx_pct * 100 if idx_start_open and idx_end_close else None,
            }
        ]
    )
    metrics_df = pd.concat([metrics_df, total_row], ignore_index=True)
    return metrics_df


def _build_empty_summary(symbol: str, start_day: str, last_day: str) -> pd.DataFrame:
    name_map = _load_symbol_name_map([symbol])
    name = name_map.get(symbol, symbol)

    with _get_conn() as conn:
        daily_df = pd.read_sql_query(
            """
            SELECT date, open, close FROM daily_bars
            WHERE symbol = ? AND date IN (?, ?)
            """,
            conn,
            params=(symbol, start_day, last_day),
        )
    start_open = None
    end_close = None
    if not daily_df.empty:
        start_open_row = daily_df[daily_df["date"] == start_day]
        end_close_row = daily_df[daily_df["date"] == last_day]
        if not start_open_row.empty:
            start_open = float(start_open_row.iloc[0]["open"])
        if not end_close_row.empty:
            end_close = float(end_close_row.iloc[0]["close"])
    price_pct = ((end_close - start_open) / start_open) if start_open and end_close else 0.0

    config = load_config()
    index_symbols = config.get("index_symbols", ["000001"])
    index_symbol = index_symbols[0] if index_symbols else "000001"
    with _get_conn() as conn:
        idx_df = pd.read_sql_query(
            """
            SELECT date, open, close FROM daily_bars
            WHERE symbol = ? AND date IN (?, ?)
            """,
            conn,
            params=(index_symbol, start_day, last_day),
        )
    idx_start_open = None
    idx_end_close = None
    if not idx_df.empty:
        idx_start_row = idx_df[idx_df["date"] == start_day]
        idx_end_row = idx_df[idx_df["date"] == last_day]
        if not idx_start_row.empty:
            idx_start_open = float(idx_start_row.iloc[0]["open"])
        if not idx_end_row.empty:
            idx_end_close = float(idx_end_row.iloc[0]["close"])
    idx_pct = ((idx_end_close - idx_start_open) / idx_start_open) if idx_start_open and idx_end_close else 0.0

    row = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "name": name,
                "total_pnl": 0.0,
                "pnl_pct": 0.0,
                "start_open": start_open,
                "end_close": end_close,
                "price_pct": price_pct * 100,
                "index_start_open": idx_start_open,
                "index_end_close": idx_end_close,
                "index_price_pct": idx_pct * 100,
            }
        ]
    )
    for col in [
        "total_pnl",
        "pnl_pct",
        "start_open",
        "end_close",
        "price_pct",
        "index_start_open",
        "index_end_close",
        "index_price_pct",
    ]:
        if col in row.columns:
            row[col] = row[col].astype(float).round(2)
    return row


def run_portfolio_backtest() -> None:
    symbols, _ = get_symbols_and_market()
    params = _load_strategy_params()
    mode = params.get("market_mode", "range")
    print(f"市场模式: {mode} | 生效参数: {params}")
    trades, equity_df, pnl_rows, start_day, last_day = backtest_portfolio(symbols)

    if trades:
        df = pd.DataFrame([t.__dict__ for t in trades])
        for col in ["price", "cash", "fee"]:
            df[col] = df[col].astype(float).round(2)
        out_path = ROOT_DIR / "data" / "backtest_trades.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Trades saved: {out_path}")
    else:
        print("No trades.")

    if trades:
        metrics_df = _build_pnl_summary(trades, start_day, last_day)
        pnl_path = ROOT_DIR / "data" / "backtest_pnl_by_symbol.csv"
        metrics_df.to_csv(pnl_path, index=False, encoding="utf-8-sig")
        print(f"PNL by symbol saved: {pnl_path}")

    if pnl_rows:
        pnl_series = pd.Series(pnl_rows)
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
        win_rate = float((pnl_series > 0).mean()) * 100
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        total_pnl = float(pnl_series.sum())
        max_dd = _max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0
        summary_df = pd.DataFrame(
            [
                {
                    "win_rate": round(win_rate, 2),
                    "profit_loss_ratio": round(pl_ratio, 2),
                    "total_pnl": round(total_pnl, 2),
                    "max_drawdown": round(max_dd, 4),
                }
            ]
        )
        summary_path = ROOT_DIR / "data" / "backtest_metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Summary metrics saved: {summary_path}")


def run_single_full_backtest() -> None:
    symbols, _ = get_symbols_and_market()
    single_rows = []
    for symbol in symbols:
        trades_s, _, _, start_s, last_s = backtest_portfolio(
            [symbol],
            max_pos_pct_override=1.0,
            cash_reserve_pct_override=0.0,
            buy_ratio_override=1.0,
        )
        if not start_s or not last_s:
            continue
        if trades_s:
            summary_df = _build_pnl_summary(trades_s, start_s, last_s)
            if summary_df.empty:
                continue
            summary_df = summary_df[summary_df["symbol"] != "TOTAL"]
            single_rows.append(summary_df)
        else:
            single_rows.append(_build_empty_summary(symbol, start_s, last_s))

    if single_rows:
        single_df = pd.concat(single_rows, ignore_index=True)
        single_path = ROOT_DIR / "data" / "backtest_single_full_summary.csv"
        single_df.to_csv(single_path, index=False, encoding="utf-8-sig")
        print(f"Single full summary saved: {single_path}")


if __name__ == "__main__":
    run_portfolio_backtest()
    run_single_full_backtest()

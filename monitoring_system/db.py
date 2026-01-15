from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "monitoring.db"


def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_bars (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                close REAL,
                high REAL,
                low REAL,
                volume REAL,
                amount REAL,
                pct_chg REAL,
                change REAL,
                turnover REAL,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS minute_bars (
                symbol TEXT NOT NULL,
                datetime TEXT NOT NULL,
                period TEXT NOT NULL,
                open REAL,
                close REAL,
                high REAL,
                low REAL,
                volume REAL,
                amount REAL,
                pct_chg REAL,
                change REAL,
                turnover REAL,
                PRIMARY KEY (symbol, datetime, period)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spot_snapshot (
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT,
                latest REAL,
                pct_chg REAL,
                change REAL,
                high REAL,
                low REAL,
                open REAL,
                prev_close REAL,
                volume REAL,
                amount REAL,
                turnover REAL,
                pe_dynamic REAL,
                pe_static REAL,
                total_mv REAL,
                float_mv REAL,
                volume_ratio REAL,
                inner_vol REAL,
                outer_vol REAL,
                PRIMARY KEY (ts, symbol)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bid_ask (
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                buy_1 REAL,
                buy_1_vol REAL,
                buy_2 REAL,
                buy_2_vol REAL,
                buy_3 REAL,
                buy_3_vol REAL,
                buy_4 REAL,
                buy_4_vol REAL,
                buy_5 REAL,
                buy_5_vol REAL,
                sell_1 REAL,
                sell_1_vol REAL,
                sell_2 REAL,
                sell_2_vol REAL,
                sell_3 REAL,
                sell_3_vol REAL,
                sell_4 REAL,
                sell_4_vol REAL,
                sell_5 REAL,
                sell_5_vol REAL,
                PRIMARY KEY (ts, symbol)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                symbol TEXT NOT NULL,
                datetime TEXT NOT NULL,
                price REAL,
                volume REAL,
                side TEXT,
                PRIMARY KEY (symbol, datetime, price, volume)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fund_flow (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                main_net REAL,
                main_ratio REAL,
                super_net REAL,
                super_ratio REAL,
                big_net REAL,
                big_ratio REAL,
                mid_net REAL,
                mid_ratio REAL,
                small_net REAL,
                small_ratio REAL,
                PRIMARY KEY (symbol, date)
            )
            """
        )

        conn.execute("DROP VIEW IF EXISTS v_daily_bars_cn")
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_daily_bars_cn AS
            SELECT
                symbol AS 代码,
                date AS 日期,
                open AS 开盘,
                close AS 收盘,
                high AS 最高,
                low AS 最低,
                volume AS 成交量,
                amount AS 成交额,
                pct_chg AS 涨跌幅,
                change AS 涨跌额,
                turnover AS 换手率
            FROM daily_bars
            """
        )
        conn.execute("DROP VIEW IF EXISTS v_minute_bars_cn")
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_minute_bars_cn AS
            SELECT
                symbol AS 代码,
                datetime AS 时间,
                period AS 周期,
                open AS 开盘,
                close AS 收盘,
                high AS 最高,
                low AS 最低,
                volume AS 成交量,
                amount AS 成交额,
                pct_chg AS 涨跌幅,
                change AS 涨跌额,
                turnover AS 换手率
            FROM minute_bars
            """
        )
        conn.execute("DROP VIEW IF EXISTS v_spot_snapshot_cn")
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_spot_snapshot_cn AS
            SELECT
                ts AS 时间,
                symbol AS 代码,
                name AS 名称,
                latest AS 最新价,
                pct_chg AS 涨跌幅,
                change AS 涨跌额,
                high AS 最高,
                low AS 最低,
                open AS 今开,
                prev_close AS 昨收,
                volume AS 成交量,
                amount AS 成交额,
                turnover AS 换手率,
                pe_dynamic AS 市盈率_动态,
                pe_static AS 市盈率_静态,
                total_mv AS 总市值,
                float_mv AS 流通市值,
                volume_ratio AS 量比,
                inner_vol AS 内盘,
                outer_vol AS 外盘
            FROM spot_snapshot
            """
        )
        conn.execute("DROP VIEW IF EXISTS v_bid_ask_cn")
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_bid_ask_cn AS
            SELECT
                ts AS 时间,
                symbol AS 代码,
                buy_1 AS 买一价,
                buy_1_vol AS 买一量,
                buy_2 AS 买二价,
                buy_2_vol AS 买二量,
                buy_3 AS 买三价,
                buy_3_vol AS 买三量,
                buy_4 AS 买四价,
                buy_4_vol AS 买四量,
                buy_5 AS 买五价,
                buy_5_vol AS 买五量,
                sell_1 AS 卖一价,
                sell_1_vol AS 卖一量,
                sell_2 AS 卖二价,
                sell_2_vol AS 卖二量,
                sell_3 AS 卖三价,
                sell_3_vol AS 卖三量,
                sell_4 AS 卖四价,
                sell_4_vol AS 卖四量,
                sell_5 AS 卖五价,
                sell_5_vol AS 卖五量
            FROM bid_ask
            """
        )
        conn.execute("DROP VIEW IF EXISTS v_ticks_cn")
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_ticks_cn AS
            SELECT
                symbol AS 代码,
                datetime AS 时间,
                price AS 成交价,
                volume AS 成交量,
                side AS 成交方向
            FROM ticks
            """
        )
        conn.execute("DROP VIEW IF EXISTS v_fund_flow_cn")
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_fund_flow_cn AS
            SELECT
                symbol AS 代码,
                date AS 日期,
                main_net AS 主力净流入,
                main_ratio AS 主力净占比,
                super_net AS 超大单净流入,
                super_ratio AS 超大单净占比,
                big_net AS 大单净流入,
                big_ratio AS 大单净占比,
                mid_net AS 中单净流入,
                mid_ratio AS 中单净占比,
                small_net AS 小单净流入,
                small_ratio AS 小单净占比
            FROM fund_flow
            """
        )
        conn.commit()


def upsert_dataframe(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.where(pd.notna(df), None)
    cols = df.columns.tolist()
    placeholders = ",".join(["?"] * len(cols))
    col_sql = ",".join(cols)
    sql = f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})"
    conn.executemany(sql, df.itertuples(index=False, name=None))
    conn.commit()
    return len(df)

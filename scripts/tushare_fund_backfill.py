from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import tushare as ts
from sqlalchemy import create_engine, text

from quantitative_trading.config import build_database_url_from_parts, load_env_file


DDL = """
CREATE TABLE IF NOT EXISTS public.tushare_fund_records (
    endpoint text NOT NULL,
    date_key text NOT NULL DEFAULT '',
    date_type text NOT NULL DEFAULT '',
    ts_code text NOT NULL DEFAULT '',
    entity_key text NOT NULL DEFAULT '',
    market text NOT NULL DEFAULT '',
    content_type text NOT NULL DEFAULT '',
    row_hash text NOT NULL,
    fetched_at timestamptz NOT NULL,
    raw jsonb NOT NULL,
    PRIMARY KEY (endpoint, date_key, ts_code, entity_key, market, content_type, row_hash)
);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_records_endpoint_date
    ON public.tushare_fund_records (endpoint, date_key);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_records_ts_code
    ON public.tushare_fund_records (ts_code);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_records_entity_key
    ON public.tushare_fund_records (entity_key);

CREATE TABLE IF NOT EXISTS public.tushare_fund_basic (
    ts_code text PRIMARY KEY,
    name text,
    management text,
    custodian text,
    fund_type text,
    found_date date,
    due_date date,
    list_date date,
    issue_date date,
    delist_date date,
    issue_amount numeric,
    m_fee numeric,
    c_fee numeric,
    duration_year numeric,
    p_value numeric,
    min_amount numeric,
    exp_return text,
    benchmark text,
    status text,
    invest_type text,
    type text,
    trustee text,
    purc_startdate date,
    redm_startdate date,
    market text,
    fetched_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_basic_market_type
    ON public.tushare_fund_basic (market, fund_type, status);

CREATE TABLE IF NOT EXISTS public.tushare_fund_company (
    entity_key text PRIMARY KEY,
    name text,
    shortname text,
    province text,
    city text,
    address text,
    phone text,
    office text,
    website text,
    chairman text,
    manager text,
    reg_capital numeric,
    setup_date date,
    end_date date,
    employees numeric,
    main_business text,
    org_code text,
    credit_code text,
    fetched_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_company_setup_date
    ON public.tushare_fund_company (setup_date);

CREATE TABLE IF NOT EXISTS public.tushare_fund_manager (
    manager_key text PRIMARY KEY,
    ts_code text NOT NULL,
    ann_date date,
    name text NOT NULL,
    gender text,
    birth_year text,
    edu text,
    nationality text,
    begin_date date,
    end_date date,
    resume text,
    fetched_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_manager_name
    ON public.tushare_fund_manager (name);

CREATE TABLE IF NOT EXISTS public.tushare_fund_nav (
    ts_code text NOT NULL,
    ann_date date,
    nav_date date NOT NULL,
    unit_nav numeric,
    accum_nav numeric,
    accum_div numeric,
    net_asset numeric,
    total_netasset numeric,
    adj_nav numeric,
    update_flag text,
    fetched_at timestamptz NOT NULL,
    PRIMARY KEY (ts_code, nav_date)
);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_nav_date
    ON public.tushare_fund_nav (nav_date);

CREATE TABLE IF NOT EXISTS public.tushare_fund_daily (
    ts_code text NOT NULL,
    trade_date date NOT NULL,
    pre_close numeric,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    change numeric,
    pct_chg numeric,
    vol numeric,
    amount numeric,
    fetched_at timestamptz NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_daily_date
    ON public.tushare_fund_daily (trade_date);

CREATE TABLE IF NOT EXISTS public.tushare_fund_share (
    ts_code text NOT NULL,
    trade_date date NOT NULL,
    fd_share numeric,
    fund_type text,
    market text,
    fetched_at timestamptz NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_share_date
    ON public.tushare_fund_share (trade_date);

CREATE TABLE IF NOT EXISTS public.tushare_fund_portfolio (
    ts_code text NOT NULL,
    ann_date date NOT NULL,
    end_date date,
    symbol text NOT NULL,
    mkv numeric,
    amount numeric,
    stk_mkv_ratio numeric,
    stk_float_ratio numeric,
    fetched_at timestamptz NOT NULL,
    PRIMARY KEY (ts_code, ann_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_portfolio_symbol
    ON public.tushare_fund_portfolio (symbol, end_date);

CREATE TABLE IF NOT EXISTS public.tushare_fund_div (
    ts_code text NOT NULL,
    ann_date date NOT NULL,
    imp_anndate date,
    base_date date,
    div_proc text,
    record_date date,
    ex_date date,
    pay_date date,
    earpay_date date,
    net_ex_date date,
    div_cash numeric,
    base_unit numeric,
    ear_distr numeric,
    ear_amount numeric,
    account_date date,
    base_year text,
    fetched_at timestamptz NOT NULL,
    PRIMARY KEY (ts_code, ann_date, base_date, div_cash)
);

CREATE INDEX IF NOT EXISTS idx_tushare_fund_div_ex_date
    ON public.tushare_fund_div (ex_date);
"""


def normalize(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {key: normalize(value) for key, value in row.to_dict().items()}


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    text_value = str(value)
    if len(text_value) != 8 or not text_value.isdigit():
        return None
    return datetime.strptime(text_value, "%Y%m%d").date()


def as_numeric(value: Any) -> Any:
    if value in (None, ""):
        return None
    return value


def date_for(endpoint: str, payload: dict[str, Any]) -> tuple[str, str]:
    mapping = {
        "fund_basic": "found_date",
        "fund_company": "setup_date",
        "fund_manager": "ann_date",
        "fund_nav": "nav_date",
        "fund_daily": "trade_date",
        "fund_adj": "trade_date",
        "fund_share": "trade_date",
        "fund_div": "ann_date",
        "fund_portfolio": "ann_date",
    }
    field = mapping.get(endpoint, "")
    value = str(payload.get(field) or "") if field else ""
    return value, field


def entity_for(endpoint: str, payload: dict[str, Any]) -> str:
    if payload.get("ts_code"):
        return str(payload["ts_code"])
    for key in ("name", "shortname", "org_code", "credit_code"):
        if payload.get(key):
            return str(payload[key])
    return ""


def insert_frame(engine, endpoint: str, frame: pd.DataFrame, content_type: str = "") -> int:
    if frame.empty:
        return 0
    fetched_at = datetime.now(UTC)
    rows = []
    for _, row in frame.iterrows():
        payload = row_to_dict(row)
        date_key, date_type = date_for(endpoint, payload)
        market = str(payload.get("market") or "")
        ts_code = str(payload.get("ts_code") or "")
        rows.append(
            {
                "endpoint": endpoint,
                "date_key": date_key,
                "date_type": date_type,
                "ts_code": ts_code,
                "entity_key": entity_for(endpoint, payload),
                "market": market,
                "content_type": content_type,
                "row_hash": stable_hash(payload),
                "fetched_at": fetched_at,
                "raw": json.dumps(payload, ensure_ascii=True, sort_keys=True),
            }
        )
    stmt = text(
        """
        INSERT INTO public.tushare_fund_records (
            endpoint, date_key, date_type, ts_code, entity_key, market, content_type,
            row_hash, fetched_at, raw
        )
        VALUES (
            :endpoint, :date_key, :date_type, :ts_code, :entity_key, :market, :content_type,
            :row_hash, :fetched_at, CAST(:raw AS jsonb)
        )
        ON CONFLICT (endpoint, date_key, ts_code, entity_key, market, content_type, row_hash)
        DO UPDATE SET
            date_type = EXCLUDED.date_type,
            fetched_at = EXCLUDED.fetched_at,
            raw = EXCLUDED.raw
        """
    )
    with engine.begin() as conn:
        result = conn.execute(stmt, rows)
    insert_narrow_frame(engine, endpoint, frame, fetched_at)
    return int(result.rowcount or 0)


def insert_narrow_frame(engine, endpoint: str, frame: pd.DataFrame, fetched_at: datetime) -> None:
    if frame.empty:
        return
    handlers = {
        "fund_basic": insert_fund_basic,
        "fund_company": insert_fund_company,
        "fund_manager": insert_fund_manager,
        "fund_nav": insert_fund_nav,
        "fund_daily": insert_fund_daily,
        "fund_share": insert_fund_share,
        "fund_portfolio": insert_fund_portfolio,
        "fund_div": insert_fund_div,
    }
    handler = handlers.get(endpoint)
    if handler:
        handler(engine, frame, fetched_at)


def payloads(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [row_to_dict(row) for _, row in frame.iterrows()]


def insert_fund_basic(engine, frame: pd.DataFrame, fetched_at: datetime) -> None:
    rows = []
    for item in payloads(frame):
        if not item.get("ts_code"):
            continue
        rows.append(
            {
                **{key: item.get(key) for key in [
                    "ts_code", "name", "management", "custodian", "fund_type", "exp_return",
                    "benchmark", "status", "invest_type", "type", "trustee", "market"
                ]},
                "found_date": parse_date(item.get("found_date")),
                "due_date": parse_date(item.get("due_date")),
                "list_date": parse_date(item.get("list_date")),
                "issue_date": parse_date(item.get("issue_date")),
                "delist_date": parse_date(item.get("delist_date")),
                "issue_amount": as_numeric(item.get("issue_amount")),
                "m_fee": as_numeric(item.get("m_fee")),
                "c_fee": as_numeric(item.get("c_fee")),
                "duration_year": as_numeric(item.get("duration_year")),
                "p_value": as_numeric(item.get("p_value")),
                "min_amount": as_numeric(item.get("min_amount")),
                "purc_startdate": parse_date(item.get("purc_startdate")),
                "redm_startdate": parse_date(item.get("redm_startdate")),
                "fetched_at": fetched_at,
            }
        )
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO public.tushare_fund_basic (
            ts_code, name, management, custodian, fund_type, found_date, due_date,
            list_date, issue_date, delist_date, issue_amount, m_fee, c_fee,
            duration_year, p_value, min_amount, exp_return, benchmark, status,
            invest_type, type, trustee, purc_startdate, redm_startdate, market, fetched_at
        )
        VALUES (
            :ts_code, :name, :management, :custodian, :fund_type, :found_date, :due_date,
            :list_date, :issue_date, :delist_date, :issue_amount, :m_fee, :c_fee,
            :duration_year, :p_value, :min_amount, :exp_return, :benchmark, :status,
            :invest_type, :type, :trustee, :purc_startdate, :redm_startdate, :market, :fetched_at
        )
        ON CONFLICT (ts_code) DO UPDATE SET
            name = EXCLUDED.name,
            management = EXCLUDED.management,
            custodian = EXCLUDED.custodian,
            fund_type = EXCLUDED.fund_type,
            found_date = EXCLUDED.found_date,
            due_date = EXCLUDED.due_date,
            list_date = EXCLUDED.list_date,
            issue_date = EXCLUDED.issue_date,
            delist_date = EXCLUDED.delist_date,
            issue_amount = EXCLUDED.issue_amount,
            m_fee = EXCLUDED.m_fee,
            c_fee = EXCLUDED.c_fee,
            duration_year = EXCLUDED.duration_year,
            p_value = EXCLUDED.p_value,
            min_amount = EXCLUDED.min_amount,
            exp_return = EXCLUDED.exp_return,
            benchmark = EXCLUDED.benchmark,
            status = EXCLUDED.status,
            invest_type = EXCLUDED.invest_type,
            type = EXCLUDED.type,
            trustee = EXCLUDED.trustee,
            purc_startdate = EXCLUDED.purc_startdate,
            redm_startdate = EXCLUDED.redm_startdate,
            market = EXCLUDED.market,
            fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def insert_fund_company(engine, frame: pd.DataFrame, fetched_at: datetime) -> None:
    rows = []
    for item in payloads(frame):
        entity_key = entity_for("fund_company", item)
        if not entity_key:
            continue
        rows.append(
            {
                **{key: item.get(key) for key in [
                    "name", "shortname", "province", "city", "address", "phone", "office",
                    "website", "chairman", "manager", "main_business", "org_code", "credit_code"
                ]},
                "entity_key": entity_key,
                "reg_capital": as_numeric(item.get("reg_capital")),
                "setup_date": parse_date(item.get("setup_date")),
                "end_date": parse_date(item.get("end_date")),
                "employees": as_numeric(item.get("employees")),
                "fetched_at": fetched_at,
            }
        )
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO public.tushare_fund_company (
            entity_key, name, shortname, province, city, address, phone, office, website,
            chairman, manager, reg_capital, setup_date, end_date, employees,
            main_business, org_code, credit_code, fetched_at
        )
        VALUES (
            :entity_key, :name, :shortname, :province, :city, :address, :phone, :office, :website,
            :chairman, :manager, :reg_capital, :setup_date, :end_date, :employees,
            :main_business, :org_code, :credit_code, :fetched_at
        )
        ON CONFLICT (entity_key) DO UPDATE SET
            name = EXCLUDED.name,
            shortname = EXCLUDED.shortname,
            province = EXCLUDED.province,
            city = EXCLUDED.city,
            address = EXCLUDED.address,
            phone = EXCLUDED.phone,
            office = EXCLUDED.office,
            website = EXCLUDED.website,
            chairman = EXCLUDED.chairman,
            manager = EXCLUDED.manager,
            reg_capital = EXCLUDED.reg_capital,
            setup_date = EXCLUDED.setup_date,
            end_date = EXCLUDED.end_date,
            employees = EXCLUDED.employees,
            main_business = EXCLUDED.main_business,
            org_code = EXCLUDED.org_code,
            credit_code = EXCLUDED.credit_code,
            fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def insert_fund_manager(engine, frame: pd.DataFrame, fetched_at: datetime) -> None:
    rows = []
    for item in payloads(frame):
        if not item.get("ts_code") or not item.get("name"):
            continue
        key_payload = {
            "ts_code": item.get("ts_code"),
            "ann_date": item.get("ann_date"),
            "name": item.get("name"),
            "begin_date": item.get("begin_date"),
            "end_date": item.get("end_date"),
        }
        rows.append(
            {
                "manager_key": stable_hash(key_payload),
                "ts_code": item.get("ts_code"),
                "ann_date": parse_date(item.get("ann_date")),
                "name": item.get("name"),
                "gender": item.get("gender"),
                "birth_year": str(item.get("birth_year")) if item.get("birth_year") is not None else None,
                "edu": item.get("edu"),
                "nationality": item.get("nationality"),
                "begin_date": parse_date(item.get("begin_date")),
                "end_date": parse_date(item.get("end_date")),
                "resume": item.get("resume"),
                "fetched_at": fetched_at,
            }
        )
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO public.tushare_fund_manager (
            manager_key, ts_code, ann_date, name, gender, birth_year, edu, nationality,
            begin_date, end_date, resume, fetched_at
        )
        VALUES (
            :manager_key, :ts_code, :ann_date, :name, :gender, :birth_year, :edu, :nationality,
            :begin_date, :end_date, :resume, :fetched_at
        )
        ON CONFLICT (manager_key) DO UPDATE SET
            ts_code = EXCLUDED.ts_code,
            ann_date = EXCLUDED.ann_date,
            name = EXCLUDED.name,
            gender = EXCLUDED.gender,
            birth_year = EXCLUDED.birth_year,
            edu = EXCLUDED.edu,
            nationality = EXCLUDED.nationality,
            end_date = EXCLUDED.end_date,
            resume = EXCLUDED.resume,
            fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def insert_fund_nav(engine, frame: pd.DataFrame, fetched_at: datetime) -> None:
    rows = []
    for item in payloads(frame):
        nav_date = parse_date(item.get("nav_date"))
        if not item.get("ts_code") or not nav_date:
            continue
        rows.append(
            {
                "ts_code": item.get("ts_code"),
                "ann_date": parse_date(item.get("ann_date")),
                "nav_date": nav_date,
                "unit_nav": as_numeric(item.get("unit_nav")),
                "accum_nav": as_numeric(item.get("accum_nav")),
                "accum_div": as_numeric(item.get("accum_div")),
                "net_asset": as_numeric(item.get("net_asset")),
                "total_netasset": as_numeric(item.get("total_netasset")),
                "adj_nav": as_numeric(item.get("adj_nav")),
                "update_flag": item.get("update_flag"),
                "fetched_at": fetched_at,
            }
        )
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO public.tushare_fund_nav (
            ts_code, ann_date, nav_date, unit_nav, accum_nav, accum_div,
            net_asset, total_netasset, adj_nav, update_flag, fetched_at
        )
        VALUES (
            :ts_code, :ann_date, :nav_date, :unit_nav, :accum_nav, :accum_div,
            :net_asset, :total_netasset, :adj_nav, :update_flag, :fetched_at
        )
        ON CONFLICT (ts_code, nav_date) DO UPDATE SET
            ann_date = EXCLUDED.ann_date,
            unit_nav = EXCLUDED.unit_nav,
            accum_nav = EXCLUDED.accum_nav,
            accum_div = EXCLUDED.accum_div,
            net_asset = EXCLUDED.net_asset,
            total_netasset = EXCLUDED.total_netasset,
            adj_nav = EXCLUDED.adj_nav,
            update_flag = EXCLUDED.update_flag,
            fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def insert_fund_daily(engine, frame: pd.DataFrame, fetched_at: datetime) -> None:
    rows = []
    for item in payloads(frame):
        trade_date = parse_date(item.get("trade_date"))
        if not item.get("ts_code") or not trade_date:
            continue
        rows.append(
            {
                "ts_code": item.get("ts_code"),
                "trade_date": trade_date,
                "pre_close": as_numeric(item.get("pre_close")),
                "open": as_numeric(item.get("open")),
                "high": as_numeric(item.get("high")),
                "low": as_numeric(item.get("low")),
                "close": as_numeric(item.get("close")),
                "change": as_numeric(item.get("change")),
                "pct_chg": as_numeric(item.get("pct_chg")),
                "vol": as_numeric(item.get("vol")),
                "amount": as_numeric(item.get("amount")),
                "fetched_at": fetched_at,
            }
        )
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO public.tushare_fund_daily (
            ts_code, trade_date, pre_close, open, high, low, close, change,
            pct_chg, vol, amount, fetched_at
        )
        VALUES (
            :ts_code, :trade_date, :pre_close, :open, :high, :low, :close, :change,
            :pct_chg, :vol, :amount, :fetched_at
        )
        ON CONFLICT (ts_code, trade_date) DO UPDATE SET
            pre_close = EXCLUDED.pre_close,
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            change = EXCLUDED.change,
            pct_chg = EXCLUDED.pct_chg,
            vol = EXCLUDED.vol,
            amount = EXCLUDED.amount,
            fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def insert_fund_share(engine, frame: pd.DataFrame, fetched_at: datetime) -> None:
    rows = []
    for item in payloads(frame):
        trade_date = parse_date(item.get("trade_date"))
        if not item.get("ts_code") or not trade_date:
            continue
        rows.append(
            {
                "ts_code": item.get("ts_code"),
                "trade_date": trade_date,
                "fd_share": as_numeric(item.get("fd_share")),
                "fund_type": item.get("fund_type"),
                "market": item.get("market"),
                "fetched_at": fetched_at,
            }
        )
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO public.tushare_fund_share (
            ts_code, trade_date, fd_share, fund_type, market, fetched_at
        )
        VALUES (:ts_code, :trade_date, :fd_share, :fund_type, :market, :fetched_at)
        ON CONFLICT (ts_code, trade_date) DO UPDATE SET
            fd_share = EXCLUDED.fd_share,
            fund_type = EXCLUDED.fund_type,
            market = EXCLUDED.market,
            fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def insert_fund_portfolio(engine, frame: pd.DataFrame, fetched_at: datetime) -> None:
    rows = []
    for item in payloads(frame):
        ann_date = parse_date(item.get("ann_date"))
        if not item.get("ts_code") or not ann_date or not item.get("symbol"):
            continue
        rows.append(
            {
                "ts_code": item.get("ts_code"),
                "ann_date": ann_date,
                "end_date": parse_date(item.get("end_date")),
                "symbol": item.get("symbol"),
                "mkv": as_numeric(item.get("mkv")),
                "amount": as_numeric(item.get("amount")),
                "stk_mkv_ratio": as_numeric(item.get("stk_mkv_ratio")),
                "stk_float_ratio": as_numeric(item.get("stk_float_ratio")),
                "fetched_at": fetched_at,
            }
        )
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO public.tushare_fund_portfolio (
            ts_code, ann_date, end_date, symbol, mkv, amount, stk_mkv_ratio,
            stk_float_ratio, fetched_at
        )
        VALUES (
            :ts_code, :ann_date, :end_date, :symbol, :mkv, :amount, :stk_mkv_ratio,
            :stk_float_ratio, :fetched_at
        )
        ON CONFLICT (ts_code, ann_date, symbol) DO UPDATE SET
            end_date = EXCLUDED.end_date,
            mkv = EXCLUDED.mkv,
            amount = EXCLUDED.amount,
            stk_mkv_ratio = EXCLUDED.stk_mkv_ratio,
            stk_float_ratio = EXCLUDED.stk_float_ratio,
            fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def insert_fund_div(engine, frame: pd.DataFrame, fetched_at: datetime) -> None:
    rows = []
    for item in payloads(frame):
        ann_date = parse_date(item.get("ann_date"))
        if not item.get("ts_code") or not ann_date:
            continue
        rows.append(
            {
                "ts_code": item.get("ts_code"),
                "ann_date": ann_date,
                "imp_anndate": parse_date(item.get("imp_anndate")),
                "base_date": parse_date(item.get("base_date")),
                "div_proc": item.get("div_proc"),
                "record_date": parse_date(item.get("record_date")),
                "ex_date": parse_date(item.get("ex_date")),
                "pay_date": parse_date(item.get("pay_date")),
                "earpay_date": parse_date(item.get("earpay_date")),
                "net_ex_date": parse_date(item.get("net_ex_date")),
                "div_cash": as_numeric(item.get("div_cash")),
                "base_unit": as_numeric(item.get("base_unit")),
                "ear_distr": as_numeric(item.get("ear_distr")),
                "ear_amount": as_numeric(item.get("ear_amount")),
                "account_date": parse_date(item.get("account_date")),
                "base_year": str(item.get("base_year")) if item.get("base_year") is not None else None,
                "fetched_at": fetched_at,
            }
        )
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO public.tushare_fund_div (
            ts_code, ann_date, imp_anndate, base_date, div_proc, record_date,
            ex_date, pay_date, earpay_date, net_ex_date, div_cash, base_unit,
            ear_distr, ear_amount, account_date, base_year, fetched_at
        )
        VALUES (
            :ts_code, :ann_date, :imp_anndate, :base_date, :div_proc, :record_date,
            :ex_date, :pay_date, :earpay_date, :net_ex_date, :div_cash, :base_unit,
            :ear_distr, :ear_amount, :account_date, :base_year, :fetched_at
        )
        ON CONFLICT (ts_code, ann_date, base_date, div_cash) DO UPDATE SET
            imp_anndate = EXCLUDED.imp_anndate,
            div_proc = EXCLUDED.div_proc,
            record_date = EXCLUDED.record_date,
            ex_date = EXCLUDED.ex_date,
            pay_date = EXCLUDED.pay_date,
            earpay_date = EXCLUDED.earpay_date,
            net_ex_date = EXCLUDED.net_ex_date,
            base_unit = EXCLUDED.base_unit,
            ear_distr = EXCLUDED.ear_distr,
            ear_amount = EXCLUDED.ear_amount,
            account_date = EXCLUDED.account_date,
            base_year = EXCLUDED.base_year,
            fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def paged_query(pro, endpoint: str, params: dict[str, Any], limit: int, sleep_seconds: float):
    offset = 0
    while True:
        frame = pro.query(endpoint, **params, limit=limit, offset=offset)
        if frame is None or frame.empty:
            break
        yield offset, frame
        if len(frame) < limit:
            break
        offset += limit
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def collect_metadata(engine, pro, limit: int, sleep_seconds: float) -> None:
    with engine.begin() as conn:
        conn.execute(text(DDL))

    for market in ["O", "E"]:
        total = 0
        for offset, frame in paged_query(pro, "fund_basic", {"market": market}, limit, sleep_seconds):
            inserted = insert_frame(engine, "fund_basic", frame, content_type=market)
            total += inserted
            print(f"fund_basic market={market} offset={offset} fetched={len(frame)} inserted={inserted} total={total}", flush=True)

    for endpoint in ["fund_company", "fund_manager"]:
        total = 0
        for offset, frame in paged_query(pro, endpoint, {}, limit, sleep_seconds):
            inserted = insert_frame(engine, endpoint, frame)
            total += inserted
            print(f"{endpoint} offset={offset} fetched={len(frame)} inserted={inserted} total={total}", flush=True)


def load_fund_codes(engine) -> list[str]:
    query = text(
        """
        SELECT DISTINCT ts_code
        FROM public.tushare_fund_records
        WHERE endpoint = 'fund_basic'
          AND ts_code <> ''
          AND COALESCE(raw->>'found_date', '') >= '20000101'
        ORDER BY ts_code
        """
    )
    with engine.connect() as conn:
        return [row[0] for row in conn.execute(query).all()]


def existing_nav_codes(engine) -> set[str]:
    query = text("SELECT DISTINCT ts_code FROM public.tushare_fund_records WHERE endpoint = 'fund_nav'")
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(query).all()}


def collect_nav(engine, pro, start_date: str, end_date: str, sleep_seconds: float, max_codes: int | None, resume: bool) -> None:
    codes = load_fund_codes(engine)
    if resume:
        done = existing_nav_codes(engine)
        codes = [code for code in codes if code not in done]
    if max_codes is not None:
        codes = codes[:max_codes]
    print(f"fund_nav codes={len(codes)} start={start_date} end={end_date}", flush=True)
    total = 0
    for index, code in enumerate(codes, 1):
        try:
            frame = pro.query("fund_nav", ts_code=code, start_date=start_date, end_date=end_date)
            inserted = insert_frame(engine, "fund_nav", frame)
            total += inserted
            print(f"fund_nav {index}/{len(codes)} {code} fetched={len(frame)} inserted={inserted} total={total}", flush=True)
        except Exception as exc:
            print(f"ERROR fund_nav {index}/{len(codes)} {code}: {exc}", flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Tushare fund data into public.tushare_fund_records.")
    parser.add_argument("--metadata", action="store_true", help="Collect fund_basic, fund_company, and fund_manager.")
    parser.add_argument("--nav", action="store_true", help="Collect fund_nav for fund_basic codes found since start date.")
    parser.add_argument("--start-date", default="20000101")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--max-codes", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Do not skip codes that already have fund_nav records.")
    args = parser.parse_args()

    load_env_file()
    token = (os.getenv("TUSHARE_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is missing.")
    ts.set_token(token)
    pro = ts.pro_api()
    url = os.getenv("DATABASE_URL") or os.getenv("SMART_STOCK_DATABASE_URL") or build_database_url_from_parts()
    engine = create_engine(url, pool_pre_ping=True, future=True)
    with engine.begin() as conn:
        conn.execute(text(DDL))

    if args.metadata:
        collect_metadata(engine, pro, args.limit, args.sleep_seconds)
    if args.nav:
        collect_nav(engine, pro, args.start_date, args.end_date, args.sleep_seconds, args.max_codes, not args.no_resume)


if __name__ == "__main__":
    main()

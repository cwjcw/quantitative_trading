from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import tushare as ts
from sqlalchemy import text

from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine
from quantitative_trading.tushare.collect_financials import collect_financials, load_codes
from quantitative_trading.tushare.collect_raw import collect, parse_yyyymmdd, to_ts_code
from quantitative_trading.tushare.schema import ensure_schema


STAGES: dict[str, dict[str, object]] = {
    "core_daily": {
        "priorities": {"P0"},
        "endpoints": {
            "daily",
            "daily_basic",
            "moneyflow",
            "moneyflow_ths",
            "moneyflow_dc",
            "moneyflow_ind_ths",
            "moneyflow_cnt_ths",
            "moneyflow_ind_dc",
            "moneyflow_mkt_dc",
            "moneyflow_hsgt",
            "stk_limit",
            "limit_list_d",
            "limit_list_ths",
            "top_list",
            "top_inst",
            "margin",
            "margin_detail",
            "hk_hold",
            "hsgt_top10",
            "ggt_top10",
            "cyq_perf",
        },
    },
    "risk_events": {
        "priorities": {"P1"},
        "endpoints": {
            "suspend_d",
            "stock_st",
            "share_float",
            "pledge_stat",
            "pledge_detail",
            "stk_holdernumber",
            "stk_holdertrade",
            "repurchase",
            "block_trade",
            "anns_d",
            "report_rc",
        },
    },
    "fundamental_events": {
        "priorities": {"P2"},
        "endpoints": {"forecast", "express", "dividend", "disclosure_date", "broker_recommend"},
    },
    "static_reference": {
        "priorities": {"P1", "P3"},
        "endpoints": {
            "stock_basic",
            "stock_company",
            "namechange",
            "trade_cal",
            "ths_index",
            "dc_index",
            "dc_member",
            "index_classify",
            "index_member_all",
            "index_dailybasic",
            "index_weight",
        },
    },
}


def yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def parse_stage_names(values: list[str] | None) -> list[str]:
    if not values:
        return ["core_daily", "risk_events", "fundamental_events", "static_reference"]

    selected: list[str] = []
    for value in values:
        for item in value.split(","):
            stage = item.strip()
            if not stage:
                continue
            if stage not in STAGES:
                raise ValueError(f"Unknown stage: {stage}. Choices: {', '.join(sorted(STAGES))}")
            selected.append(stage)
    return selected


def load_recent_financial_codes(start_date: date, end_date: date, limit: int | None) -> list[str]:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    ensure_schema(engine)

    query = text(
        """
        SELECT DISTINCT ts_code
        FROM tushare_raw_records
        WHERE endpoint IN ('anns_d', 'forecast', 'express')
          AND date_key BETWEEN :start_date AND :end_date
          AND ts_code ~ '^[0-9]{6}\\.(SH|SZ|BJ)$'
        ORDER BY ts_code
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"start_date": yyyymmdd(start_date), "end_date": yyyymmdd(end_date)},
        ).scalars().all()

    codes = [to_ts_code(str(item)) for item in rows]
    if limit is not None:
        codes = codes[:limit]
    return codes


def load_all_local_codes(limit: int | None) -> list[str]:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    ensure_schema(engine)

    ts.set_token(settings.tushare_token)
    pro = ts.pro_api()
    codes = load_codes(engine, pro, include_delisted=False)
    if limit is not None:
        codes = codes[:limit]
    return codes


def run_raw_stages(
    start_date: date,
    end_date: date,
    stages: list[str],
    sleep_seconds: float,
) -> None:
    for stage_name in stages:
        stage = STAGES[stage_name]
        print(f"MAIN_GET_INFO stage={stage_name} start={yyyymmdd(start_date)} end={yyyymmdd(end_date)}")
        collect(
            start_date=start_date,
            end_date=end_date,
            priorities=set(stage["priorities"]),
            endpoint_names=set(stage["endpoints"]),
            ts_codes=None,
            all_codes=False,
            max_codes=None,
            sleep_seconds=sleep_seconds,
        )


def run_financials(
    start_date: date,
    end_date: date,
    mode: str,
    max_codes: int | None,
    sleep_seconds: float,
    retry_failed: bool,
) -> None:
    requested_mode = mode
    if mode == "auto":
        if end_date.weekday() == 6:
            mode = "all"
        else:
            print(
                "MAIN_GET_INFO financial_mode=auto "
                f"weekday={end_date.strftime('%A')} skip, financials run on Sunday"
            )
            return

    if mode == "none":
        print("MAIN_GET_INFO financial_mode=none skip")
        return

    if mode == "all":
        codes = load_all_local_codes(max_codes)
    else:
        codes = load_recent_financial_codes(start_date, end_date, max_codes)

    if not codes:
        print(f"MAIN_GET_INFO financial_mode={mode} codes=0 skip")
        return

    print(
        "MAIN_GET_INFO "
        f"financial_mode={requested_mode} resolved_mode={mode} codes={len(codes)} "
        f"start={yyyymmdd(start_date)} end={yyyymmdd(end_date)}"
    )
    collect_financials(
        start_date=start_date,
        end_date=end_date,
        endpoint_names=None,
        ts_codes=set(codes),
        all_codes=False,
        include_delisted=False,
        offset=0,
        max_codes=None,
        sleep_seconds=sleep_seconds,
        retry_failed=retry_failed,
    )


def apply_analysis_views() -> None:
    from quantitative_trading.analysis.apply_views import main as apply_views_main

    apply_views_main()


def build_parser() -> argparse.ArgumentParser:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    parser = argparse.ArgumentParser(description="Daily unified data collection entrypoint for n8n.")
    parser.add_argument("--start-date", default=None, help="Start date in YYYYMMDD. Default: end date minus lookback days.")
    parser.add_argument("--end-date", default=yyyymmdd(today), help="End date in YYYYMMDD. Default: today in Asia/Shanghai.")
    parser.add_argument("--lookback-days", type=int, default=7, help="Rolling days to refetch when start date is omitted.")
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help="Stage name or comma-separated stages. Defaults to all daily stages.",
    )
    parser.add_argument("--raw-sleep-seconds", type=float, default=0.05, help="Sleep between raw API calls.")
    parser.add_argument(
        "--financial-mode",
        choices=["auto", "none", "recent", "all"],
        default="auto",
        help="auto runs all local stock financials only when end date is Sunday; recent uses announcement-related codes; all uses every local stock code.",
    )
    parser.add_argument("--financial-max-codes", type=int, default=0, help="Limit financial stock codes. Use 0 for no limit.")
    parser.add_argument("--financial-sleep-seconds", type=float, default=0.2, help="Sleep between financial API calls.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed financial checkpoints.")
    parser.add_argument("--skip-views", action="store_true", help="Do not refresh analytics views.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    end_date = parse_yyyymmdd(args.end_date)
    start_date = parse_yyyymmdd(args.start_date) if args.start_date else end_date - timedelta(days=args.lookback_days)
    if start_date > end_date:
        raise ValueError("start date must be <= end date")

    stages = parse_stage_names(args.stage)
    financial_max_codes = None if args.financial_max_codes == 0 else args.financial_max_codes

    run_raw_stages(start_date, end_date, stages, args.raw_sleep_seconds)
    run_financials(
        start_date=start_date,
        end_date=end_date,
        mode=args.financial_mode,
        max_codes=financial_max_codes,
        sleep_seconds=args.financial_sleep_seconds,
        retry_failed=args.retry_failed,
    )
    if not args.skip_views:
        apply_analysis_views()
    print(f"MAIN_GET_INFO done start={yyyymmdd(start_date)} end={yyyymmdd(end_date)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from datetime import date, datetime

import pandas as pd

from quantitative_trading.tushare.collect_raw import collect


STAGES: dict[str, dict[str, object]] = {
    "core_daily": {
        "priorities": {"P0"},
        "endpoints": {
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
    "monthly_research": {
        "priorities": {"P2"},
        "endpoints": {"broker_recommend"},
    },
    "fundamental_events": {
        "priorities": {"P2"},
        "endpoints": {"forecast", "express", "dividend", "disclosure_date", "broker_recommend"},
    },
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def month_chunks(start_date: date, end_date: date) -> list[tuple[date, date]]:
    starts = pd.date_range(start=start_date, end=end_date, freq="MS")
    firsts = [start_date]
    firsts.extend(item.date() for item in starts if item.date() > start_date)
    chunks = []
    for index, chunk_start in enumerate(firsts):
        if index + 1 < len(firsts):
            chunk_end = firsts[index + 1] - pd.Timedelta(days=1)
            chunks.append((chunk_start, chunk_end))
        else:
            chunks.append((chunk_start, end_date))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged Tushare backfills.")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    stage = STAGES[args.stage]
    for chunk_start, chunk_end in month_chunks(parse_date(args.start_date), parse_date(args.end_date)):
        print(f"BACKFILL stage={args.stage} start={chunk_start:%Y%m%d} end={chunk_end:%Y%m%d}")
        collect(
            start_date=chunk_start,
            end_date=chunk_end,
            priorities=set(stage["priorities"]),
            endpoint_names=set(stage["endpoints"]),
            ts_codes=None,
            all_codes=False,
            max_codes=None,
            sleep_seconds=args.sleep_seconds,
        )


if __name__ == "__main__":
    main()

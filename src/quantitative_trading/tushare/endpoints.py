from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TushareEndpoint:
    name: str
    priority: str
    date_param: str | None
    date_field: str | None
    code_field: str | None = "ts_code"
    content_type: str = ""
    requires_ts_code: bool = False
    default_params: dict[str, str] = field(default_factory=dict)


ENDPOINTS: tuple[TushareEndpoint, ...] = (
    TushareEndpoint("daily", "P0", "trade_date", "trade_date"),
    TushareEndpoint("daily_basic", "P0", "trade_date", "trade_date"),
    TushareEndpoint("moneyflow", "P0", "trade_date", "trade_date"),
    TushareEndpoint("moneyflow_ths", "P0", "trade_date", "trade_date"),
    TushareEndpoint("moneyflow_dc", "P0", "trade_date", "trade_date"),
    TushareEndpoint("moneyflow_ind_ths", "P0", "trade_date", "trade_date", "ts_code"),
    TushareEndpoint("moneyflow_cnt_ths", "P0", "trade_date", "trade_date", "ts_code"),
    TushareEndpoint("moneyflow_ind_dc", "P0", "trade_date", "trade_date", "ts_code"),
    TushareEndpoint("moneyflow_mkt_dc", "P0", "trade_date", "trade_date", None),
    TushareEndpoint("moneyflow_hsgt", "P0", "trade_date", "trade_date", None),
    TushareEndpoint("stk_limit", "P0", "trade_date", "trade_date"),
    TushareEndpoint("limit_list_d", "P0", "trade_date", "trade_date"),
    TushareEndpoint("limit_list_ths", "P0", "trade_date", "trade_date"),
    TushareEndpoint("top_list", "P0", "trade_date", "trade_date"),
    TushareEndpoint("top_inst", "P0", "trade_date", "trade_date"),
    TushareEndpoint("margin", "P0", "trade_date", "trade_date", "exchange_id"),
    TushareEndpoint("margin_detail", "P0", "trade_date", "trade_date"),
    TushareEndpoint("hk_hold", "P0", "trade_date", "trade_date"),
    TushareEndpoint("hsgt_top10", "P0", "trade_date", "trade_date"),
    TushareEndpoint("ggt_top10", "P0", "trade_date", "trade_date"),
    TushareEndpoint("cyq_perf", "P0", "trade_date", "trade_date"),
    TushareEndpoint("suspend_d", "P1", "trade_date", "trade_date"),
    TushareEndpoint("stock_st", "P1", "trade_date", "trade_date"),
    TushareEndpoint("share_float", "P1", "ann_date", "ann_date"),
    TushareEndpoint("pledge_stat", "P1", "end_date", "end_date"),
    TushareEndpoint("pledge_detail", "P1", "ann_date", "ann_date"),
    TushareEndpoint("stk_holdernumber", "P1", "ann_date", "ann_date"),
    TushareEndpoint("stk_holdertrade", "P1", "ann_date", "ann_date"),
    TushareEndpoint("repurchase", "P1", "ann_date", "ann_date"),
    TushareEndpoint("block_trade", "P1", "trade_date", "trade_date"),
    TushareEndpoint("stock_basic", "P1", None, "list_date", content_type="L", default_params={"exchange": "", "list_status": "L"}),
    TushareEndpoint("stock_basic", "P1", None, "list_date", content_type="D", default_params={"exchange": "", "list_status": "D"}),
    TushareEndpoint("stock_basic", "P1", None, "list_date", content_type="P", default_params={"exchange": "", "list_status": "P"}),
    TushareEndpoint("stock_company", "P1", None, None, content_type="SSE", default_params={"exchange": "SSE"}),
    TushareEndpoint("stock_company", "P1", None, None, content_type="SZSE", default_params={"exchange": "SZSE"}),
    TushareEndpoint("namechange", "P1", None, "ann_date"),
    TushareEndpoint("anns_d", "P1", "date_range", "ann_date"),
    TushareEndpoint("report_rc", "P1", "date_range", "report_date"),
    TushareEndpoint("fina_indicator", "P2", "ann_date", "ann_date", requires_ts_code=True),
    TushareEndpoint("income", "P2", "ann_date", "ann_date", requires_ts_code=True),
    TushareEndpoint("balancesheet", "P2", "ann_date", "ann_date", requires_ts_code=True),
    TushareEndpoint("cashflow", "P2", "ann_date", "ann_date", requires_ts_code=True),
    TushareEndpoint("forecast", "P2", "ann_date", "ann_date"),
    TushareEndpoint("express", "P2", "ann_date", "ann_date"),
    TushareEndpoint("dividend", "P2", "ann_date", "ann_date"),
    TushareEndpoint("fina_mainbz", "P2", "period", "end_date", requires_ts_code=True),
    TushareEndpoint("disclosure_date", "P2", "end_date", "end_date"),
    TushareEndpoint("broker_recommend", "P2", "month", "month"),
    TushareEndpoint("index_dailybasic", "P3", "trade_date", "trade_date"),
    TushareEndpoint("index_weight", "P3", "trade_date", "trade_date", "index_code"),
    TushareEndpoint("trade_cal", "P3", "date_range", "cal_date", "exchange", default_params={"exchange": "SSE"}),
    TushareEndpoint("ths_index", "P3", None, "list_date"),
    TushareEndpoint("dc_index", "P3", "trade_date", "trade_date"),
    TushareEndpoint("dc_member", "P3", "trade_date", "trade_date", "con_code"),
    TushareEndpoint("index_classify", "P3", None, None, "index_code", content_type="SW2021_L1", default_params={"level": "L1", "src": "SW2021"}),
    TushareEndpoint("index_classify", "P3", None, None, "index_code", content_type="SW2021_L2", default_params={"level": "L2", "src": "SW2021"}),
    TushareEndpoint("index_classify", "P3", None, None, "index_code", content_type="SW2021_L3", default_params={"level": "L3", "src": "SW2021"}),
    TushareEndpoint("index_member_all", "P3", None, None),
)


def select_endpoints(priorities: set[str], names: set[str] | None = None) -> list[TushareEndpoint]:
    selected = [item for item in ENDPOINTS if item.priority in priorities]
    if names:
        selected = [item for item in selected if item.name in names]
    return selected

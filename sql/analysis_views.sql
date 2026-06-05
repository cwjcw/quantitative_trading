CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.tushare_daily_basic AS
SELECT
    raw->>'ts_code' AS ts_code,
    to_date(raw->>'trade_date', 'YYYYMMDD') AS trade_date,
    NULLIF(raw->>'turnover_rate', '')::numeric AS turnover_rate,
    NULLIF(raw->>'turnover_rate_f', '')::numeric AS turnover_rate_f,
    NULLIF(raw->>'volume_ratio', '')::numeric AS volume_ratio,
    NULLIF(raw->>'pe', '')::numeric AS pe,
    NULLIF(raw->>'pe_ttm', '')::numeric AS pe_ttm,
    NULLIF(raw->>'pb', '')::numeric AS pb,
    NULLIF(raw->>'ps_ttm', '')::numeric AS ps_ttm,
    NULLIF(raw->>'dv_ttm', '')::numeric AS dv_ttm,
    NULLIF(raw->>'total_share', '')::numeric AS total_share,
    NULLIF(raw->>'float_share', '')::numeric AS float_share,
    NULLIF(raw->>'free_share', '')::numeric AS free_share,
    NULLIF(raw->>'total_mv', '')::numeric AS total_mv,
    NULLIF(raw->>'circ_mv', '')::numeric AS circ_mv,
    fetched_at
FROM tushare_raw_records
WHERE endpoint = 'daily_basic';

CREATE OR REPLACE VIEW analytics.tushare_moneyflow_stock AS
SELECT
    raw->>'ts_code' AS ts_code,
    to_date(raw->>'trade_date', 'YYYYMMDD') AS trade_date,
    NULLIF(raw->>'buy_sm_amount', '')::numeric AS buy_sm_amount,
    NULLIF(raw->>'sell_sm_amount', '')::numeric AS sell_sm_amount,
    NULLIF(raw->>'buy_md_amount', '')::numeric AS buy_md_amount,
    NULLIF(raw->>'sell_md_amount', '')::numeric AS sell_md_amount,
    NULLIF(raw->>'buy_lg_amount', '')::numeric AS buy_lg_amount,
    NULLIF(raw->>'sell_lg_amount', '')::numeric AS sell_lg_amount,
    NULLIF(raw->>'buy_elg_amount', '')::numeric AS buy_elg_amount,
    NULLIF(raw->>'sell_elg_amount', '')::numeric AS sell_elg_amount,
    NULLIF(raw->>'net_mf_amount', '')::numeric AS net_mf_amount,
    NULLIF(raw->>'net_mf_vol', '')::numeric AS net_mf_vol,
    fetched_at
FROM tushare_raw_records
WHERE endpoint = 'moneyflow';

CREATE OR REPLACE VIEW analytics.tushare_cyq_perf AS
SELECT
    raw->>'ts_code' AS ts_code,
    to_date(raw->>'trade_date', 'YYYYMMDD') AS trade_date,
    NULLIF(raw->>'his_low', '')::numeric AS his_low,
    NULLIF(raw->>'his_high', '')::numeric AS his_high,
    NULLIF(raw->>'cost_5pct', '')::numeric AS cost_5pct,
    NULLIF(raw->>'cost_15pct', '')::numeric AS cost_15pct,
    NULLIF(raw->>'cost_50pct', '')::numeric AS cost_50pct,
    NULLIF(raw->>'cost_85pct', '')::numeric AS cost_85pct,
    NULLIF(raw->>'cost_95pct', '')::numeric AS cost_95pct,
    NULLIF(raw->>'weight_avg', '')::numeric AS weight_avg,
    NULLIF(raw->>'winner_rate', '')::numeric AS winner_rate,
    fetched_at
FROM tushare_raw_records
WHERE endpoint = 'cyq_perf';

CREATE OR REPLACE VIEW analytics.tushare_market_sentiment_daily AS
SELECT
    trade_date,
    count(*) FILTER (WHERE endpoint = 'limit_list_d') AS limit_event_count,
    count(*) FILTER (WHERE endpoint = 'top_list') AS top_list_count,
    count(*) FILTER (WHERE endpoint = 'top_inst') AS top_inst_count,
    count(*) FILTER (WHERE endpoint = 'block_trade') AS block_trade_count,
    count(*) FILTER (WHERE endpoint = 'anns_d') AS announcement_count,
    count(*) FILTER (WHERE endpoint = 'report_rc') AS research_report_count
FROM (
    SELECT endpoint, to_date(date_key, 'YYYYMMDD') AS trade_date
    FROM tushare_raw_records
    WHERE endpoint IN ('limit_list_d', 'top_list', 'top_inst', 'block_trade', 'anns_d', 'report_rc')
      AND date_key ~ '^[0-9]{8}$'
) t
GROUP BY trade_date;

CREATE OR REPLACE VIEW analytics.tushare_sector_moneyflow_dc AS
SELECT
    raw->>'ts_code' AS sector_code,
    raw->>'name' AS sector_name,
    to_date(raw->>'trade_date', 'YYYYMMDD') AS trade_date,
    raw->>'leading_code' AS leading_code,
    raw->>'leading' AS leading_name,
    NULLIF(raw->>'pct_change', '')::numeric AS pct_change,
    NULLIF(raw->>'leading_pct', '')::numeric AS leading_pct,
    NULLIF(raw->>'net_amount', '')::numeric AS net_amount,
    NULLIF(raw->>'net_amount_rate', '')::numeric AS net_amount_rate,
    NULLIF(raw->>'total_mv', '')::numeric AS total_mv,
    fetched_at
FROM tushare_raw_records
WHERE endpoint = 'dc_index';

CREATE OR REPLACE VIEW analytics.tushare_stock_events AS
SELECT
    endpoint,
    raw->>'ts_code' AS ts_code,
    to_date(date_key, 'YYYYMMDD') AS event_date,
    raw,
    fetched_at
FROM tushare_raw_records
WHERE endpoint IN (
    'stock_st', 'suspend_d', 'share_float', 'pledge_detail',
    'stk_holdernumber', 'stk_holdertrade', 'repurchase', 'anns_d',
    'report_rc', 'namechange'
)
  AND date_key ~ '^[0-9]{8}$';

CREATE OR REPLACE VIEW analytics.tushare_stock_feature_daily AS
SELECT
    COALESCE(db.ts_code, mf.ts_code, cyq.ts_code) AS ts_code,
    COALESCE(db.trade_date, mf.trade_date, cyq.trade_date) AS trade_date,
    db.turnover_rate,
    db.turnover_rate_f,
    db.volume_ratio,
    db.pe_ttm,
    db.pb,
    db.total_mv,
    db.circ_mv,
    mf.net_mf_amount,
    mf.net_mf_vol,
    cyq.winner_rate,
    cyq.cost_50pct,
    cyq.weight_avg
FROM analytics.tushare_daily_basic db
FULL OUTER JOIN analytics.tushare_moneyflow_stock mf
    ON db.ts_code = mf.ts_code AND db.trade_date = mf.trade_date
FULL OUTER JOIN analytics.tushare_cyq_perf cyq
    ON COALESCE(db.ts_code, mf.ts_code) = cyq.ts_code
   AND COALESCE(db.trade_date, mf.trade_date) = cyq.trade_date;

CREATE OR REPLACE VIEW analytics.tushare_fina_indicator AS
SELECT
    raw->>'ts_code' AS ts_code,
    to_date(raw->>'ann_date', 'YYYYMMDD') AS ann_date,
    to_date(raw->>'end_date', 'YYYYMMDD') AS end_date,
    NULLIF(raw->>'eps', '')::numeric AS eps,
    NULLIF(raw->>'dt_eps', '')::numeric AS dt_eps,
    NULLIF(raw->>'bps', '')::numeric AS bps,
    NULLIF(raw->>'ocfps', '')::numeric AS ocfps,
    NULLIF(raw->>'grossprofit_margin', '')::numeric AS grossprofit_margin,
    NULLIF(raw->>'netprofit_margin', '')::numeric AS netprofit_margin,
    NULLIF(raw->>'roe', '')::numeric AS roe,
    NULLIF(raw->>'roe_waa', '')::numeric AS roe_waa,
    NULLIF(raw->>'roa', '')::numeric AS roa,
    NULLIF(raw->>'roic', '')::numeric AS roic,
    NULLIF(raw->>'debt_to_assets', '')::numeric AS debt_to_assets,
    NULLIF(raw->>'current_ratio', '')::numeric AS current_ratio,
    NULLIF(raw->>'quick_ratio', '')::numeric AS quick_ratio,
    NULLIF(raw->>'assets_turn', '')::numeric AS assets_turn,
    NULLIF(raw->>'inv_turn', '')::numeric AS inv_turn,
    NULLIF(raw->>'ar_turn', '')::numeric AS ar_turn,
    raw,
    fetched_at
FROM tushare_raw_records
WHERE endpoint = 'fina_indicator'
  AND raw->>'ann_date' ~ '^[0-9]{8}$'
  AND raw->>'end_date' ~ '^[0-9]{8}$';

CREATE OR REPLACE VIEW analytics.tushare_income_statement AS
SELECT
    raw->>'ts_code' AS ts_code,
    to_date(raw->>'ann_date', 'YYYYMMDD') AS ann_date,
    to_date(raw->>'end_date', 'YYYYMMDD') AS end_date,
    raw->>'report_type' AS report_type,
    NULLIF(raw->>'total_revenue', '')::numeric AS total_revenue,
    NULLIF(raw->>'revenue', '')::numeric AS revenue,
    NULLIF(raw->>'oper_cost', '')::numeric AS oper_cost,
    NULLIF(raw->>'sell_exp', '')::numeric AS sell_exp,
    NULLIF(raw->>'admin_exp', '')::numeric AS admin_exp,
    NULLIF(raw->>'fin_exp', '')::numeric AS fin_exp,
    NULLIF(raw->>'operate_profit', '')::numeric AS operate_profit,
    NULLIF(raw->>'total_profit', '')::numeric AS total_profit,
    NULLIF(raw->>'n_income', '')::numeric AS n_income,
    NULLIF(raw->>'n_income_attr_p', '')::numeric AS n_income_attr_p,
    raw,
    fetched_at
FROM tushare_raw_records
WHERE endpoint = 'income'
  AND raw->>'ann_date' ~ '^[0-9]{8}$'
  AND raw->>'end_date' ~ '^[0-9]{8}$';

CREATE OR REPLACE VIEW analytics.tushare_balance_sheet AS
SELECT
    raw->>'ts_code' AS ts_code,
    to_date(raw->>'ann_date', 'YYYYMMDD') AS ann_date,
    to_date(raw->>'end_date', 'YYYYMMDD') AS end_date,
    raw->>'report_type' AS report_type,
    NULLIF(raw->>'money_cap', '')::numeric AS money_cap,
    NULLIF(raw->>'accounts_receiv', '')::numeric AS accounts_receiv,
    NULLIF(raw->>'inventories', '')::numeric AS inventories,
    NULLIF(raw->>'total_cur_assets', '')::numeric AS total_cur_assets,
    NULLIF(raw->>'total_assets', '')::numeric AS total_assets,
    NULLIF(raw->>'total_cur_liab', '')::numeric AS total_cur_liab,
    NULLIF(raw->>'total_liab', '')::numeric AS total_liab,
    NULLIF(raw->>'total_hldr_eqy_exc_min_int', '')::numeric AS total_hldr_eqy_exc_min_int,
    raw,
    fetched_at
FROM tushare_raw_records
WHERE endpoint = 'balancesheet'
  AND raw->>'ann_date' ~ '^[0-9]{8}$'
  AND raw->>'end_date' ~ '^[0-9]{8}$';

CREATE OR REPLACE VIEW analytics.tushare_cashflow_statement AS
SELECT
    raw->>'ts_code' AS ts_code,
    to_date(raw->>'ann_date', 'YYYYMMDD') AS ann_date,
    to_date(raw->>'end_date', 'YYYYMMDD') AS end_date,
    raw->>'report_type' AS report_type,
    NULLIF(raw->>'c_fr_sale_sg', '')::numeric AS c_fr_sale_sg,
    NULLIF(raw->>'c_paid_goods_s', '')::numeric AS c_paid_goods_s,
    NULLIF(raw->>'n_cashflow_act', '')::numeric AS n_cashflow_act,
    NULLIF(raw->>'n_cashflow_inv_act', '')::numeric AS n_cashflow_inv_act,
    NULLIF(raw->>'n_cash_flows_fnc_act', '')::numeric AS n_cash_flows_fnc_act,
    NULLIF(raw->>'free_cashflow', '')::numeric AS free_cashflow,
    NULLIF(raw->>'c_cash_equ_end_period', '')::numeric AS c_cash_equ_end_period,
    raw,
    fetched_at
FROM tushare_raw_records
WHERE endpoint = 'cashflow'
  AND raw->>'ann_date' ~ '^[0-9]{8}$'
  AND raw->>'end_date' ~ '^[0-9]{8}$';

CREATE OR REPLACE VIEW analytics.tushare_main_business AS
SELECT
    raw->>'ts_code' AS ts_code,
    to_date(raw->>'end_date', 'YYYYMMDD') AS end_date,
    raw->>'bz_item' AS business_item,
    NULLIF(raw->>'bz_sales', '')::numeric AS bz_sales,
    NULLIF(raw->>'bz_profit', '')::numeric AS bz_profit,
    NULLIF(raw->>'bz_cost', '')::numeric AS bz_cost,
    raw,
    fetched_at
FROM tushare_raw_records
WHERE endpoint = 'fina_mainbz'
  AND raw->>'end_date' ~ '^[0-9]{8}$';

CREATE OR REPLACE VIEW analytics.tushare_financial_quality_latest AS
SELECT DISTINCT ON (fi.ts_code)
    fi.ts_code,
    fi.ann_date,
    fi.end_date,
    fi.eps,
    fi.ocfps,
    fi.grossprofit_margin,
    fi.netprofit_margin,
    fi.roe,
    fi.roa,
    fi.roic,
    fi.debt_to_assets,
    fi.current_ratio,
    inc.total_revenue,
    inc.revenue,
    inc.n_income_attr_p,
    bs.total_assets,
    bs.total_liab,
    cf.n_cashflow_act,
    cf.free_cashflow
FROM analytics.tushare_fina_indicator fi
LEFT JOIN analytics.tushare_income_statement inc
    ON fi.ts_code = inc.ts_code AND fi.end_date = inc.end_date
LEFT JOIN analytics.tushare_balance_sheet bs
    ON fi.ts_code = bs.ts_code AND fi.end_date = bs.end_date
LEFT JOIN analytics.tushare_cashflow_statement cf
    ON fi.ts_code = cf.ts_code AND fi.end_date = cf.end_date
ORDER BY fi.ts_code, fi.end_date DESC, fi.ann_date DESC;

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


DDL = """
CREATE TABLE IF NOT EXISTS tushare_raw_runs (
    run_id uuid PRIMARY KEY,
    started_at timestamptz NOT NULL,
    finished_at timestamptz,
    status text NOT NULL,
    start_date date,
    end_date date,
    endpoints jsonb NOT NULL,
    trade_dates jsonb NOT NULL,
    row_count integer NOT NULL DEFAULT 0,
    error_count integer NOT NULL DEFAULT 0,
    elapsed_seconds numeric,
    note text
);

CREATE TABLE IF NOT EXISTS tushare_raw_records (
    endpoint text NOT NULL,
    date_key text NOT NULL DEFAULT '',
    date_type text NOT NULL DEFAULT '',
    ts_code text NOT NULL DEFAULT '',
    content_type text NOT NULL DEFAULT '',
    row_hash text NOT NULL,
    fetched_at timestamptz NOT NULL,
    raw jsonb NOT NULL,
    PRIMARY KEY (endpoint, date_key, ts_code, content_type, row_hash)
);

CREATE INDEX IF NOT EXISTS idx_tushare_raw_records_endpoint_date
    ON tushare_raw_records (endpoint, date_key);

CREATE INDEX IF NOT EXISTS idx_tushare_raw_records_ts_code
    ON tushare_raw_records (ts_code);

CREATE TABLE IF NOT EXISTS tushare_collection_checkpoints (
    checkpoint_key text PRIMARY KEY,
    endpoint text NOT NULL,
    ts_code text NOT NULL DEFAULT '',
    content_type text NOT NULL DEFAULT '',
    range_start date,
    range_end date,
    status text NOT NULL,
    attempts integer NOT NULL DEFAULT 0,
    row_count integer NOT NULL DEFAULT 0,
    error_message text,
    updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tushare_collection_checkpoints_status
    ON tushare_collection_checkpoints (endpoint, status, updated_at);
"""


def ensure_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(DDL))

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import text

from quantitative_trading.config import build_database_url_from_parts, load_env_file
from quantitative_trading.db import make_engine

PBKDF2_ITERATIONS = 600_000
SESSION_DAYS = 14
STOCK_CODE_PATTERN = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
USERNAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9]{4,}$")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS public.app_users (
    user_id bigserial PRIMARY KEY,
    name text NOT NULL,
    email text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    role text NOT NULL DEFAULT 'user' CHECK (role IN ('super_admin', 'user')),
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    must_change_password boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    last_login_at timestamptz
);

CREATE TABLE IF NOT EXISTS public.app_sessions (
    token_hash text PRIMARY KEY,
    user_id bigint NOT NULL REFERENCES public.app_users(user_id) ON DELETE CASCADE,
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.user_watchlist (
    user_id bigint NOT NULL REFERENCES public.app_users(user_id) ON DELETE CASCADE,
    stock_code text NOT NULL,
    stock_name text,
    tags jsonb NOT NULL DEFAULT '[]'::jsonb,
    note text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, stock_code)
);

CREATE TABLE IF NOT EXISTS public.app_user_asset (
    user_id bigint PRIMARY KEY REFERENCES public.app_users(user_id) ON DELETE CASCADE,
    cash numeric NOT NULL DEFAULT 0,
    frozen_cash numeric NOT NULL DEFAULT 0,
    market_value numeric NOT NULL DEFAULT 0,
    total_asset numeric NOT NULL DEFAULT 0,
    data_source text NOT NULL DEFAULT 'manual' CHECK (data_source IN ('manual', 'import', 'migrated')),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.app_user_positions (
    user_id bigint NOT NULL REFERENCES public.app_users(user_id) ON DELETE CASCADE,
    stock_code text NOT NULL,
    stock_name text,
    volume bigint NOT NULL DEFAULT 0,
    can_use_volume bigint NOT NULL DEFAULT 0,
    avg_price numeric NOT NULL DEFAULT 0,
    current_price numeric NOT NULL DEFAULT 0,
    market_value numeric NOT NULL DEFAULT 0,
    data_source text NOT NULL DEFAULT 'manual' CHECK (data_source IN ('manual', 'import', 'migrated')),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, stock_code)
);

CREATE INDEX IF NOT EXISTS idx_app_sessions_user_expiry
    ON public.app_sessions (user_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_app_user_positions_user
    ON public.app_user_positions (user_id, market_value DESC);
"""


def database_url() -> str:
    load_env_file()
    value = (
        os.getenv("DATABASE_URL")
        or os.getenv("SMART_STOCK_DATABASE_URL")
        or build_database_url_from_parts()
    )
    if not value:
        raise RuntimeError("Database connection is missing.")
    return value


ENGINE = make_engine(database_url())


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, expected = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), int(iterations)
        ).hex()
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def json_default(value: object) -> str | float:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def public_user(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["user_id"],
        "name": row["name"],
        "username": row["username"],
        "role": row["role"],
        "status": row["status"],
        "must_change_password": row["must_change_password"],
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
        "avatar": row["name"][:2].upper(),
    }


def ensure_schema() -> tuple[str, str | None]:
    admin_username = os.getenv("WEB_ADMIN_USERNAME", "jerry01").strip()
    admin_password = os.getenv("WEB_ADMIN_PASSWORD", "Jianwei@2026!")
    admin_name = os.getenv("WEB_ADMIN_NAME", "Jerry").strip() or "Jerry"
    with ENGINE.begin() as connection:
        for statement in SCHEMA_SQL.split(";"):
            if statement.strip():
                connection.execute(text(statement))
        connection.execute(
            text("ALTER TABLE public.app_users ADD COLUMN IF NOT EXISTS username text")
        )
        connection.execute(
            text(
                """
                UPDATE public.app_users
                SET username = CASE
                    WHEN lower(split_part(email, '@', 1)) ~ '^[a-z][a-z0-9]{4,}$'
                        THEN lower(split_part(email, '@', 1))
                    ELSE 'user' || user_id::text
                END
                WHERE username IS NULL
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS app_users_username_key
                ON public.app_users (lower(username))
                """
            )
        )
        connection.execute(
            text("ALTER TABLE public.app_users ALTER COLUMN username SET NOT NULL")
        )
        username_constraint = connection.execute(
            text(
                """
                SELECT 1 FROM pg_constraint
                WHERE conname = 'app_users_username_format'
                  AND conrelid = 'public.app_users'::regclass
                """
            )
        ).scalar_one_or_none()
        if not username_constraint:
            connection.execute(
                text(
                    """
                    ALTER TABLE public.app_users
                    ADD CONSTRAINT app_users_username_format
                    CHECK (username ~ '^[A-Za-z][A-Za-z0-9]{4,}$')
                    """
                )
            )
        existing = connection.execute(
            text("SELECT user_id FROM public.app_users WHERE lower(username) = lower(:username)"),
            {"username": admin_username},
        ).scalar_one_or_none()
        initial_password: str | None = None
        if existing is None:
            existing = connection.execute(
                text(
                    """
                    SELECT user_id FROM public.app_users
                    WHERE role = 'super_admin'
                    ORDER BY created_at
                    LIMIT 1
                    """
                )
            ).scalar_one_or_none()
            if existing is None:
                existing = connection.execute(
                    text(
                        """
                        INSERT INTO public.app_users
                            (name, username, email, password_hash, role, status, must_change_password)
                        VALUES
                            (:name, :username, :email, :password_hash, 'super_admin', 'active', true)
                        RETURNING user_id
                        """
                    ),
                    {
                        "name": admin_name,
                        "username": admin_username,
                        "email": f"{admin_username.lower()}@local.invalid",
                        "password_hash": hash_password(admin_password),
                    },
                ).scalar_one()
                initial_password = admin_password
        if existing is not None:
            connection.execute(
                text(
                    """
                    UPDATE public.app_users
                    SET username = :username
                    WHERE user_id = :user_id
                    """
                ),
                {"username": admin_username, "user_id": existing},
            )
        user_asset_exists = connection.execute(
            text("SELECT 1 FROM public.app_user_asset WHERE user_id = :user_id"),
            {"user_id": existing},
        ).scalar_one_or_none()
        legacy_asset = connection.execute(
            text(
                """
                SELECT cash, frozen_cash, market_value, total_asset, fetched_at
                FROM public.portfolio_asset
                WHERE account_id = '8887301023'
                """
            )
        ).mappings().one_or_none()
        if not user_asset_exists and legacy_asset:
            connection.execute(
                text(
                    """
                    INSERT INTO public.app_user_asset (
                        user_id, cash, frozen_cash, market_value, total_asset,
                        data_source, updated_at
                    ) VALUES (
                        :user_id, :cash, :frozen_cash, :market_value, :total_asset,
                        'migrated', :updated_at
                    )
                    """
                ),
                {
                    "user_id": existing,
                    "cash": legacy_asset["cash"],
                    "frozen_cash": legacy_asset["frozen_cash"],
                    "market_value": legacy_asset["market_value"],
                    "total_asset": legacy_asset["total_asset"],
                    "updated_at": legacy_asset["fetched_at"],
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO public.app_user_positions (
                        user_id, stock_code, stock_name, volume, can_use_volume,
                        avg_price, current_price, market_value, data_source, updated_at
                    )
                    SELECT :user_id, stock_code, stock_name, volume, can_use_volume,
                           COALESCE(avg_price, 0),
                           CASE WHEN volume > 0 THEN market_value / volume ELSE 0 END,
                           COALESCE(market_value, 0), 'migrated', fetched_at
                    FROM public.portfolio_positions
                    WHERE account_id = '8887301023' AND COALESCE(volume, 0) > 0
                    ON CONFLICT (user_id, stock_code) DO NOTHING
                    """
                ),
                {"user_id": existing},
            )
        watchlist_count = connection.execute(
            text("SELECT count(*) FROM public.user_watchlist WHERE user_id = :user_id"),
            {"user_id": existing},
        ).scalar_one()
        watchlist_path = Path("config/watchlist.json")
        if watchlist_count == 0 and watchlist_path.exists():
            watchlist_data = json.loads(watchlist_path.read_text(encoding="utf-8"))
            for stock in watchlist_data.get("stocks", []):
                if stock.get("enabled", True):
                    connection.execute(
                        text(
                            """
                            INSERT INTO public.user_watchlist
                                (user_id, stock_code, stock_name, tags, note)
                            VALUES
                                (:user_id, :stock_code, :stock_name, CAST(:tags AS jsonb), :note)
                            ON CONFLICT DO NOTHING
                            """
                        ),
                        {
                            "user_id": existing,
                            "stock_code": stock["stock_code"],
                            "stock_name": stock.get("stock_name", ""),
                            "tags": json.dumps(stock.get("tags", []), ensure_ascii=False),
                            "note": stock.get("note", ""),
                        },
                    )
        connection.execute(text("DROP TABLE IF EXISTS public.user_accounts"))
    return admin_username, initial_password


def authenticate(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    with ENGINE.begin() as connection:
        connection.execute(
            text("DELETE FROM public.app_sessions WHERE expires_at <= now()")
        )
        row = connection.execute(
            text(
                """
                SELECT u.*
                FROM public.app_sessions s
                JOIN public.app_users u ON u.user_id = s.user_id
                WHERE s.token_hash = :token_hash
                  AND s.expires_at > now()
                  AND u.status = 'active'
                """
            ),
            {"token_hash": token_hash(token)},
        ).mappings().one_or_none()
    return dict(row) if row else None


def load_portfolio(user_id: int) -> dict[str, object]:
    with ENGINE.connect() as connection:
        asset = connection.execute(
            text(
                """
                SELECT user_id, cash, frozen_cash, market_value, total_asset,
                       data_source, updated_at AT TIME ZONE 'Asia/Shanghai' AS fetched_at
                FROM public.app_user_asset
                WHERE user_id = :user_id
                """
            ),
            {"user_id": user_id},
        ).mappings().one_or_none()
        positions = connection.execute(
            text(
                """
                SELECT stock_code, stock_name, volume, can_use_volume, avg_price,
                       current_price, market_value, data_source,
                       market_value - (avg_price * volume) AS profit,
                       updated_at AT TIME ZONE 'Asia/Shanghai' AS fetched_at
                FROM public.app_user_positions
                WHERE user_id = :user_id AND volume > 0
                ORDER BY market_value DESC
                """
            ),
            {"user_id": user_id},
        ).mappings().all()
    return {
        "asset": dict(asset) if asset else None,
        "positions": [dict(row) for row in positions],
    }


def load_market_overview() -> dict[str, object]:
    with ENGINE.connect() as connection:
        breadth = connection.execute(
            text(
                """
                WITH dates AS (
                    SELECT DISTINCT date_key
                    FROM public.tushare_raw_records
                    WHERE endpoint = 'daily'
                    ORDER BY date_key DESC
                    LIMIT 2
                ),
                daily AS (
                    SELECT to_date(d.date_key, 'YYYYMMDD') AS trade_date,
                           count(*) FILTER (WHERE (d.raw->>'pct_chg')::numeric > 0) AS up_count,
                           count(*) FILTER (WHERE (d.raw->>'pct_chg')::numeric < 0) AS down_count,
                           count(*) FILTER (WHERE (d.raw->>'pct_chg')::numeric = 0) AS flat_count,
                           count(*) AS stock_count,
                           sum((d.raw->>'amount')::numeric) * 1000 AS turnover_yuan
                    FROM public.tushare_raw_records d
                    JOIN dates USING (date_key)
                    WHERE d.endpoint = 'daily'
                    GROUP BY d.date_key
                )
                SELECT current.*,
                       previous.turnover_yuan AS previous_turnover_yuan
                FROM daily current
                LEFT JOIN daily previous
                  ON previous.trade_date = to_date((
                      SELECT min(date_key) FROM dates
                  ), 'YYYYMMDD')
                WHERE current.trade_date = to_date((
                    SELECT max(date_key) FROM dates
                ), 'YYYYMMDD')
                """
            )
        ).mappings().one_or_none()
        limits = connection.execute(
            text(
                """
                WITH latest AS (
                    SELECT max(date_key) AS latest_date
                    FROM public.tushare_raw_records
                    WHERE endpoint = 'limit_list_d'
                )
                SELECT to_date(l.latest_date, 'YYYYMMDD') AS trade_date,
                       count(*) FILTER (WHERE r.raw->>'limit' = 'U') AS limit_up_count,
                       count(*) FILTER (WHERE r.raw->>'limit' = 'D') AS limit_down_count,
                       count(*) FILTER (WHERE r.raw->>'limit' = 'Z') AS open_board_count
                FROM latest l
                JOIN public.tushare_raw_records r
                  ON r.endpoint = 'limit_list_d'
                 AND r.date_key = l.latest_date
                GROUP BY l.latest_date
                """
            )
        ).mappings().one_or_none()
        moneyflow = connection.execute(
            text(
                """
                WITH latest AS (
                    SELECT max(trade_date) AS trade_date
                    FROM analytics.tushare_moneyflow_stock
                )
                SELECT l.trade_date, count(*) AS stock_count,
                       sum(m.net_mf_amount) * 10000 AS main_net_amount_yuan
                FROM latest l
                JOIN analytics.tushare_moneyflow_stock m
                  ON m.trade_date = l.trade_date
                GROUP BY l.trade_date
                """
            )
        ).mappings().one_or_none()
        sectors = connection.execute(
            text(
                """
                WITH latest AS (
                    SELECT max(captured_at) AS captured_at
                    FROM analytics.realtime_sector_moneyflow_latest
                    WHERE scope = 'industry'
                )
                SELECT s.name, s.pct_change,
                       CASE
                           WHEN s.source = 'akshare_ths' THEN s.main_net_amount * 100000000
                           ELSE s.main_net_amount
                       END AS main_net_amount_yuan,
                       s.captured_at AT TIME ZONE 'Asia/Shanghai' AS captured_at
                FROM analytics.realtime_sector_moneyflow_latest s
                CROSS JOIN latest l
                WHERE s.scope = 'industry'
                  AND s.captured_at = l.captured_at
                ORDER BY s.main_net_amount DESC NULLS LAST
                LIMIT 8
                """
            )
        ).mappings().all()
    if not breadth:
        return {
            "trade_date": None,
            "breadth": None,
            "limits": None,
            "moneyflow": None,
            "sectors": [dict(row) for row in sectors],
            "sources": {},
        }
    breadth_data = dict(breadth)
    current_turnover = breadth_data.get("turnover_yuan") or 0
    previous_turnover = breadth_data.pop("previous_turnover_yuan", None)
    breadth_data["turnover_change_percent"] = (
        float((current_turnover / previous_turnover - 1) * 100)
        if previous_turnover
        else None
    )
    return {
        "trade_date": breadth_data["trade_date"],
        "breadth": breadth_data,
        "limits": dict(limits) if limits else None,
        "moneyflow": dict(moneyflow) if moneyflow else None,
        "sectors": [dict(row) for row in sectors],
        "sources": {
            "breadth": "public.tushare_raw_records (daily)",
            "limits": "public.tushare_raw_records (limit_list_d)",
            "moneyflow": "analytics.tushare_moneyflow_stock",
        },
    }


def load_analysis(user_id: int) -> dict[str, object]:
    with ENGINE.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT w.stock_code, w.stock_name,
                       to_date(p.date_key, 'YYYYMMDD') AS trade_date,
                       (p.raw->>'close')::numeric AS close,
                       (p.raw->>'pct_chg')::numeric AS pct_chg,
                       (b.raw->>'pe_ttm')::numeric AS pe_ttm,
                       (b.raw->>'pb')::numeric AS pb,
                       (b.raw->>'turnover_rate')::numeric AS turnover_rate,
                       (b.raw->>'volume_ratio')::numeric AS volume_ratio,
                       (m.raw->>'net_mf_amount')::numeric * 10000 AS net_mf_amount_yuan,
                       (c.raw->>'winner_rate')::numeric AS winner_rate
                FROM public.user_watchlist w
                LEFT JOIN LATERAL (
                    SELECT date_key, raw
                    FROM public.tushare_raw_records
                    WHERE endpoint = 'daily' AND ts_code = w.stock_code
                    ORDER BY date_key DESC
                    LIMIT 1
                ) p ON true
                LEFT JOIN LATERAL (
                    SELECT raw
                    FROM public.tushare_raw_records
                    WHERE endpoint = 'daily_basic' AND ts_code = w.stock_code
                    ORDER BY date_key DESC
                    LIMIT 1
                ) b ON true
                LEFT JOIN LATERAL (
                    SELECT raw
                    FROM public.tushare_raw_records
                    WHERE endpoint = 'moneyflow' AND ts_code = w.stock_code
                    ORDER BY date_key DESC
                    LIMIT 1
                ) m ON true
                LEFT JOIN LATERAL (
                    SELECT raw
                    FROM public.tushare_raw_records
                    WHERE endpoint = 'cyq_perf' AND ts_code = w.stock_code
                    ORDER BY date_key DESC
                    LIMIT 1
                ) c ON true
                WHERE w.user_id = :user_id
                ORDER BY w.created_at
                """
            ),
            {"user_id": user_id},
        ).mappings().all()
    records = [dict(row) for row in rows]
    return {
        "trade_date": max(
            (row["trade_date"] for row in records if row["trade_date"]),
            default=None,
        ),
        "records": records,
        "summary": {
            "covered": sum(row["trade_date"] is not None for row in records),
            "rising": sum((row["pct_chg"] or 0) > 0 for row in records),
            "positive_moneyflow": sum(
                (row["net_mf_amount_yuan"] or 0) > 0 for row in records
            ),
            "total": len(records),
        },
    }


def replace_manual_portfolio(user_id: int, payload: dict[str, Any]) -> dict[str, object]:
    source = "import" if payload.get("source") == "import" else "manual"
    cash = max(float(payload.get("cash", 0) or 0), 0)
    frozen_cash = max(float(payload.get("frozen_cash", 0) or 0), 0)
    raw_positions = payload.get("positions")
    if not isinstance(raw_positions, list):
        raise ValueError("positions 必须是数组")
    positions: list[dict[str, Any]] = []
    for index, item in enumerate(raw_positions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 行格式错误")
        code = str(item.get("stock_code", "")).strip().upper()
        name = str(item.get("stock_name", "")).strip()
        if not STOCK_CODE_PATTERN.match(code):
            raise ValueError(f"第 {index} 行股票代码格式错误：{code}")
        volume = int(float(item.get("volume", 0) or 0))
        can_use = int(float(item.get("can_use_volume", volume) or 0))
        avg_price = float(item.get("avg_price", 0) or 0)
        current_price = float(item.get("current_price", 0) or 0)
        if volume < 0 or can_use < 0 or avg_price < 0 or current_price < 0:
            raise ValueError(f"第 {index} 行不能包含负数")
        if volume == 0:
            continue
        positions.append(
            {
                "stock_code": code,
                "stock_name": name or code,
                "volume": volume,
                "can_use_volume": min(can_use, volume),
                "avg_price": avg_price,
                "current_price": current_price,
                "market_value": round(volume * current_price, 4),
            }
        )
    market_value = sum(item["market_value"] for item in positions)
    total_asset = cash + market_value
    now = datetime.now(timezone.utc)
    with ENGINE.begin() as connection:
        connection.execute(
            text("DELETE FROM public.app_user_positions WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        for item in positions:
            connection.execute(
                text(
                    """
                    INSERT INTO public.app_user_positions (
                        user_id, stock_code, stock_name, volume, can_use_volume,
                        avg_price, current_price, market_value, data_source, updated_at
                    ) VALUES (
                        :user_id, :stock_code, :stock_name, :volume, :can_use_volume,
                        :avg_price, :current_price, :market_value, :data_source, :updated_at
                    )
                    """
                ),
                {**item, "user_id": user_id, "data_source": source, "updated_at": now},
            )
        connection.execute(
            text(
                """
                INSERT INTO public.app_user_asset (
                    user_id, cash, frozen_cash, market_value, total_asset,
                    data_source, updated_at
                ) VALUES (
                    :user_id, :cash, :frozen_cash, :market_value, :total_asset,
                    :data_source, :updated_at
                )
                ON CONFLICT (user_id) DO UPDATE SET
                    cash = EXCLUDED.cash,
                    frozen_cash = EXCLUDED.frozen_cash,
                    market_value = EXCLUDED.market_value,
                    total_asset = EXCLUDED.total_asset,
                    data_source = EXCLUDED.data_source,
                    updated_at = EXCLUDED.updated_at
                """
            ),
            {
                "user_id": user_id,
                "cash": cash,
                "frozen_cash": frozen_cash,
                "market_value": market_value,
                "total_asset": total_asset,
                "data_source": source,
                "updated_at": now,
            },
        )
    return load_portfolio(user_id)


class ApiHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json({"status": "ok"})
            return
        user = self.require_user()
        if not user:
            return
        if path == "/api/auth/me":
            self.send_json({"user": public_user(user)})
        elif path == "/api/portfolio":
            self.send_json(load_portfolio(user["user_id"]))
        elif path == "/api/market/overview":
            self.send_json(load_market_overview())
        elif path == "/api/analysis":
            self.send_json(load_analysis(user["user_id"]))
        elif path == "/api/users":
            if not self.require_admin(user):
                return
            with ENGINE.connect() as connection:
                rows = connection.execute(
                    text("SELECT * FROM public.app_users ORDER BY created_at")
                ).mappings().all()
            self.send_json({"users": [public_user(dict(row)) for row in rows]})
        elif path == "/api/watchlist":
            with ENGINE.connect() as connection:
                rows = connection.execute(
                    text(
                        """
                        SELECT w.stock_code, w.stock_name, w.tags, w.note,
                               COALESCE(q.last_price, 0) AS price,
                               COALESCE(q.change_percent, 0) AS change,
                               COALESCE(h.closes, ARRAY[]::numeric[]) AS spark
                        FROM public.user_watchlist w
                        LEFT JOIN LATERAL (
                            SELECT last_price, change_percent
                            FROM public.stock_snapshots s
                            WHERE s.stock_code = w.stock_code
                              AND s.last_price > 0
                            ORDER BY s.raw_time DESC NULLS LAST, s.captured_at DESC
                            LIMIT 1
                        ) q ON true
                        LEFT JOIN LATERAL (
                            SELECT array_agg((raw->>'close')::numeric ORDER BY date_key) AS closes
                            FROM (
                                SELECT date_key, raw
                                FROM public.tushare_raw_records
                                WHERE endpoint = 'daily' AND ts_code = w.stock_code
                                ORDER BY date_key DESC
                                LIMIT 7
                            ) prices
                        ) h ON true
                        WHERE w.user_id = :user_id
                        ORDER BY w.created_at
                        """
                    ),
                    {"user_id": user["user_id"]},
                ).mappings().all()
            self.send_json({"stocks": [dict(row) for row in rows]})
        else:
            self.send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        payload = self.read_json()
        if payload is None:
            return
        if path == "/api/auth/login":
            self.login(payload)
            return
        user = self.require_user()
        if not user:
            return
        try:
            if path == "/api/auth/logout":
                token = self.session_token()
                if token:
                    with ENGINE.begin() as connection:
                        connection.execute(
                            text("DELETE FROM public.app_sessions WHERE token_hash = :token_hash"),
                            {"token_hash": token_hash(token)},
                        )
                self.send_json({"ok": True}, clear_cookie=True)
            elif path == "/api/auth/change-password":
                current_password = str(payload.get("current_password", ""))
                password = str(payload.get("password", ""))
                if not verify_password(current_password, user["password_hash"]):
                    self.send_json({"error": "当前密码不正确"}, status=400)
                    return
                if len(password) < 10:
                    self.send_json({"error": "密码至少需要 10 位"}, status=400)
                    return
                if current_password == password:
                    self.send_json({"error": "新密码不能与当前密码相同"}, status=400)
                    return
                current_token_hash = token_hash(self.session_token() or "")
                with ENGINE.begin() as connection:
                    connection.execute(
                        text(
                            """
                            UPDATE public.app_users
                            SET password_hash = :password_hash, must_change_password = false
                            WHERE user_id = :user_id
                            """
                        ),
                        {"password_hash": hash_password(password), "user_id": user["user_id"]},
                    )
                    connection.execute(
                        text(
                            """
                            DELETE FROM public.app_sessions
                            WHERE user_id = :user_id AND token_hash <> :current_token_hash
                            """
                        ),
                        {
                            "user_id": user["user_id"],
                            "current_token_hash": current_token_hash,
                        },
                    )
                self.send_json({"ok": True})
            elif path == "/api/portfolio/manual":
                self.send_json(replace_manual_portfolio(user["user_id"], payload))
            elif path == "/api/users":
                if not self.require_admin(user):
                    return
                self.create_user(payload)
            elif match := re.fullmatch(r"/api/users/(\d+)/reset-password", path):
                if not self.require_admin(user):
                    return
                target_id = int(match.group(1))
                password = str(payload.get("password", ""))
                if len(password) < 10:
                    self.send_json({"error": "临时密码至少需要 10 位"}, status=400)
                    return
                with ENGINE.begin() as connection:
                    target = connection.execute(
                        text("SELECT role FROM public.app_users WHERE user_id = :user_id"),
                        {"user_id": target_id},
                    ).scalar_one_or_none()
                    if target is None:
                        self.send_json({"error": "用户不存在"}, status=404)
                        return
                    if target == "super_admin" and target_id != user["user_id"]:
                        self.send_json({"error": "不能重置其他超级用户密码"}, status=400)
                        return
                    connection.execute(
                        text(
                            """
                            UPDATE public.app_users
                            SET password_hash = :password_hash, must_change_password = true
                            WHERE user_id = :user_id
                            """
                        ),
                        {"password_hash": hash_password(password), "user_id": target_id},
                    )
                    connection.execute(
                        text("DELETE FROM public.app_sessions WHERE user_id = :user_id"),
                        {"user_id": target_id},
                    )
                self.send_json({"ok": True})
            elif path == "/api/watchlist":
                code = str(payload.get("stock_code", "")).strip().upper()
                if not STOCK_CODE_PATTERN.match(code):
                    self.send_json({"error": "股票代码格式应为 600000.SH"}, status=400)
                    return
                with ENGINE.begin() as connection:
                    connection.execute(
                        text(
                            """
                            INSERT INTO public.user_watchlist
                                (user_id, stock_code, stock_name, tags, note)
                            VALUES
                                (:user_id, :stock_code, :stock_name, CAST(:tags AS jsonb), :note)
                            ON CONFLICT (user_id, stock_code) DO UPDATE SET
                                stock_name = EXCLUDED.stock_name,
                                tags = EXCLUDED.tags,
                                note = EXCLUDED.note
                            """
                        ),
                        {
                            "user_id": user["user_id"],
                            "stock_code": code,
                            "stock_name": str(payload.get("stock_name", "")).strip(),
                            "tags": json.dumps(payload.get("tags", []), ensure_ascii=False),
                            "note": str(payload.get("note", "")).strip(),
                        },
                    )
                self.send_json({"ok": True}, status=201)
            else:
                self.send_json({"error": "Not found"}, status=404)
        except (ValueError, TypeError) as error:
            self.send_json({"error": str(error)}, status=400)
        except Exception as error:  # pragma: no cover
            self.send_json({"error": str(error)}, status=500)

    def do_PATCH(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        user = self.require_user()
        if not user or not self.require_admin(user):
            return
        match = re.fullmatch(r"/api/users/(\d+)/status", path)
        if not match:
            self.send_json({"error": "Not found"}, status=404)
            return
        payload = self.read_json()
        if payload is None:
            return
        status = payload.get("status")
        if status not in {"active", "disabled"}:
            self.send_json({"error": "无效状态"}, status=400)
            return
        target_id = int(match.group(1))
        with ENGINE.begin() as connection:
            target_role = connection.execute(
                text("SELECT role FROM public.app_users WHERE user_id = :user_id"),
                {"user_id": target_id},
            ).scalar_one_or_none()
            if target_role == "super_admin":
                self.send_json({"error": "不能停用超级用户"}, status=400)
                return
            connection.execute(
                text("UPDATE public.app_users SET status = :status WHERE user_id = :user_id"),
                {"status": status, "user_id": target_id},
            )
        self.send_json({"ok": True})

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        user = self.require_user()
        if not user:
            return
        match = re.fullmatch(r"/api/watchlist/([^/]+)", path)
        if not match:
            self.send_json({"error": "Not found"}, status=404)
            return
        with ENGINE.begin() as connection:
            connection.execute(
                text(
                    """
                    DELETE FROM public.user_watchlist
                    WHERE user_id = :user_id AND stock_code = :stock_code
                    """
                ),
                {"user_id": user["user_id"], "stock_code": match.group(1).upper()},
            )
        self.send_json({"ok": True})

    def login(self, payload: dict[str, Any]) -> None:
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        with ENGINE.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM public.app_users WHERE lower(username) = lower(:username)"),
                {"username": username},
            ).mappings().one_or_none()
            if not row or row["status"] != "active" or not verify_password(password, row["password_hash"]):
                self.send_json({"error": "用户名或密码错误"}, status=401)
                return
            token = secrets.token_urlsafe(32)
            expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
            connection.execute(
                text(
                    """
                    INSERT INTO public.app_sessions (token_hash, user_id, expires_at)
                    VALUES (:token_hash, :user_id, :expires_at)
                    """
                ),
                {"token_hash": token_hash(token), "user_id": row["user_id"], "expires_at": expires_at},
            )
            connection.execute(
                text("UPDATE public.app_users SET last_login_at = now() WHERE user_id = :user_id"),
                {"user_id": row["user_id"]},
            )
        self.send_json({"user": public_user(dict(row))}, session_token=token)

    def create_user(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("name", "")).strip()
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        if not name:
            self.send_json({"error": "请填写显示名称"}, status=400)
            return
        if not USERNAME_PATTERN.fullmatch(username):
            self.send_json(
                {"error": "用户名必须以英文字母开头，仅包含英文和数字，且至少 5 个字符"},
                status=400,
            )
            return
        if len(password) < 10:
            self.send_json({"error": "密码至少需要 10 位"}, status=400)
            return
        try:
            with ENGINE.begin() as connection:
                row = connection.execute(
                    text(
                        """
                        INSERT INTO public.app_users
                            (name, username, email, password_hash, role, status, must_change_password)
                        VALUES
                            (:name, :username, :email, :password_hash, 'user', 'active', true)
                        RETURNING *
                        """
                    ),
                    {
                        "name": name,
                        "username": username,
                        "email": f"{username.lower()}@local.invalid",
                        "password_hash": hash_password(password),
                    },
                ).mappings().one()
            self.send_json({"user": public_user(dict(row))}, status=201)
        except Exception as error:
            if "unique" in str(error).lower():
                self.send_json({"error": "该用户名已存在"}, status=409)
            else:
                raise

    def require_admin(self, user: dict[str, Any]) -> bool:
        if user["role"] != "super_admin":
            self.send_json({"error": "需要超级用户权限"}, status=403)
            return False
        return True

    def require_user(self) -> dict[str, Any] | None:
        user = authenticate(self.session_token())
        if not user:
            self.send_json({"error": "请先登录"}, status=401)
        return user

    def session_token(self) -> str | None:
        authorization = self.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            return authorization[7:]
        raw_cookie = self.headers.get("Cookie")
        if not raw_cookie:
            return None
        jar = cookies.SimpleCookie()
        jar.load(raw_cookie)
        return jar["jianwei_session"].value if "jianwei_session" in jar else None

    def read_json(self) -> dict[str, Any] | None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 2_000_000:
                self.send_json({"error": "请求内容过大"}, status=413)
                return None
            data = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(data, dict):
                raise ValueError
            return data
        except (ValueError, json.JSONDecodeError):
            self.send_json({"error": "JSON 格式错误"}, status=400)
            return None

    def send_json(
        self,
        payload: object,
        status: int = 200,
        session_token: str | None = None,
        clear_cookie: bool = False,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=json_default).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        if session_token:
            self.send_header(
                "Set-Cookie",
                f"jianwei_session={session_token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_DAYS * 86400}",
            )
        elif clear_cookie:
            self.send_header(
                "Set-Cookie",
                "jianwei_session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0",
            )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"WEB_API {self.address_string()} {format % args}")


def main() -> None:
    admin_username, initial_password = ensure_schema()
    host = os.getenv("WEB_API_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_API_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), ApiHandler)
    print(f"WEB_API listening on http://{host}:{port}")
    if initial_password:
        print(f"WEB_API bootstrap admin: {admin_username} / {initial_password}")
    server.serve_forever()


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    database_url: str
    tushare_token: str


def get_settings() -> Settings:
    load_env_file()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is missing. Add it to .env or export it.")

    database_url = (
        os.getenv("DATABASE_URL")
        or os.getenv("SMART_STOCK_DATABASE_URL")
        or build_database_url_from_parts()
    )
    if not database_url:
        raise RuntimeError(
            "Database connection is missing. Set DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD."
        )
    return Settings(database_url=database_url, tushare_token=token)


def build_database_url_from_parts() -> str:
    host = os.getenv("DB_HOST", "").strip()
    port = os.getenv("DB_PORT", "").strip()
    name = os.getenv("DB_NAME", "").strip()
    user = os.getenv("DB_USER", "").strip()
    password = os.getenv("DB_PASSWORD", "").strip()
    if not all([host, port, name, user, password]):
        return ""
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"

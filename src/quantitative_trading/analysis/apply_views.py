from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from quantitative_trading.config import get_settings
from quantitative_trading.db import make_engine


def apply_analysis_views() -> None:
    sql_path = Path("sql/analysis_views.sql")
    if not sql_path.exists():
        raise RuntimeError(f"Missing SQL file: {sql_path}")

    engine = make_engine(get_settings().database_url)
    sql = sql_path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql))


def main() -> None:
    apply_analysis_views()
    print("analysis views applied")


if __name__ == "__main__":
    main()

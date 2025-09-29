"""筛选在近期多次涨停但未出现连续三板的个股。"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar
from urllib.parse import quote_plus

import pandas as pd
import time
import requests

try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:  # pragma: no cover
    create_engine = None  # type: ignore[assignment]
    Engine = Any  # type: ignore[assignment]
    SQLAlchemyError = Exception  # type: ignore[assignment]

from .config import LimitUpConfig, get_default_config

try:
    import akshare as ak
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 akshare: pip install akshare") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_FILE = BASE_DIR / "data" / "limit_up_frequency.xlsx"

_ORIGINAL_REQUESTS_GET = requests.get
_REQUEST_HEADERS: dict[str, str] = {}
_REQUESTS_PATCHED = False
_MYSQL_ENGINES: dict[str, Engine] = {}


def _apply_requests_header_patch(headers: dict[str, str]) -> None:
    """Ensure AkShare's internal requests include custom headers."""

    global _REQUEST_HEADERS, _REQUESTS_PATCHED
    _REQUEST_HEADERS = dict(headers)

    if _REQUESTS_PATCHED:
        return

    def _patched_get(url: str, *args, **kwargs):
        merged = dict(_REQUEST_HEADERS)
        user_headers = kwargs.pop("headers", None)
        if user_headers:
            merged.update(user_headers)
        kwargs["headers"] = merged
        return _ORIGINAL_REQUESTS_GET(url, *args, **kwargs)

    requests.get = _patched_get  # type: ignore[assignment]
    _REQUESTS_PATCHED = True


def _extract_proxy_candidate(payload: Any) -> Optional[str]:
    if isinstance(payload, str):
        text = payload.replace("\r", "\n")
        for token in text.splitlines():
            candidate = token.strip().split()[0] if token.strip() else ""
            if candidate and ":" in candidate:
                candidate = candidate.rstrip(",;")
                if candidate.startswith("http://") or candidate.startswith("https://"):
                    candidate = candidate.split("://", 1)[1]
                if "/" in candidate:
                    candidate = candidate.split("/", 1)[0]
                return candidate
        return None
    if isinstance(payload, dict):
        for value in payload.values():
            candidate = _extract_proxy_candidate(value)
            if candidate:
                return candidate
        return None
    if isinstance(payload, (list, tuple)):
        for item in payload:
            candidate = _extract_proxy_candidate(item)
            if candidate:
                return candidate
        return None
    return None


def _request_new_proxy(config: LimitUpConfig) -> str:
    if not config.proxy_api_url:
        raise RuntimeError("未配置 proxy_api_url，无法获取新的代理 IP")

    headers = {"User-Agent": config.http_headers.get("User-Agent", "Mozilla/5.0")}
    if config.proxy_api_headers:
        headers.update(config.proxy_api_headers)

    try:
        response = _ORIGINAL_REQUESTS_GET(
            config.proxy_api_url,
            params=config.proxy_api_params or None,
            headers=headers,
            timeout=config.proxy_api_timeout,
            proxies=None,
        )
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on network
        raise RuntimeError(f"代理获取失败: {exc}") from exc

    candidate: Optional[str]
    try:
        data = response.json()
    except ValueError:
        candidate = None
    else:
        candidate = _extract_proxy_candidate(data)

    if not candidate:
        candidate = _extract_proxy_candidate(response.text)

    if not candidate:
        raise RuntimeError("代理响应无法解析为 ip:port 格式")

    return candidate


def _build_proxy_uri(config: LimitUpConfig, endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint

    auth = ""
    if config.proxy_username and config.proxy_password and "@" not in endpoint:
        auth = f"{config.proxy_username}:{config.proxy_password}@"

    scheme = config.proxy_scheme or "http"
    return f"{scheme}://{auth}{endpoint}"


@contextmanager
def _proxy_context(config: LimitUpConfig) -> Iterable[None]:
    keys = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]
    backups = {k: os.environ.get(k) for k in keys}

    proxy_endpoint: Optional[str] = None
    if config.proxy_api_url:
        proxy_endpoint = _request_new_proxy(config)
        proxy_uri = _build_proxy_uri(config, proxy_endpoint)
        for key in keys:
            os.environ[key] = proxy_uri
        display = proxy_endpoint.split("@")[-1]
        print(f"已启用代理：{display}")
    else:
        for key in keys:
            if key in os.environ:
                del os.environ[key]

    try:
        yield
    finally:
        for key, value in backups.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


T = TypeVar("T")


def _kv_pair(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("需要 key=value 格式")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("键名不能为空")
    return key, value.strip()


@dataclass(slots=True)
class LimitUpResult:
    code: str
    name: str
    limit_up_count: int
    max_consecutive: int
    limit_up_dates: str
    last_close: float
    last_trade_date: dt.date
    limit_threshold: float
    latest_main_flow: Optional[float] = None


COLUMN_LABELS = {
    "code": "股票代码",
    "name": "股票名称",
    "limit_up_count": "涨停次数",
    "max_consecutive": "连续涨停天数",
    "limit_up_dates": "涨停日期列表",
    "last_close": "最新收盘价",
    "last_trade_date": "最后交易日",
    "limit_threshold": "涨停阈值",
    "latest_main_flow": "主力净流入-净额",
}


def _fetch_with_retry(label: str, func: Callable[[], T], config: LimitUpConfig) -> T:
    attempts = max(1, config.fetch_retry_attempts)
    delay = max(0.5, config.fetch_retry_delay)
    backoff = config.fetch_retry_backoff if config.fetch_retry_backoff > 1 else 1.0
    wait = delay
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - network errors only
            last_exc = exc
            if attempt == attempts:
                break
            print(f"{label} 获取失败（第 {attempt} 次）：{exc}；{wait:.1f} 秒后重试...")
            time.sleep(wait)
            wait *= backoff
    assert last_exc is not None
    raise last_exc


def _quote_identifier(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def _build_mysql_uri(config: LimitUpConfig) -> str:
    user = quote_plus(config.mysql_user or "")
    password = quote_plus(config.mysql_password or "")
    auth = f"{user}:{password}@" if password or user else ""
    return (
        f"mysql+pymysql://{auth}{config.mysql_host}:{config.mysql_port}/"
        f"{config.mysql_database}?charset=utf8mb4"
    )


def _get_mysql_engine(config: LimitUpConfig) -> Engine:
    if create_engine is None:
        raise SystemExit("请先安装 SQLAlchemy: pip install SQLAlchemy")
    uri = _build_mysql_uri(config)
    engine = _MYSQL_ENGINES.get(uri)
    if engine is None:
        engine = create_engine(uri, pool_pre_ping=True, pool_recycle=3600)
        _MYSQL_ENGINES[uri] = engine
    return engine


def _fetch_mysql_universe(config: LimitUpConfig) -> pd.DataFrame:
    select_parts = [f"DISTINCT {_quote_identifier(config.mysql_code_column)} AS code"]
    if config.mysql_name_column:
        select_parts.append(f"{_quote_identifier(config.mysql_name_column)} AS name")
    query = f"SELECT {', '.join(select_parts)} FROM {_quote_identifier(config.mysql_table)}"

    try:
        universe = pd.read_sql(query, _get_mysql_engine(config))
    except SQLAlchemyError as exc:  # pragma: no cover - depends on DB
        raise RuntimeError(f"获取股票列表失败：{exc}") from exc

    if universe.empty:
        return universe

    if "name" not in universe:
        universe["name"] = universe["code"]

    if config.exclude_st:
        universe = universe.loc[~universe["name"].astype(str).str.contains("ST", case=False, na=False)].copy()

    universe["code"] = universe["code"].astype(str)
    universe["name"] = universe["name"].astype(str)
    universe.reset_index(drop=True, inplace=True)
    return universe[["code", "name"]]


def _fetch_history_mysql(code: str, config: LimitUpConfig) -> pd.DataFrame:
    selects = [
        f"{_quote_identifier(config.mysql_date_column)} AS date",
        f"{_quote_identifier(config.mysql_change_column)} AS change_pct",
    ]
    if config.mysql_flow_column:
        selects.append(f"{_quote_identifier(config.mysql_flow_column)} AS main_flow")
    if config.mysql_close_column:
        selects.append(f"{_quote_identifier(config.mysql_close_column)} AS close")

    limit = max(config.lookback_days + 10, config.lookback_days * 2)
    query = (
        f"SELECT {', '.join(selects)} FROM {_quote_identifier(config.mysql_table)} "
        f"WHERE {_quote_identifier(config.mysql_code_column)} = %s "
        f"ORDER BY {_quote_identifier(config.mysql_date_column)} DESC LIMIT %s"
    )

    try:
        df = pd.read_sql(query, _get_mysql_engine(config), params=(code, limit))
    except SQLAlchemyError as exc:  # pragma: no cover - depends on DB
        raise RuntimeError(f"查询 {code} 历史数据失败：{exc}") from exc

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    if config.mysql_flow_column and "main_flow" in df:
        df["main_flow"] = pd.to_numeric(df["main_flow"], errors="coerce")
    if config.mysql_close_column and "close" in df:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

    return df


def _fetch_spot_snapshot(config: LimitUpConfig) -> pd.DataFrame:
    if config.use_mysql:
        return _fetch_mysql_universe(config)

    def _fetch_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
        return ak.stock_zh_a_spot_em(), ak.stock_bj_a_spot_em()

    spot_main, spot_bj = _fetch_with_retry("实时行情", _fetch_pair, config)
    spot = pd.concat([spot_main, spot_bj], ignore_index=True, sort=False)
    spot = spot.rename(columns={"代码": "code", "名称": "name"})
    spot["code"] = spot["code"].astype(str)
    if config.exclude_st:
        spot = spot.loc[~spot["name"].str.contains("ST", case=False, na=False)].copy()
    spot.dropna(subset=["code", "name"], inplace=True)
    return spot[["code", "name"]]


def _infer_limit_threshold(code: str, name: str) -> float:
    upper_name = (name or "").upper()
    if "ST" in upper_name:
        return 5.0
    if code.startswith(("300", "301", "302", "303", "688", "689", "787")):
        return 20.0
    if code.startswith(("4", "8")) and len(code) == 6:
        return 30.0
    return 10.0


def _fetch_history(code: str, start: dt.date, end: dt.date, config: LimitUpConfig) -> pd.DataFrame:
    if config.use_mysql:
        return _fetch_history_mysql(code, config)

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    def _do_fetch() -> pd.DataFrame:
        if code.startswith(("4", "8")):
            return ak.stock_bj_a_hist(symbol=code, start_date=start_str, end_date=end_str, adjust="qfq")
        return ak.stock_zh_a_hist(symbol=code, start_date=start_str, end_date=end_str, adjust="qfq")

    history = _fetch_with_retry(f"{code} 历史行情", _do_fetch, config)
    if history is None or history.empty:
        return pd.DataFrame()
    rename_map = {
        "日期": "date",
        "收盘": "close",
        "涨跌幅": "change_pct",
    }
    history = history.rename(columns=rename_map)
    if "date" not in history:
        return pd.DataFrame()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history.dropna(subset=["date"], inplace=True)
    for column in ["close", "change_pct"]:
        if column in history:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    history.dropna(subset=["change_pct", "close"], inplace=True)
    history.sort_values("date", inplace=True)
    history.reset_index(drop=True, inplace=True)
    return history[["date", "close", "change_pct"]]


def _max_consecutive_true(flags: Iterable[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _evaluate_single(code: str, name: str, config: LimitUpConfig, end_date: dt.date) -> Optional[LimitUpResult]:
    threshold = _infer_limit_threshold(code, name)
    start_date = end_date - dt.timedelta(days=config.history_buffer_days)
    try:
        history = _fetch_history(code, start_date, end_date, config)
    except Exception as exc:  # pragma: no cover - network layer failures only
        print(f"{code} 历史行情多次失败，跳过该票：{exc}")
        return None
    if history.empty:
        return None
    recent = history.tail(config.lookback_days)
    if len(recent) < config.lookback_days:
        return None
    recent = recent.copy()
    recent["change_pct"] = pd.to_numeric(recent["change_pct"], errors="coerce")
    mask = recent["change_pct"] >= (threshold - config.percent_tolerance)
    limit_count = int(mask.sum())
    if not (config.min_limit_ups <= limit_count <= config.max_limit_ups):
        return None
    max_consecutive = _max_consecutive_true(mask.tolist())
    if max_consecutive > config.max_consecutive_limit_ups:
        return None
    if limit_count == 0:
        return None
    limit_dates = recent.loc[mask, "date"].dt.strftime("%Y-%m-%d").tolist()
    last_row = recent.iloc[-1]
    last_close_value = last_row.get("close")
    last_close = float(last_close_value) if pd.notna(last_close_value) else float("nan")
    return LimitUpResult(
        code=code,
        name=name,
        limit_up_count=limit_count,
        max_consecutive=max_consecutive,
        limit_up_dates=", ".join(limit_dates),
        last_close=last_close,
        last_trade_date=last_row["date"].date(),
        limit_threshold=threshold,
        latest_main_flow=float(last_row.get("main_flow")) if "main_flow" in recent and pd.notna(last_row.get("main_flow")) else None,
    )


def run_screen(
    config: LimitUpConfig,
    max_stocks: Optional[int] = None,
    show_progress: bool = False,
) -> list[LimitUpResult]:
    if config.use_mysql:
        end_date = dt.date.today()
        try:
            spot = _fetch_spot_snapshot(config)
        except Exception as exc:  # pragma: no cover
            print(f"数据库股票列表获取失败：{exc}")
            return []
        if spot.empty:
            return []
        if max_stocks is not None:
            spot = spot.head(max_stocks)
        iterator = spot.itertuples(index=False)
        futures = {}
        results: list[LimitUpResult] = []
        workers = max(1, config.workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in iterator:
                futures[pool.submit(_evaluate_single, row.code, row.name, config, end_date)] = row.code
            if show_progress and tqdm is not None:
                for future in tqdm(as_completed(futures), total=len(futures), desc="筛选中"):
                    result = future.result()
                    if result is not None:
                        results.append(result)
            else:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
        results.sort(key=lambda item: (item.limit_up_count, -item.max_consecutive), reverse=True)
        return results

    _apply_requests_header_patch(config.http_headers)
    with _proxy_context(config):
        end_date = dt.date.today()
        try:
            spot = _fetch_spot_snapshot(config)
        except Exception as exc:  # pragma: no cover
            print(f"实时行情拉取失败，已放弃本次筛选：{exc}")
            return []
        if spot.empty:
            return []
        if max_stocks is not None:
            spot = spot.head(max_stocks)
        iterator = spot.itertuples(index=False)
        futures = {}
        results: list[LimitUpResult] = []
        workers = max(1, config.workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in iterator:
                futures[pool.submit(_evaluate_single, row.code, row.name, config, end_date)] = row.code
            if show_progress and tqdm is not None:
                for future in tqdm(as_completed(futures), total=len(futures), desc="筛选中"):
                    result = future.result()
                    if result is not None:
                        results.append(result)
            else:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
        results.sort(key=lambda item: (item.limit_up_count, -item.max_consecutive), reverse=True)
        return results


def _results_to_dataframe(results: list[LimitUpResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=list(COLUMN_LABELS.values()))
    data = [
        {
            "code": r.code,
            "name": r.name,
            "limit_up_count": r.limit_up_count,
            "max_consecutive": r.max_consecutive,
            "limit_up_dates": r.limit_up_dates,
            "last_close": r.last_close,
            "last_trade_date": r.last_trade_date,
            "limit_threshold": r.limit_threshold,
            "latest_main_flow": r.latest_main_flow,
        }
        for r in results
    ]
    df = pd.DataFrame(data)
    df.sort_values(["limit_up_count", "max_consecutive", "code"], ascending=[False, True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[list(COLUMN_LABELS.keys())]
    df.rename(columns=COLUMN_LABELS, inplace=True)
    return df


def _export_dataframe(df: pd.DataFrame, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(path, index=False)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("请先安装 openpyxl: pip install openpyxl") from exc
    print(f"已导出 {len(df)} 条记录到 {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选最近 20 个交易日多次涨停的股票")
    parser.add_argument("--lookback", type=int, default=None, help="统计的交易日数量，默认使用配置值")
    parser.add_argument("--min", dest="min_limit", type=int, default=None, help="最少涨停次数，默认使用配置值")
    parser.add_argument("--max", dest="max_limit", type=int, default=None, help="最多涨停次数，默认使用配置值")
    parser.add_argument(
        "--max-consecutive",
        dest="max_consecutive",
        type=int,
        default=None,
        help="允许的最大连续涨停天数，默认使用配置值（拒绝 3 连板）",
    )
    parser.add_argument("--tolerance", type=float, default=None, help="涨停判定阈值冗余，单位百分比，默认使用配置值")
    parser.add_argument(
        "--history-buffer",
        type=int,
        default=None,
        help="为了取得完整交易样本额外回溯的自然日天数，默认使用配置值",
    )
    parser.add_argument("--include-st", action="store_true", help="包含 ST 股票（默认排除）")
    parser.add_argument("--max-stocks", type=int, default=None, help="只测前 N 只股票，调试时使用")
    parser.add_argument("--workers", type=int, default=None, help="并发线程数，默认使用配置值")
    parser.add_argument("--show-progress", action="store_true", help="显示 tqdm 进度条")
    parser.add_argument("--export-xlsx", default=None, help="导出筛选结果 Excel 文件的路径，不传则默认写入 data 目录")
    parser.add_argument("--use-mysql", action="store_true", help="从 MySQL 而非 AkShare 获取行情")
    parser.add_argument("--mysql-host", default=None, help="MySQL 主机地址")
    parser.add_argument("--mysql-port", type=int, default=None, help="MySQL 端口")
    parser.add_argument("--mysql-user", default=None, help="MySQL 用户名")
    parser.add_argument("--mysql-password", default=None, help="MySQL 密码")
    parser.add_argument("--mysql-database", default=None, help="MySQL 库名")
    parser.add_argument("--mysql-table", default=None, help="行情所在的表名")
    parser.add_argument("--mysql-code-column", default=None, help="股票代码列名")
    parser.add_argument("--mysql-date-column", default=None, help="交易日期列名")
    parser.add_argument("--mysql-change-column", default=None, help="涨跌幅列名")
    parser.add_argument("--mysql-flow-column", default=None, help="主力净流入列名")
    parser.add_argument("--mysql-name-column", default=None, help="股票名称列名，如无可留空")
    parser.add_argument("--mysql-close-column", default=None, help="收盘价列名，可选")
    parser.add_argument("--proxy-api-url", default=None, help="快代理提取 API 地址，配置后每次执行都会申请新 IP")
    parser.add_argument(
        "--proxy-api-param",
        action="append",
        type=_kv_pair,
        metavar="KEY=VALUE",
        help="附加到代理提取 API 的查询参数，可多次填写",
    )
    parser.add_argument(
        "--proxy-api-header",
        action="append",
        type=_kv_pair,
        metavar="KEY=VALUE",
        help="附加到代理提取 API 的请求头，可多次填写",
    )
    parser.add_argument("--proxy-api-timeout", type=float, default=None, help="代理提取 API 超时时间，单位秒")
    parser.add_argument("--proxy-scheme", default=None, help="代理协议（http/https），默认 http")
    parser.add_argument("--proxy-user", default=None, help="代理认证用户名")
    parser.add_argument("--proxy-password", default=None, help="代理认证密码")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_default_config()

    overrides = {}
    if args.lookback is not None:
        overrides["lookback_days"] = args.lookback
    if args.min_limit is not None:
        overrides["min_limit_ups"] = args.min_limit
    if args.max_limit is not None:
        overrides["max_limit_ups"] = args.max_limit
    if args.max_consecutive is not None:
        overrides["max_consecutive_limit_ups"] = args.max_consecutive
    if args.tolerance is not None:
        overrides["percent_tolerance"] = args.tolerance
    if args.history_buffer is not None:
        overrides["history_buffer_days"] = args.history_buffer
    if args.include_st:
        overrides["exclude_st"] = False
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.use_mysql:
        overrides["use_mysql"] = True
    if args.mysql_host is not None:
        overrides["mysql_host"] = args.mysql_host
    if args.mysql_port is not None:
        overrides["mysql_port"] = args.mysql_port
    if args.mysql_user is not None:
        overrides["mysql_user"] = args.mysql_user
    if args.mysql_password is not None:
        overrides["mysql_password"] = args.mysql_password
    if args.mysql_database is not None:
        overrides["mysql_database"] = args.mysql_database
    if args.mysql_table is not None:
        overrides["mysql_table"] = args.mysql_table
    if args.mysql_code_column is not None:
        overrides["mysql_code_column"] = args.mysql_code_column
    if args.mysql_date_column is not None:
        overrides["mysql_date_column"] = args.mysql_date_column
    if args.mysql_change_column is not None:
        overrides["mysql_change_column"] = args.mysql_change_column
    if args.mysql_flow_column is not None:
        overrides["mysql_flow_column"] = args.mysql_flow_column
    if args.mysql_name_column is not None:
        overrides["mysql_name_column"] = args.mysql_name_column
    if args.mysql_close_column is not None:
        overrides["mysql_close_column"] = args.mysql_close_column
    if args.proxy_api_url is not None:
        overrides["proxy_api_url"] = args.proxy_api_url
    if args.proxy_api_param:
        overrides["proxy_api_params"] = dict(args.proxy_api_param)
    if args.proxy_api_header:
        overrides["proxy_api_headers"] = dict(args.proxy_api_header)
    if args.proxy_api_timeout is not None:
        overrides["proxy_api_timeout"] = args.proxy_api_timeout
    if args.proxy_scheme is not None:
        overrides["proxy_scheme"] = args.proxy_scheme
    if args.proxy_user is not None:
        overrides["proxy_username"] = args.proxy_user
    if args.proxy_password is not None:
        overrides["proxy_password"] = args.proxy_password

    if overrides:
        config = replace(config, **overrides)

    results = run_screen(
        config=config,
        max_stocks=args.max_stocks,
        show_progress=args.show_progress,
    )
    df = _results_to_dataframe(results)
    if df.empty:
        print("未找到符合条件的股票")
    else:
        print(df.head(20).to_string(index=False))
    if args.export_xlsx:
        _export_dataframe(df, args.export_xlsx)
    elif not df.empty:
        _export_dataframe(df, DEFAULT_EXPORT_FILE)


if __name__ == "__main__":
    main()

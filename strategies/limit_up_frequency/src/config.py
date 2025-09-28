"""限涨频次策略配置项。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


_ENV_LOADED = False


def _load_env_file() -> None:
    """从项目根目录读取 .env，填充缺失的环境变量。"""

    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        _ENV_LOADED = True
        return

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key:
                continue
            if key not in os.environ:
                os.environ[key] = value.strip()
    finally:
        _ENV_LOADED = True


_load_env_file()


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 4
    return max(4, min(16, cpu_count * 2))


def _default_http_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://quote.eastmoney.com/",
    }


@dataclass(slots=True)
class LimitUpConfig:
    """限涨筛选所需的全部条件与连接参数。"""

    lookback_days: int = 30  # 统计的最近交易日数量
    min_limit_ups: int = 3  # 窗口内最少涨停次数（含）
    max_limit_ups: int = 5  # 窗口内最多涨停次数（含）
    max_consecutive_limit_ups: int = 2  # 允许的最大连续涨停天数
    percent_tolerance: float = 0.2  # 涨停判定冗余（百分点）
    history_buffer_days: int = 90  # 额外回溯的自然日天数
    exclude_st: bool = True  # 是否剔除 ST 股票
    workers: int = _default_workers()  # 并发线程数

    fetch_retry_attempts: int = 3  # 行情请求最大重试次数
    fetch_retry_delay: float = 1.5  # 首次重试等待秒数
    fetch_retry_backoff: float = 1.8  # 重试等待倍数
    http_headers: Dict[str, str] = field(default_factory=_default_http_headers)  # 默认请求头

    proxy_api_url: str | None = None  # 代理提取接口地址
    proxy_api_params: Dict[str, str] = field(default_factory=dict)  # 代理提取接口的查询参数
    proxy_api_headers: Dict[str, str] = field(default_factory=dict)  # 代理提取接口的请求头
    proxy_api_timeout: float = 8.0  # 代理提取接口超时时间（秒）
    proxy_scheme: str = "http"  # 代理协议前缀
    proxy_username: str | None = None  # 代理认证用户名
    proxy_password: str | None = None  # 代理认证密码

    use_mysql: bool = False  # 是否改用 MySQL 数据源
    mysql_host: str = "127.0.0.1"  # MySQL 主机地址
    mysql_port: int = 3306  # MySQL 端口
    mysql_user: str = os.getenv("MYSQL_USER", "cwjcw")  # MySQL 用户名
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "bj210726")  # MySQL 密码
    mysql_database: str = "mystock"  # MySQL 库名
    mysql_table: str = "fund_flow_daily"  # 行情所在表名
    mysql_code_column: str = "代码"  # 股票代码列名
    mysql_date_column: str = "日期"  # 交易日期列名
    mysql_change_column: str = "涨跌幅"  # 涨跌幅列名（小数）
    mysql_flow_column: Optional[str] = "主力净流入-净额"  # 主力净流入列名
    mysql_name_column: Optional[str] = "名称"  # 股票名称列名
    mysql_close_column: Optional[str] = None  # 收盘价列名
    mysql_change_multiplier: float = 100.0  # 涨跌幅换算倍数（小数转百分比）


def get_default_config() -> LimitUpConfig:
    """Return a fresh config instance with default values."""

    return LimitUpConfig()


__all__ = ["LimitUpConfig", "get_default_config"]

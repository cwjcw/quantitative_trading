from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def normalize_symbol(symbol: str) -> Tuple[str, str]:
    symbol = symbol.strip()
    if symbol.endswith(".SZ"):
        return symbol[:6], "sz"
    if symbol.endswith(".SH"):
        return symbol[:6], "sh"
    if symbol.endswith(".BJ"):
        return symbol[:6], "bj"
    if symbol.isdigit() and len(symbol) == 6:
        if symbol.startswith(("6", "9")):
            return symbol, "sh"
        if symbol.startswith(("8", "4")):
            return symbol, "bj"
        return symbol, "sz"
    raise ValueError(f"股票代码格式不正确: {symbol}")


def get_symbols_and_market() -> Tuple[List[str], Dict[str, str]]:
    config = load_config()
    symbols = config.get("symbols", [])
    market_map = config.get("market_map", {})
    return symbols, market_map

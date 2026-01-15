from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MAX_BOARDS = None  # 设为整数可限制抓取数量，避免耗时过长

INTRO_KEYS = ["概念解析", "板块简介", "板块介绍", "概念介绍", "核心逻辑"]


def _extract_intro(info_df: pd.DataFrame) -> str:
    if info_df.empty or not {"项目", "值"}.issubset(info_df.columns):
        return ""
    for key in INTRO_KEYS:
        hit = info_df[info_df["项目"] == key]
        if not hit.empty:
            return str(hit.iloc[0]["值"]).strip()
    return ""


def _fetch_concept_boards() -> pd.DataFrame:
    names = ak.stock_board_concept_name_ths()
    names = names.rename(columns={"name": "名称", "code": "代码"})
    if MAX_BOARDS:
        names = names.head(MAX_BOARDS)

    intro_list: List[str] = []
    for name in names["名称"].astype(str).tolist():
        try:
            info_df = ak.stock_board_concept_info_ths(symbol=name)
            intro_list.append(_extract_intro(info_df))
        except Exception:
            intro_list.append("")

    names["介绍"] = intro_list
    return names


def _fetch_industry_boards() -> pd.DataFrame:
    names = ak.stock_board_industry_name_ths()
    names = names.rename(columns={"name": "名称", "code": "代码"})
    if MAX_BOARDS:
        names = names.head(MAX_BOARDS)

    intro_list: List[str] = []
    for name in names["名称"].astype(str).tolist():
        try:
            info_df = ak.stock_board_industry_info_ths(symbol=name)
            intro_list.append(_extract_intro(info_df))
        except Exception:
            intro_list.append("")

    names["介绍"] = intro_list
    return names


def main() -> None:
    concept_df = _fetch_concept_boards()
    industry_df = _fetch_industry_boards()

    concept_json = DATA_DIR / "concept_boards.json"
    industry_json = DATA_DIR / "industry_boards.json"

    concept_df.to_json(concept_json, orient="records", force_ascii=False, indent=2)
    industry_df.to_json(industry_json, orient="records", force_ascii=False, indent=2)

    print(f"概念板块: {len(concept_df)} 条 -> {concept_json}")
    print(f"行业板块: {len(industry_df)} 条 -> {industry_json}")


if __name__ == "__main__":
    main()

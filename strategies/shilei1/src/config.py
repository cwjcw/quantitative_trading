"""参数配置：集中管理首波回撤策略的可调项。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScreenConfig:
    """首波回撤筛选策略参数。"""

    history_years: int = 3  # 回溯历史K线的年份数，用于下载足够的日线数据
    volatility_window_short: int = 20  # 短周期波动率窗口，衡量近期波动
    volatility_window_long: int = 60  # 长周期波动率窗口，作为参考基准
    volatility_ratio_threshold: float = 0.7  # 波动率收敛阈值，越小要求收敛越明显
    pullback_threshold: float = 0.3  # 回撤比例阈值，默认要求回撤至少30%
    first_wave_gain: float = 1.0  # 首波涨幅要求，1.0 即涨幅≥100%
    first_wave_lookback_days: int = 200  # 寻找首波行情的回看天数
    market_cap_floor: float = 0  # 市值下限，可按需调整
    price_floor_ratio: float = 0.35  # 当前价格相对首波起点的最低比例
    trend_days_required: int = 144  # 均线上方所需天数（360天的40%）
    trend_window: int = 360  # 统计均线占比的窗口长度


def get_default_config() -> ScreenConfig:
    """返回一份可编辑的默认配置。"""

    return ScreenConfig()

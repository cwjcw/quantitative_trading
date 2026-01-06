"""
数据获取模块 - 使用AKSHARE获取A股数据
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_all_stocks() -> List[str]:
    """
    获取所有A股上市公司代码
    """
    try:
        logger.info("正在获取所有A股上市公司代码...")
        # 使用 stock_zh_a_spot_em 获取A股列表
        stock_list = ak.stock_zh_a_spot_em()
        codes = stock_list['代码'].tolist()
        logger.info(f"获取到{len(codes)}只A股")
        return codes
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return []


def get_daily_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定股票的日线数据
    
    Args:
        code: 股票代码（如'600000'）
        start_date: 开始日期（格式：'20250101'）
        end_date: 结束日期（格式：'20260106'）
    
    Returns:
        包含日线数据的DataFrame
    """
    try:
        # 使用 stock_zh_a_hist 获取日线数据
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        
        if df is not None and not df.empty:
            # 重命名列以保持一致性
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_percent',
                '涨跌额': 'change_amount',
                '换手率': 'turnover'
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.debug(f"获取 {code} 数据失败: {e}")
        return pd.DataFrame()


def detect_consecutive_limit_ups(df: pd.DataFrame, consecutive_days: int = 2, threshold: float = 0.095) -> List[Tuple[str, str]]:
    """
    检测连续涨停的日期
    
    Args:
        df: 包含日线数据的DataFrame
        consecutive_days: 连续涨停天数
        threshold: 涨停阈值（默认9.5%）
    
    Returns:
        [(日期, 涨幅), ...] 表示连续涨停周期的最后一个日期
    """
    if df.empty:
        return []
    
    limit_up_dates = []
    
    # 识别涨停日 - 处理字符串格式的涨幅
    df_copy = df.copy()
    
    # 处理涨幅列 - 可能是字符串格式
    if df_copy['change_percent'].dtype == 'object':
        df_copy['change_percent'] = df_copy['change_percent'].str.rstrip('%').astype(float) / 100
    else:
        df_copy['change_percent'] = df_copy['change_percent'] / 100
    
    df_copy['is_limit_up'] = df_copy['change_percent'] >= threshold
    
    # 找出连续涨停的最后一天
    for i in range(consecutive_days - 1, len(df_copy)):
        if all(df_copy.loc[i-consecutive_days+1:i, 'is_limit_up'].values):
            # 返回连续涨停周期的最后一个日期
            date = df_copy.loc[i, 'date'].strftime('%Y-%m-%d')
            last_pct = df_copy.loc[i, 'change_percent']
            limit_up_dates.append((date, last_pct))
    
    return limit_up_dates


def get_next_trading_day_price(code: str, reference_date: str, start_date: str, end_date: str) -> float:
    """
    获取指定日期之后的下一个交易日的收盘价
    
    Args:
        code: 股票代码
        reference_date: 参考日期（格式：'2025-01-01'）
        start_date: 数据起始日期
        end_date: 数据结束日期
    
    Returns:
        下一个交易日的收盘价，如果没有找到返回0
    """
    try:
        df = get_daily_data(code, start_date, end_date)
        if df.empty:
            return 0
        
        ref_datetime = pd.to_datetime(reference_date)
        next_day_data = df[df['date'] > ref_datetime]
        
        if next_day_data.empty:
            return 0
        
        return float(next_day_data.iloc[0]['close'])
    except Exception as e:
        logger.debug(f"获取下一交易日价格失败 {code}: {e}")
        return 0

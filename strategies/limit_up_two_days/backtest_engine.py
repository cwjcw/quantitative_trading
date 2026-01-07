"""
策略回测引擎
"""
import pandas as pd
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from data_fetcher import get_daily_data, detect_consecutive_limit_ups, get_next_trading_day_price
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Portfolio:
    """投资组合管理类"""
    
    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.holdings = {}  # {date: [(code, price, amount, quantity), ...]}
        self.trades = []  # 交易记录
        self.daily_values = {}  # 日净值
        
    def buy(self, code: str, price: float, amount: float, date: str):
        """
        买入股票
        """
        if self.cash < amount:
            logger.warning(f"{date} 现金不足，无法买入 {code}")
            return False
        
        quantity = amount / price
        self.cash -= amount
        
        # 记录持仓
        if date not in self.holdings:
            self.holdings[date] = []
        
        self.holdings[date].append({
            'code': code,
            'price': price,
            'amount': amount,
            'quantity': quantity,
            'buy_date': date
        })
        
        self.trades.append({
            'date': date,
            'code': code,
            'action': 'BUY',
            'price': price,
            'amount': amount,
            'quantity': quantity
        })
        
        logger.info(f"{date} 买入 {code}: 数量={quantity:.2f}, 金额={amount:.2f}, 价格={price:.2f}")
        return True
    
    def sell(self, code: str, price: float, quantity: float, date: str, buy_date: str = None):
        """
        卖出股票
        """
        amount = price * quantity
        self.cash += amount
        
        self.trades.append({
            'date': date,
            'code': code,
            'action': 'SELL',
            'price': price,
            'amount': amount,
            'quantity': quantity,
            'buy_date': buy_date
        })
        
        profit = amount - (price * quantity)  # 这里需要用买入价格计算
        logger.info(f"{date} 卖出 {code}: 数量={quantity:.2f}, 金额={amount:.2f}, 价格={price:.2f}")
        return True
    
    def get_total_value(self, market_prices: Dict[str, float], date: str) -> float:
        """
        计算总资产价值
        """
        total = self.cash
        
        # 计算所有持仓的市值
        for buy_date, positions in self.holdings.items():
            for pos in positions:
                code = pos['code']
                if code in market_prices:
                    total += market_prices[code] * pos['quantity']
        
        self.daily_values[date] = total
        return total
    
    def get_summary(self) -> Dict:
        """
        获取回测总结
        """
        total_trades = len(self.trades)
        buy_trades = len([t for t in self.trades if t['action'] == 'BUY'])
        sell_trades = len([t for t in self.trades if t['action'] == 'SELL'])
        
        final_value = list(self.daily_values.values())[-1] if self.daily_values else self.initial_capital
        profit = final_value - self.initial_capital
        profit_pct = (profit / self.initial_capital) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'profit': profit,
            'profit_pct': profit_pct,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'final_cash': self.cash
        }


class BacktestEngine:
    """策略回测引擎"""
    
    def __init__(self, capital: float, start_date: str, end_date: str, final_date: str):
        self.portfolio = Portfolio(capital)
        self.start_date = start_date
        self.end_date = end_date
        self.final_date = final_date
        self.daily_prices = {}  # {date: {code: price}}
        self.buy_signals = {}  # {date: [codes]}
        
    def generate_signals(self, stock_codes: List[str]):
        """
        生成买入信号 - 找出所有连续2个涨停的股票
        """
        logger.info("正在生成买入信号...")
        
        all_buy_signals = {}  # {date: [codes]}
        
        for i, code in enumerate(stock_codes):
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(stock_codes)} 只股票")
            
            # 获取该股票的日线数据
            df = get_daily_data(code, self.start_date, self.end_date)
            if df.empty:
                continue
            
            # 检测连续涨停
            limit_up_dates = detect_consecutive_limit_ups(
                df, 
                consecutive_days=config.CONSECUTIVE_LIMIT_UP_DAYS,
                threshold=config.LIMIT_UP_THRESHOLD
            )
            
            # 记录买入信号（连续涨停最后一天的下一个交易日买入）
            for limit_up_date, pct in limit_up_dates:
                # 获取连续涨停那天的收盘价作为下一天的参考价
                prev_close_vals = df[df['date'].dt.strftime('%Y-%m-%d') == limit_up_date]['close'].values
                if len(prev_close_vals) == 0:
                    continue
                prev_close = prev_close_vals[0]

                # 下一个交易日
                next_trading_date = self._get_next_trading_date(df, limit_up_date)
                if not next_trading_date:
                    continue

                # 检查如果第三天（即下一个交易日）开盘价为涨停（相对于前一日收盘），则跳过买入
                next_row = df[df['date'].dt.strftime('%Y-%m-%d') == next_trading_date]
                skip_buy = False
                if not next_row.empty:
                    open_price = next_row.iloc[0]['open']
                    # 计算开盘相对前一日收盘的涨幅
                    try:
                        open_pct = (open_price - prev_close) / prev_close
                        if open_pct >= config.LIMIT_UP_THRESHOLD:
                            skip_buy = True
                    except Exception:
                        skip_buy = False

                if skip_buy:
                    # 跳过该买入信号
                    continue

                # 使用第三天收盘价作为买入价格
                close_price = next_row.iloc[0]['close']
                entry_price = close_price
                if next_trading_date not in all_buy_signals:
                    all_buy_signals[next_trading_date] = []
                all_buy_signals[next_trading_date].append({
                    'code': code,
                    'entry_price': entry_price,
                    'entry_date': limit_up_date
                })
        
        self.buy_signals = all_buy_signals
        logger.info(f"生成了 {sum(len(v) for v in all_buy_signals.values())} 个买入信号")
        return all_buy_signals
    
    def _get_next_trading_date(self, df: pd.DataFrame, date_str: str) -> str:
        """
        获取指定日期之后的下一个交易日
        """
        ref_date = pd.to_datetime(date_str)
        next_dates = df[df['date'] > ref_date]
        if not next_dates.empty:
            return next_dates.iloc[0]['date'].strftime('%Y-%m-%d')
        return None
    
    def run(self, stock_codes: List[str]):
        """
        运行回测
        """
        logger.info("开始回测...")
        
        # 第一步：生成买入信号
        self.generate_signals(stock_codes)
        
        # 第二步：获取所有股票的完整数据
        all_stock_data = {}
        for code in stock_codes:
            df = get_daily_data(code, self.start_date, self.final_date)
            if not df.empty:
                all_stock_data[code] = df
        
        # 获取所有交易日期（从开始日期到最终清仓日期）
        all_dates = set()
        for df in all_stock_data.values():
            all_dates.update(df['date'].dt.strftime('%Y-%m-%d').tolist())
        
        all_dates = sorted(list(all_dates))
        
        # 按日期进行回测
        for current_date in all_dates:
            logger.info(f"处理日期: {current_date}")
            
            # 处理买入信号
            if current_date in self.buy_signals:
                buy_signals_today = self.buy_signals[current_date]
                if buy_signals_today:
                    # 计算每个股票的买入金额（平均分配）
                    available_cash = self.portfolio.cash
                    amount_per_stock = available_cash / len(buy_signals_today)
                    
                    logger.info(f"{current_date} 有 {len(buy_signals_today)} 个买入信号，每个买入 {amount_per_stock:.2f}")
                    
                    for signal in buy_signals_today:
                        self.portfolio.buy(
                            code=signal['code'],
                            price=signal['entry_price'],
                            amount=amount_per_stock,
                            date=current_date
                        )
            
            # 处理卖出（T+1）
            self._process_sells(current_date, all_stock_data)
            
            # 更新日净值
            market_prices = self._get_market_prices(current_date, all_stock_data)
            total_value = self.portfolio.get_total_value(market_prices, current_date)
            logger.info(f"{current_date} 总资产: {total_value:.2f}")
            
            # 最后一个交易日清仓
            if current_date == self.final_date:
                self._liquidate_all(current_date, market_prices)
        
        logger.info("回测完成")
        return self.portfolio.get_summary()
    
    def _get_market_prices(self, date_str: str, all_stock_data: Dict) -> Dict[str, float]:
        """
        获取指定日期的所有股票收盘价
        """
        prices = {}
        date = pd.to_datetime(date_str)
        
        for code, df in all_stock_data.items():
            price_data = df[df['date'] == date]
            if not price_data.empty:
                prices[code] = price_data.iloc[0]['close']
        
        return prices
    
    def _process_sells(self, current_date: str, all_stock_data: Dict):
        """
        处理T+2卖出（第四天开盘价卖出）
        """
        # 遍历所有持仓，找出应该在今天卖出的（买入后的第三个交易日）
        for buy_date, positions in list(self.portfolio.holdings.items()):
            for pos in positions:
                code = pos['code']
                # 计算从买入日期到今天经过了多少个交易日
                buy_date_dt = pd.to_datetime(buy_date)
                current_date_dt = pd.to_datetime(current_date)
                
                if code in all_stock_data:
                    df = all_stock_data[code]
                    # 获取买入日期和当前日期之间的交易日数
                    trading_dates = df[(df['date'] > buy_date_dt) & (df['date'] <= current_date_dt)]['date'].tolist()
                    
                    # 如果已经经过了2个交易日，就卖出（第四天开盘价）
                    if len(trading_dates) >= 2:
                        # 获取第二个交易日（买入后的第三个交易日）的开盘价作为卖出价
                        sell_price = df[df['date'] == trading_dates[1]]['open'].values[0]
                        self.portfolio.sell(
                            code=code,
                            price=sell_price,
                            quantity=pos['quantity'],
                            date=trading_dates[1].strftime('%Y-%m-%d'),
                            buy_date=buy_date
                        )
                        # 从持仓中移除
                        positions.remove(pos)
    
    def _liquidate_all(self, current_date: str, market_prices: Dict[str, float]):
        """
        清仓所有持仓
        """
        logger.info(f"{current_date} 清仓所有持仓")
        
        for buy_date, positions in list(self.portfolio.holdings.items()):
            for pos in positions:
                code = pos['code']
                if code in market_prices:
                    sell_price = market_prices[code]
                    self.portfolio.sell(
                        code=code,
                        price=sell_price,
                        quantity=pos['quantity'],
                        date=current_date,
                        buy_date=buy_date
                    )
            self.portfolio.holdings[buy_date] = []

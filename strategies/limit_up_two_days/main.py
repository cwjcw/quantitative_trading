"""
主程序 - 运行策略回测
"""
import pandas as pd
import logging
from data_fetcher import get_all_stocks
from backtest_engine import BacktestEngine
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    主函数 - 执行回测
    """
    logger.info("=" * 80)
    logger.info("量化交易策略回测系统")
    logger.info("=" * 80)
    logger.info(f"策略: 连续2个涨停 T+1 卖出")
    logger.info(f"初始资金: {config.INITIAL_CAPITAL:,.0f}元")
    logger.info(f"回测周期: {config.START_DATE} 至 {config.END_DATE}")
    logger.info(f"清仓日期: {config.FINAL_LIQUIDATION_DATE}")
    logger.info("=" * 80)
    
    try:
        # 获取所有A股股票代码
        stock_codes = get_all_stocks()
        if not stock_codes:
            logger.error("无法获取股票列表，回测终止")
            return
        
        logger.info(f"获取到 {len(stock_codes)} 只A股")
        
        # 为了加快运行速度，这里只取前500只股票作为演示
        # 实际生产环境可以移除或扩大这个限制
        sample_size = min(50000, len(stock_codes))
        stock_codes_sample = stock_codes[:sample_size]
        logger.info(f"本次回测使用前 {sample_size} 只股票 (可在main.py中修改sample_size)")
        
        # 创建回测引擎并运行
        engine = BacktestEngine(
            capital=config.INITIAL_CAPITAL,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            final_date=config.FINAL_LIQUIDATION_DATE
        )
        
        summary = engine.run(stock_codes_sample)
        
        # 输出回测结果
        logger.info("=" * 80)
        logger.info("回测结果总结")
        logger.info("=" * 80)
        logger.info(f"初始资金:        {summary['initial_capital']:>15,.2f}元")
        logger.info(f"最终资产:        {summary['final_value']:>15,.2f}元")
        logger.info(f"盈利金额:        {summary['profit']:>15,.2f}元")
        logger.info(f"盈利率:          {summary['profit_pct']:>15.2f}%")
        logger.info(f"总交易次数:      {summary['total_trades']:>15}次")
        logger.info(f"买入次数:        {summary['buy_trades']:>15}次")
        logger.info(f"卖出次数:        {summary['sell_trades']:>15}次")
        logger.info(f"最终现金:        {summary['final_cash']:>15,.2f}元")
        logger.info("=" * 80)
        
        # 保存详细交易记录
        trades_df = pd.DataFrame(engine.portfolio.trades)
        trades_df.to_csv('trading_records.csv', index=False, encoding='utf-8-sig')
        logger.info("交易记录已保存到 trading_records.csv")
        
        # 保存日净值
        daily_values_df = pd.DataFrame(
            list(engine.portfolio.daily_values.items()),
            columns=['date', 'total_value']
        )
        daily_values_df.to_csv('daily_values.csv', index=False, encoding='utf-8-sig')
        logger.info("日净值已保存到 daily_values.csv")
        
    except Exception as e:
        logger.error(f"回测过程中出错: {e}", exc_info=True)


if __name__ == '__main__':
    main()

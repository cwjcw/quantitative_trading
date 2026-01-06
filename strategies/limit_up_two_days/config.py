# 策略配置文件
import datetime

# 回测时间范围
START_DATE = "20250101"
END_DATE = "20260106"
FINAL_LIQUIDATION_DATE = "20260107"

# 资金配置
INITIAL_CAPITAL = 200000  # 初始资金20万

# 策略参数
CONSECUTIVE_LIMIT_UP_DAYS = 2  # 连续涨停天数
HOLD_PERIOD = 2  # 持有周期（T+2，即第四天开盘价卖出）

# 涨停判断阈值（考虑到浮动）
LIMIT_UP_THRESHOLD = 0.095  # 9.5%的涨幅视为涨停

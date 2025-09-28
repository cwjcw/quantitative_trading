# 限涨筛选器

该策略用于在最近 20 个交易日内寻找出现多次涨停但未出现 3 连板的个股，结果默认导出到 `strategies/limit_up_frequency/data/limit_up_frequency.csv`。

## 筛选逻辑

- 使用 AkShare 抓取沪深 A 股与北交所的日线行情数据。
- 统计配置窗口（默认 20 个交易日）内满足涨跌幅大于等于涨停阈值的交易日：
  - 默认阈值：主板 10%，创业板/科创板 20%，北交所 30%，ST 股票 5%。
  - 判定时预留 0.2 个百分点冗余，用于处理四舍五入造成的 9.99% 等情况。
- 仅保留涨停天数在 `[3, 4]` 且未出现连续 3 天涨停的股票。

## 使用方式

```bash
# 查看参数
python -m strategies.limit_up_frequency.src.screener --help

# 执行筛选并显示结果，默认也会写入 CSV
python -m strategies.limit_up_frequency.src.screener --show-progress --workers 12

# 指定自定义导出路径
python -m strategies.limit_up_frequency.src.screener --export-csv /tmp/limitup.csv
```

> 提示：如需在无网络代理环境中运行，可直接使用默认参数；脚本会自动暂时关闭系统代理设置以避免连接被劫持。

## 参数配置

- 默认参数集中在 `strategies/limit_up_frequency/src/config.py`，可直接修改该文件中的 `LimitUpConfig` 默认值。
- 配置中包含并发线程数 `workers`，脚本会基于此自动启用多线程抓取；也可以通过命令行的 `--workers` 临时覆盖。
- 若网络偶发断开，可调整 `fetch_retry_attempts`、`fetch_retry_delay` 与 `fetch_retry_backoff` 以控制重试次数与等待时间。
- `http_headers` 提供默认的浏览器 UA、Referer 等字段，若目标接口策略变动，可在此处新增或修改请求头。
- 如需通过快代理等服务动态申请 IP，可配置：
  ```python
  proxy_api_url = "https://dps.kdlapi.com/api/getdps"
  proxy_api_params = {
      "secret_id": "你的ID",
      "signature": "签名",
      "num": "1",
      "pt": "1",
      "format": "json"
  }
  proxy_scheme = "http"
  proxy_username = "代理帐号"
  proxy_password = "代理密码"
  ```
  程序在每次运行时会先调用提取 API 获取全新的 IP，并自动写入 `HTTP(S)_PROXY` 环境变量；也可通过命令行使用 `--proxy-api-url`、`--proxy-api-param secret_id=...` 等参数临时覆盖。
- 要从自建 MySQL 数据库读取行情，可在 `config.py` 中设置：
  ```python
  use_mysql = True
  mysql_host = "127.0.0.1"
  mysql_port = 3306
  mysql_user = "cwjcw"
  mysql_password = "bj210726"
  mysql_database = "mystock"
  mysql_table = "fund_flow_daily"
  mysql_code_column = "代码"
  mysql_date_column = "日期"
  mysql_change_column = "涨跌幅"
  mysql_flow_column = "主力净流入-净额"
  mysql_change_multiplier = 100.0  # 若涨跌幅以小数存储
  ```
  建议把账号密码放在项目根目录的 `.env` 文件中，例如：
  ```INI
  MYSQL_USER=cwjcw
  MYSQL_PASSWORD=bj210726
  ```
  配置文件会自动读取 `.env`，也可通过系统环境变量覆盖。
  命令行也可临时覆盖，例如：
  ```bash
  python -m strategies.limit_up_frequency.src.screener \
    --use-mysql --mysql-host 127.0.0.1 --mysql-user cwjcw --mysql-password bj210726 \
    --mysql-database mystock --mysql-table fund_flow_daily --mysql-change-column 涨跌幅 \
    --mysql-change-multiplier 100 --show-progress
  ```

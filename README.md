# 量化交易研究仓库

该仓库用于沉淀我在A股市场的量化交易与选股策略。每个策略都放在 `strategies/` 下的独立子目录，并配套自己的 README、数据、笔记与代码，方便快速定位和复用。

## 目录结构

- `strategies/` 各策略的独立工作区（含 `README.md`、`src/`、`data/`、`notebooks/`）
- `scripts/` 通用脚本与批量任务工具
- `docs/` 共享研究文档或背景分析
- `requirements.txt` 全局依赖清单

## 使用方式

1. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 进入对应策略目录（如 `strategies/shilei1/`）阅读 README，按照说明运行或调整参数。
3. 若需要对策略做深入研究，可在该目录的 `notebooks/` 中创建 Jupyter Notebook，或在 `data/` 存储本地缓存数据。

## 策略索引

- [`strategies/shilei1/`](strategies/shilei1/): “首波翻倍+深度回撤” 选股逻辑。

## 新增策略指引

1. 在 `strategies/` 下新建子文件夹（建议使用蛇形命名）。
2. 拷贝基础结构或参考现有策略，务必补充自己的 `README.md` 来说明策略逻辑、参数与使用步骤。
3. 共用的工具/脚本请放入 `scripts/` 或未来的共享包中，避免策略之间直接耦合。

祝研究顺利！

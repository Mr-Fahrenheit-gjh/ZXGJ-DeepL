# 688981 A 股 T+0 机器学习研究项目

本项目用于研究中芯国际 `688981` 的 5 分钟级别机器学习信号、A 股底仓 T+0 回测、模型解释性分析和可选 vn.py / VeighNa 事件驱动回测。

详细说明见：[系统实现与运行指南.md](系统实现与运行指南.md)。

## 1. 环境安装

推荐 Python 3.10-3.12。

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

可选 vn.py / VeighNa 回测依赖：

```bash
python -m pip install -r requirements-vnpy.txt
```

可选数据源依赖：

```bash
python -m pip install -r requirements-data.txt
```

## 2. 快速验证

```bash
python run_research_pipeline.py \
  --quick \
  --run-walk-forward \
  --model-names logistic_regression random_forest \
  --max-folds 1 \
  --output-dir outputs/diagnostics/smoke_basic
```

切换到 TA 库特征工程版本：

```bash
python run_research_pipeline.py \
  --quick \
  --run-walk-forward \
  --feature-set ta \
  --model-names logistic_regression random_forest \
  --max-folds 1 \
  --output-dir outputs/diagnostics/smoke_ta_features
```

可选特征集：

```text
basic     项目内置手工特征
ta        TA 库指标特征，已过滤绝对价格/成交量水平
basic_ta  两套特征合并，用于对照实验
```

## 3. 题目完整模型族验证

```bash
python run_research_pipeline.py \
  --quick \
  --run-walk-forward \
  --full-model-suite \
  --run-explainability \
  --run-vnpy-backtest \
  --max-folds 1 \
  --output-dir outputs/diagnostics/smoke_full
```

## 4. 自动超参数优化

优先使用 walk-forward 级别的自动调参，因为它直接优化成本后的交易结果，而不是只优化 AUC。

快速方向验证：

```bash
python run_research_pipeline.py \
  --quick \
  --run-walk-forward \
  --run-wf-optuna \
  --wf-optuna-trials 5 \
  --model-names logistic_regression random_forest \
  --max-folds 3 \
  --output-dir outputs/diagnostics/wf_optuna_check
```

完整模型族调参会很慢，建议只在快速验证方向成立后再跑：

```bash
python run_research_pipeline.py \
  --run-walk-forward \
  --run-wf-optuna \
  --wf-optuna-trials 20 \
  --full-model-suite \
  --max-folds 3 \
  --output-dir outputs/diagnostics/wf_optuna_full
```

关键输出文件：

```text
outputs/diagnostics/<run_name>/walk_forward_optuna/walk_forward_optuna_trials.csv
outputs/diagnostics/<run_name>/walk_forward_optuna/walk_forward_optuna_best.json
outputs/diagnostics/<run_name>/applied_walk_forward_optuna_config.json
outputs/diagnostics/<run_name>/final_run_summary.json
```

原来的 `--run-optuna` 仍可用于单个深度模型的训练超参搜索；`--run-wf-optuna` 则用于整体交易系统参数搜索。

## 5. 路径说明

代码通过 `project_paths.py` 解析项目根目录。默认数据和输出路径都是相对项目根目录的，因此把整个文件夹复制到另一台电脑后，不需要改绝对路径。

默认输入数据：

```text
688981_5min_20200716-20260602.parquet
```

默认输出目录：

```text
outputs/diagnostics/research_pipeline
```

## 6. 长任务进度查看

长时间运行时，不需要盯终端。主流程会持续更新：

```text
outputs/diagnostics/<run_name>/pipeline_status.json
outputs/diagnostics/<run_name>/walk_forward/walk_forward_progress.json
outputs/diagnostics/<run_name>/walk_forward/walk_forward_progress.csv
```

Windows PowerShell 查看当前阶段：

```powershell
Get-Content outputs\diagnostics\<run_name>\pipeline_status.json
Get-Content outputs\diagnostics\<run_name>\walk_forward\walk_forward_progress.json
```

跑完后最终总览会保存为：

```text
outputs/diagnostics/<run_name>/final_run_summary.json
```

## 7. 重要提醒

`live_readiness_report.json` 是实盘前硬门槛文件。当前项目可以跑通研究闭环，但不代表已经可以真实投入资金。只有完整 walk-forward、压力测试、影子盘和 vn.py 交叉验证都稳定通过后，才可以进入小资金仿真阶段。

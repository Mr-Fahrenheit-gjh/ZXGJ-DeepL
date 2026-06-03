# MVP 模块化说明

当前系统目标是先完成可审计的机器学习信号和研究回测闭环：`buy_prob` / `sell_prob` -> 多模型集成 -> A 股底仓 T+0 状态机 -> walk-forward 外样本验证。  
注意：当前还没有证明可以真实资金上线，报告会明确标记质量门槛是否通过。

## 模块职责

```text
1. project_paths.py
   项目根目录、默认数据路径、默认输出路径和相对路径解析，方便跨电脑迁移。

2. mvp_config.py
   全局参数：成本、tp/sl、lookback/horizon、模型超参、风控参数、输出路径。

3. feature_engineering.py
   基础特征、特征筛选、时间切分、缩尾、标准化、序列样本构造。

4. label_builder.py
   路径依赖机会标签：
   - buy_label: 未来路径是否出现可交易的低吸/买入机会
   - sell_label: 未来路径是否出现可交易的高抛/卖出机会

5. diagnostics.py
   标签诊断、特征尺度检查、信号分层诊断。

6. model_signals.py
   信号模型层：Logistic/RF/Transformer-LSTM/LSTM/CNN/MLP，统一输出 buy_prob / sell_prob。

7. ensemble.py
   多模型集成层：按验证集 AUC 超过 0.5 的边际信息加权。

8. explainability.py
   排列重要性、输入梯度重要性、可选 SHAP 汇总。

9. risk_management.py
   A 股底仓 T+0 状态机、动态仓位、成交量参与率约束、Sharpe/Sortino/Calmar/最大回撤。

10. validation.py
   walk-forward 时间序列切分和切分留痕。

11. walk_forward_runner.py
   subtrain/calibration/test 编排、训练、集成、回测、质量门槛和 methodology 留痕。

12. hyperparameter_optimization.py
   Optuna 可选搜索入口，导出 trials 表和 best params。

13. vnpy_backtest.py
   vn.py 数据写入、信号导出、CTA 策略和 vn.py 回测接口。

14. run_research_pipeline.py
   命令行复现实验入口：parquet -> 特征 -> 标签 -> walk-forward -> Markdown 报告。

15. research_report.py
   将 walk-forward JSON/CSV 结果导出为 Markdown 研究报告。

16. production_readiness.py
   实盘前审计：数据/代码/config 指纹、特征泄漏审计、live readiness 硬门槛。

17. live_monitoring.py
   影子盘/纸面交易监控：特征 PSI、buy/sell 概率 PSI、影子盘收益、回撤、连续亏损、交易次数和胜率。

18. execution_audit.py
   A 股数据清洗和执行可行性审计：OHLCV 完整性、重复时间戳、零成交量、交易时段、短交易日和异常涨跌幅。

19. verify_production_readiness.py
   一键检查 pipeline 输出目录是否包含完整证据链，并导出 production_readiness_checks.csv。

20. MVP.ipynb
   编排入口和结果展示，不再堆大段模型实现。
```

## 标签口径

当前不是三分类 action label，而是两个独立二分类机会标签：

```text
buy_label = 1
表示从下一根 K 线 open 入场后，未来路径存在低吸/买入机会。

sell_label = 1
表示从下一根 K 线 open 入场后，未来路径存在高抛/卖出机会。
```

核心参数：

```text
horizon = 3
lookback = 32
tp = 0.003
sl = 0.002
commission_rate = 0.001
slippage_rate = 0.0005
one_side_cost = 0.0015
round_trip_cost = 0.003
```

## Walk-forward 防泄漏口径

```text
1. 按时间顺序滚动切分，不随机打乱。
2. 每个训练窗口拆成 subtrain / calibration。
3. 缩尾和标准化只 fit subtrain。
4. calibration 负责 ensemble 权重和交易阈值。
5. fold test 只做外样本评估和回测。
6. 每个 fold 输出 fold_manifest.json。
7. 总目录输出 methodology.json、walk_forward_summary.json、walk_forward_report.md。
```

## A 股底仓 T+0 口径

```text
账户先持有 base_position_pct 对应底仓。
sell_first: 先卖出昨日可用底仓，再用 buy_prob 信号买回。
buy_first: 先买入加仓，再卖出同等数量旧库存恢复底仓。
不把当日新买股票当作可卖库存。
单笔成交受 max_participation_rate 成交量参与率约束。
报告同时输出总权益和相对静态底仓的 alpha_equity。
质量门槛优先检查 alpha_total_return / alpha_max_drawdown。
```

## 输出目录

```text
outputs/diagnostics/model_signals/logistic_regression/
outputs/diagnostics/model_signals/randomforestclassifier/
outputs/diagnostics/model_signals/transformer_lstm/
outputs/diagnostics/model_signals/lstm/
outputs/diagnostics/model_signals/cnn/
outputs/diagnostics/model_signals/mlp/
outputs/diagnostics/model_signals/auc_weighted_ensemble/
outputs/diagnostics/research_checks/explainability/
outputs/diagnostics/walk_forward/
outputs/diagnostics/optuna/
```

## 命令行复现实验

```bash
python run_research_pipeline.py --quick --output-dir outputs/diagnostics/research_pipeline_smoke_prepare
python run_research_pipeline.py --quick --run-walk-forward --output-dir outputs/diagnostics/research_pipeline_smoke_inventory_t0_v2 --model-names logistic_regression random_forest --max-folds 2
python verify_production_readiness.py outputs/diagnostics/research_pipeline_smoke_inventory_t0_v2
```

## 实盘硬门槛

```text
live_readiness_report.json 是最终硬否决文件。
只有 status = PASS，才允许进入人工复核和小资金仿真阶段。
FAIL 不是程序错误，而是“禁止实盘”的明确结论。

当前 live gates 包括：
1. 源代码工作区可复现，忽略 outputs/ 和 __pycache__ 等非源码产物。
2. 特征泄漏审计通过。
3. A 股数据和执行可行性审计通过。
4. walk-forward 研究质量门槛通过。
5. fold 数和交易数达到最低要求。
6. buy/sell 外样本 AUC 超过 live floor。
7. 中位 alpha 收益为正。
8. 中位回撤在限制内。
9. 压力测试下 alpha 仍为正且回撤可控。
10. 影子盘监控通过；如果 `require_shadow_monitoring = True` 但没有 `shadow_monitoring_report.json`，则禁止实盘。
```

## 当前真实数据 quick smoke

```text
Output = outputs/diagnostics/research_pipeline_smoke_inventory_t0_v2
Prepared rows = 67705
Feature count = 194
Executed folds = 2
Total trades = 21
Median test buy AUC ≈ 0.5255
Median test sell AUC ≈ 0.6092
Median total return per fold ≈ -2.0820%
Median alpha total return per fold ≈ -0.4303%
Quality gate = failed
Failed check = alpha_return_above_gate
```

解释：流水线可运行，分类信号有一点外样本排序边际，但在更真实的 A 股底仓 T+0 口径下，扣除静态底仓基准后的 T+0 alpha 仍为负。因此当前模型不能被视为可投入真实资金。

## 上线前缺口

```text
1. 扩大 walk-forward fold 覆盖，确认 AUC 和 alpha 是否稳定。
2. Optuna 搜索必须有 trials 留痕和过拟合审计。
3. 加入更严格的盘口、涨跌停、停牌、成交量和冲击成本约束。
4. 做多阈值、多仓位、多滑点、多成交量参与率压力测试。
5. 与 vn.py / VeighNa 事件撮合回测交叉验证。
6. 至少完成 20 个交易日以上影子盘，输出 `shadow_monitoring_report.json`。
7. 只有长期外样本 alpha 稳定为正、影子盘通过、回撤可控，才可以讨论真实资金级别。
```

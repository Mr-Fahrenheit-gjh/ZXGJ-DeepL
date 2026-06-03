# MVP 改动方向

本文档用于整理当前 `MVP.ipynb` 的下一步改造方向。当前阶段先不直接改 notebook，先把标签、配置、模型参数和回测验证的改动口径统一下来，避免继续在 notebook 里零散试验。

## 1. 当前问题

当前 MVP 已经完成了数据读取、特征工程、LSTM、RandomForest baseline、信号分层诊断和初步回测，但代码结构和研究口径还比较散。

主要问题如下：

- 标签定义过多，包括 `label`、`label_cost_adjusted`、`label_3class`、`label_dynamic_binary`、`label_dynamic_3class`、`label_trade_success` 等，部分标签是探索中临时定义的，主线不清晰。
- 跨模块变量分散，比如 `HORIZON`、`LOOKBACK`、`COMMISSION_RATE`、`SLIPPAGE_RATE`、`TOTAL_COST`、`target_col`、`SEQUENCE_MODE` 在不同 cell 中重复定义。
- LSTM、RandomForest、训练循环和回测参数散落在各自 cell 中，不方便统一调参。
- 当前 AUC 有一定表现，但交易信号不强，因此后续重点不应只是提高 AUC，而是验证高分信号是否能在扣除成本后转化为稳定交易机会。
- 当前回测仍偏诊断性质，部分地方直接使用 `future_return` 作为收益，后续需要过渡到逐 bar 持仓收益和更接近 T+0 的执行逻辑。

## 2. 总体改造目标

下一步不追求一次性做成完整系统，而是把 MVP 收敛成一条可复现主线：

```text
全局配置 -> 特征工程 -> 标签构造 -> 特征选择 -> 时间切分 -> 标准化 -> baseline -> LSTM -> 信号诊断 -> pandas 回测 -> vn.py 接入
```

改造重点：

- 标签只保留少数有明确金融含义的版本。
- 所有跨模块变量统一放到一个全局配置 cell。
- 每个模型模块的参数集中放在该模块最前面。
- 主模型目标先固定为成本过滤后的二分类标签，兼容 BCE 和 AUC。
- Triple Barrier 标签先作为高级对照实验，不直接作为第一主线。

## 3. 全局配置设计

建议在 notebook 前部增加一个“全局配置”cell，放在数据读取之后、特征工程之前。

建议配置如下：

```python
from pathlib import Path

OUTPUT_DIR = Path("outputs/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
EPS = 1e-12

BAR_MINUTES = 5
TRADING_DAYS_PER_YEAR = 242
BARS_PER_DAY = 48
PERIODS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR

# 标签和交易成本
HORIZON = 3
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005
TOTAL_COST = COMMISSION_RATE + SLIPPAGE_RATE

LABEL_THRESHOLD = TOTAL_COST
LABEL_EXTRA_MARGIN = 0.0005
LABEL_UP_THRESHOLD = TOTAL_COST + LABEL_EXTRA_MARGIN
LABEL_DOWN_THRESHOLD = -(TOTAL_COST + LABEL_EXTRA_MARGIN)

TRIPLE_BARRIER_HORIZON = 12
TRIPLE_BARRIER_TAKE_PROFIT = 0.003
TRIPLE_BARRIER_STOP_LOSS = -0.002

# 主实验设置
TARGET_COL = "label_cost_binary"
LOOKBACK = 32
SEQUENCE_MODE = "continuous"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 信号和回测
SIGNAL_QUANTILES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
FIXED_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
PANDAS_BT_THRESHOLD = 0.55
PANDAS_BT_USE_QUANTILE = False
PANDAS_BT_QUANTILE = 0.90
```

注意：后续代码中尽量不再写死 `0.001`、`0.0005`、`32`、`3`、`0.55` 等参数，而是引用这些全局变量。

## 4. 标签改造方向

旧标签中临时探索性质比较强的部分先不作为主线，包括：

- `label_dynamic_binary`
- `label_dynamic_3class`
- `label_trade_success`
- `hit_take_profit`
- `hit_stop_loss`
- `dynamic_up_threshold`
- `dynamic_down_threshold`
- `target_vol`

保留并重构为以下几类。

### 4.1 未来收益回归标签

用途：保留为分析字段，不作为当前 BCE/AUC 主模型目标。

```python
df["future_close"] = df["close"].shift(-HORIZON)
df["future_return"] = df["future_close"] / df["close"] - 1
df["future_log_return"] = np.log(df["future_close"] / df["close"])
```

说明：

- 这是最基础的未来收益定义。
- 后续分层收益、回测诊断、label 质量评估都要用它。
- 当前题目强调 BCE 和 AUC，因此主模型暂不做回归。

### 4.2 简单方向二分类标签

用途：作为最基础 baseline label。

```python
df["label_direction_binary"] = (df["future_return"] > 0).astype(int)
```

说明：

- 优点是简单，正负样本通常比较均衡。
- 缺点是很多微小上涨无法覆盖成本，对交易不够友好。
- 可以作为对照，不建议作为最终主标签。

### 4.3 成本过滤后二分类标签

用途：建议作为当前主标签。

```python
df["label_cost_binary"] = (df["future_return"] > LABEL_THRESHOLD).astype(int)
```

其中：

```python
LABEL_THRESHOLD = TOTAL_COST
```

或更保守：

```python
LABEL_THRESHOLD = TOTAL_COST + 0.0005
```

说明：

- 含义是未来收益必须至少覆盖佣金和滑点，才认为存在可交易上涨机会。
- 与 BCE、AUC、long-only T+0 模拟最匹配。
- 当前 A股普通股票不能裸卖空，因此先预测“是否值得做多/低吸”比同时预测多空更现实。

### 4.4 带阈值三分类标签

用途：作为进阶实验标签，适合后续多分类模型或信号分层。

```python
df["label_threshold_3class"] = np.where(
    df["future_return"] > LABEL_UP_THRESHOLD,
    1,
    np.where(df["future_return"] < LABEL_DOWN_THRESHOLD, -1, 0),
)
```

说明：

- `1` 表示显著上涨。
- `0` 表示噪声区，不交易。
- `-1` 表示显著下跌。
- 如果继续使用 BCE，则可以从这个标签派生：

```python
df["label_threshold_binary"] = (df["label_threshold_3class"] == 1).astype(int)
```

### 4.5 Triple Barrier 标签

用途：作为高级交易事件标签，后续验证用，不建议第一版主线直接依赖。

逻辑：

```text
未来 N 根 K 线内：
先触发止盈 -> 1
先触发止损 -> -1
都没有触发 -> 0
```

建议字段：

```python
df["label_triple_barrier_3class"]
df["label_triple_barrier_binary"]
df["tb_first_touch"]
df["tb_touch_bar"]
```

说明：

- Triple Barrier 更贴近交易，因为它考虑了止盈、止损和最大持仓时间。
- 但实现和解释复杂度更高，因此当前阶段建议作为对照实验，而不是第一主线。

## 5. 标签构造模块建议

建议把标签构造 cell 改成函数化结构：

```python
def add_fixed_horizon_labels(df):
    df = df.copy()
    df["future_close"] = df["close"].shift(-HORIZON)
    df["future_return"] = df["future_close"] / df["close"] - 1
    df["future_log_return"] = np.log(df["future_close"] / df["close"])
    df["next_ret_1"] = df["close"].shift(-1) / df["close"] - 1

    df["label_direction_binary"] = (df["future_return"] > 0).astype(int)
    df["label_cost_binary"] = (df["future_return"] > LABEL_THRESHOLD).astype(int)
    df["label_threshold_3class"] = np.where(
        df["future_return"] > LABEL_UP_THRESHOLD,
        1,
        np.where(df["future_return"] < LABEL_DOWN_THRESHOLD, -1, 0),
    )
    df["label_threshold_binary"] = (df["label_threshold_3class"] == 1).astype(int)
    return df
```

Triple Barrier 可以单独一个函数：

```python
def add_triple_barrier_labels(df):
    ...
    return df
```

最后统一：

```python
df = add_fixed_horizon_labels(df)
df = add_triple_barrier_labels(df)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["future_return", "future_log_return", "next_ret_1", TARGET_COL]).copy()
```

不要再对全表直接 `dropna()`，否则一些非主线标签或辅助字段缺失可能导致样本被过度删除。更稳妥的是先对核心字段 dropna。

## 6. 特征选择改造方向

特征选择模块需要跟随新标签清理。

建议把目标字段统一维护为：

```python
LABEL_COLS = [
    "label_direction_binary",
    "label_cost_binary",
    "label_threshold_3class",
    "label_threshold_binary",
    "label_triple_barrier_3class",
    "label_triple_barrier_binary",
]

FUTURE_COLS = [
    "future_close",
    "future_return",
    "future_log_return",
    "next_ret_1",
    "tb_first_touch",
    "tb_touch_bar",
]
```

然后：

```python
target_cols = [c for c in LABEL_COLS + FUTURE_COLS if c in df.columns]
exclude_cols = base_exclude_cols + target_cols
```

保留现有的 future/leak keyword 检查：

```python
leak_keywords = ["future", "label", "target", "next", "hit", "tb_"]
```

## 7. 时间切分模块改造方向

原来在时间切分 cell 中写：

```python
target_col = "label_dynamic_binary"
```

后续应改成：

```python
target_col = TARGET_COL
```

切分比例也引用全局配置：

```python
train_end = int(n * TRAIN_RATIO)
valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))
```

这样后续切换标签或切分比例，只需要改全局配置。

## 8. 实验矩阵改造方向

原实验矩阵中的标签：

```python
["label", "label_cost_adjusted", "label_dynamic_binary"]
```

建议改成：

```python
EXPERIMENT_TARGET_COLS = [
    "label_direction_binary",
    "label_cost_binary",
    "label_threshold_binary",
    "label_triple_barrier_binary",
]
```

实验矩阵可以保留：

```python
LOOKBACK: 8, 16, 24, 32
HORIZON: 1, 3, 6
SEQUENCE_MODE: continuous, intraday_only
```

但注意：如果实验矩阵真的改变 `HORIZON`，标签需要按对应 horizon 重新生成。当前 notebook 里只是生成实验计划，不是真正循环训练，因此可以先作为计划表保存。

## 9. LSTM 模块参数整理

LSTM 模型 cell 顶部增加参数区：

```python
LSTM_PARAMS = {
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
}

TRAINING_PARAMS = {
    "batch_size_train": 128,
    "batch_size_eval": 256,
    "lr": 5e-4,
    "weight_decay": 1e-6,
    "max_epochs": 100,
    "patience": 10,
    "grad_clip": 1.0,
    "use_pos_weight": True,
}
```

DataLoader 中使用：

```python
batch_size=TRAINING_PARAMS["batch_size_train"]
```

训练循环中使用：

```python
for epoch in range(TRAINING_PARAMS["max_epochs"]):
...
if wait >= TRAINING_PARAMS["patience"]:
...
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAINING_PARAMS["grad_clip"])
```

模型初始化中使用：

```python
model = LSTMClassifier(
    input_dim=len(feature_cols),
    **LSTM_PARAMS,
).to(device)
```

## 10. RandomForest 模块参数整理

RandomForest cell 顶部增加：

```python
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 50,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}
```

然后：

```python
rf_model = RandomForestClassifier(**RF_PARAMS)
```

这样后续调 baseline 不需要在 cell 中到处找参数。

## 11. 信号诊断和回测参数整理

信号诊断中的成本和分位数不要重复定义：

```python
for q in SIGNAL_QUANTILES:
    ...

for threshold in FIXED_THRESHOLDS:
    ...
```

成本统一引用：

```python
TOTAL_COST
PERIODS_PER_YEAR
```

pandas 回测中也不要再重复写：

```python
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.0005
```

## 12. 回测口径后续优化

当前阶段可以先保留原有诊断式回测，但需要在文档或注释中说明：

```text
当前 quick pandas 回测主要用于观察模型信号强度，不是最终严格交易回测。
```

下一步更严格的回测应改为：

- 当前 bar 生成信号。
- 下一根 bar 执行。
- 用 `next_ret_1` 做逐 bar 持仓收益。
- 仓位变化时扣交易成本。
- 尾盘强制平仓。
- 可加入止损和最大持仓时间。

推荐后续新增字段：

```python
pandas_bt["position"] = pandas_bt["signal"].shift(1).fillna(0)
pandas_bt["raw_strategy_return"] = pandas_bt["position"] * pandas_bt["next_ret_1"]
```

## 13. 推荐执行顺序

后续真正改 notebook 时，建议按以下顺序执行：

1. 新增全局配置 cell。
2. 替换标签构造 cell。
3. 替换特征排除列表。
4. 把 `target_col = "label_dynamic_binary"` 改为 `target_col = TARGET_COL`。
5. 更新 Label 诊断候选列。
6. 更新实验矩阵中的 label 列表。
7. 更新 DataLoader、LSTM、RF 参数引用。
8. 更新信号诊断和 pandas 回测中的成本/阈值引用。
9. 重新运行到 RandomForest baseline，先确认 baseline 正常。
10. 再运行 LSTM，比较新标签下的 AUC、分层收益和 top 分位交易表现。

## 14. 当前主线建议

当前最稳的主线设置建议为：

```text
TARGET_COL = label_cost_binary
HORIZON = 3
LOOKBACK = 32
SEQUENCE_MODE = continuous
LABEL_THRESHOLD = 0.0015
```

然后做两个对照：

```text
label_direction_binary
label_threshold_binary
```

Triple Barrier 暂时作为进阶实验，等主流程稳定后再切换。

## 15. 面试表述口径

可以这样解释这次改造：

```text
我没有继续堆更多临时标签，而是把标签体系收敛为几类金融机器学习中更常见的定义：未来收益、方向二分类、成本过滤后的交易机会二分类、带阈值三分类和 Triple Barrier。当前主模型使用成本过滤后的二分类标签，因为它与 BCE、AUC 和 A股 T+0 long-only 模拟最匹配。代码上，我把 horizon、交易成本、目标标签、序列窗口、模型参数和回测阈值统一成全局配置，后续做实验只需要改配置，不需要在 notebook 多处手动修改。
```


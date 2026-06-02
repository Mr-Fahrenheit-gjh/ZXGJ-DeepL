# 中芯国际 688981 5分钟K线特征工程清单

本文档用于整理“使用 VeighNa/vn.py 对中芯国际（688981）进行 5分钟或更大周期走势预测与 T+0 回测”的特征工程方案。

假设原始数据字段为：

```text
open_t, high_t, low_t, close_t, volume_t, amount_t
```

其中 `t` 表示当前第 `t` 根 5分钟K线。

## 1. 收益率类特征

### 1.1 单周期收益率

```text
return_1_t = close_t / close_{t-1} - 1
```

### 1.2 多周期收益率

```text
return_n_t = close_t / close_{t-n} - 1
```

常用窗口：

```text
n = 3, 6, 12, 24
```

分别对应：

```text
15分钟、30分钟、1小时、2小时
```

### 1.3 对数收益率

```text
log_return_t = ln(close_t / close_{t-1})
```

### 1.4 未来收益率

该字段通常用于构造预测标签。

```text
future_return_t = close_{t+h} / close_t - 1
```

如果预测未来 15分钟走势，则：

```text
h = 3
```

## 2. 价格位置类特征

### 2.1 收盘价相对开盘价

```text
body_ratio_t = (close_t - open_t) / open_t
```

### 2.2 振幅

```text
amplitude_t = (high_t - low_t) / close_{t-1}
```

### 2.3 收盘价在当前K线区间中的位置

```text
close_position_t = (close_t - low_t) / (high_t - low_t)
```

如果 `high_t - low_t = 0`，需要进行异常处理。

### 2.4 上影线比例

```text
upper_shadow_t = (high_t - max(open_t, close_t)) / open_t
```

### 2.5 下影线比例

```text
lower_shadow_t = (min(open_t, close_t) - low_t) / open_t
```

### 2.6 实体占比

```text
body_to_range_t = abs(close_t - open_t) / (high_t - low_t)
```

## 3. 均线趋势特征

### 3.1 移动平均线

```text
MA_n_t = mean(close_{t-n+1}, ..., close_t)
```

常用窗口：

```text
n = 5, 10, 20, 48, 96
```

对于 5分钟K线，分别对应：

```text
25分钟、50分钟、100分钟、1天、2天
```

### 3.2 收盘价偏离均线

```text
close_ma_n_gap_t = close_t / MA_n_t - 1
```

### 3.3 短长均线差

```text
ma_short_long_gap_t = MA_short_t / MA_long_t - 1
```

示例：

```text
MA_5 / MA_20 - 1
MA_10 / MA_48 - 1
```

### 3.4 均线斜率

```text
ma_slope_n_t = MA_n_t / MA_n_{t-k} - 1
```

## 4. 波动率特征

### 4.1 滚动收益率标准差

```text
volatility_n_t = std(return_1_{t-n+1}, ..., return_1_t)
```

### 4.2 真实波动范围 TR

```text
TR_t = max(
    high_t - low_t,
    abs(high_t - close_{t-1}),
    abs(low_t - close_{t-1})
)
```

### 4.3 ATR

```text
ATR_n_t = mean(TR_{t-n+1}, ..., TR_t)
```

### 4.4 归一化 ATR

```text
atr_ratio_t = ATR_n_t / close_t
```

## 5. 成交量与成交额特征

### 5.1 成交量变化率

```text
volume_change_t = volume_t / volume_{t-1} - 1
```

### 5.2 成交额变化率

```text
amount_change_t = amount_t / amount_{t-1} - 1
```

### 5.3 成交量均线

```text
volume_MA_n_t = mean(volume_{t-n+1}, ..., volume_t)
```

### 5.4 放量比例

```text
volume_ratio_n_t = volume_t / volume_MA_n_t
```

### 5.5 成交额放大比例

```text
amount_ratio_n_t = amount_t / amount_MA_n_t
```

### 5.6 VWAP

```text
vwap_t = amount_t / volume_t
```

如果 `amount` 单位为元，`volume` 单位为股，则该公式成立。

### 5.7 收盘价偏离 VWAP

```text
close_vwap_gap_t = close_t / vwap_t - 1
```

## 6. 技术指标特征

### 6.1 MACD

```text
EMA_fast_t = EMA(close, 12)
EMA_slow_t = EMA(close, 26)

DIF_t = EMA_fast_t - EMA_slow_t
DEA_t = EMA(DIF, 9)
MACD_t = 2 * (DIF_t - DEA_t)
```

### 6.2 RSI

```text
RSI_n_t = 100 - 100 / (1 + RS_t)
```

其中：

```text
RS_t = mean(up returns over n periods) / mean(down returns over n periods)
```

### 6.3 Bollinger Bands

```text
MID_t = MA_n_t
UPPER_t = MID_t + k * std(close, n)
LOWER_t = MID_t - k * std(close, n)
```

常用参数：

```text
n = 20
k = 2
```

布林带宽度：

```text
boll_width_t = (UPPER_t - LOWER_t) / MID_t
```

价格相对布林带位置：

```text
boll_position_t = (close_t - LOWER_t) / (UPPER_t - LOWER_t)
```

### 6.4 Momentum

```text
momentum_n_t = close_t - close_{t-n}
```

### 6.5 ROC

```text
ROC_n_t = close_t / close_{t-n} - 1
```

## 7. 日内时间特征

### 7.1 小时

```text
hour_t = hour(datetime_t)
```

### 7.2 分钟

```text
minute_t = minute(datetime_t)
```

### 7.3 距离开盘第几根K线

```text
bar_index_in_day_t = 当日第几根5分钟K线
```

### 7.4 是否开盘前30分钟

```text
is_open_period_t = 1 if 09:30 <= time_t <= 10:00 else 0
```

### 7.5 是否收盘前30分钟

```text
is_close_period_t = 1 if 14:30 <= time_t <= 15:00 else 0
```

## 8. 滚动统计特征

### 8.1 滚动最高价

```text
rolling_high_n_t = max(high_{t-n+1}, ..., high_t)
```

### 8.2 滚动最低价

```text
rolling_low_n_t = min(low_{t-n+1}, ..., low_t)
```

### 8.3 当前价格相对过去高点

```text
close_to_high_n_t = close_t / rolling_high_n_t - 1
```

### 8.4 当前价格相对过去低点

```text
close_to_low_n_t = close_t / rolling_low_n_t - 1
```

### 8.5 滚动偏度

```text
skew_n_t = skew(return_1 over past n periods)
```

### 8.6 滚动峰度

```text
kurt_n_t = kurt(return_1 over past n periods)
```

## 9. 标签构造

### 9.1 二分类标签

预测未来是否上涨：

```text
label_t = 1 if future_return_t > 0 else 0
```

### 9.2 考虑交易成本的二分类标签

题目中交易成本包括：

```text
commission = 0.001
slippage = 0.0005
cost = 0.0015
```

标签为：

```text
label_t = 1 if future_return_t > 0.0015 else 0
```

### 9.3 三分类标签

更适合 T+0 交易信号：

```text
label_t = 1   if future_return_t > cost
label_t = 0   if abs(future_return_t) <= cost
label_t = -1  if future_return_t < -cost
```

## 10. MVP 推荐特征组合

### 10.1 收益率类

```text
return_1
return_3
return_6
log_return
```

### 10.2 趋势类

```text
MA_5
MA_10
MA_20
close_ma5_gap
ma5_ma20_gap
```

### 10.3 波动类

```text
amplitude
volatility_10
volatility_20
ATR_14
```

### 10.4 成交量类

```text
volume_change
volume_ratio_10
amount_ratio_10
close_vwap_gap
```

### 10.5 K线形态类

```text
body_ratio
upper_shadow
lower_shadow
close_position
```

### 10.6 技术指标类

```text
RSI_14
MACD
MACD_signal
MACD_hist
boll_width
boll_position
```

### 10.7 时间特征

```text
bar_index_in_day
is_open_period
is_close_period
```

## 11. 使用建议

模型训练阶段建议优先使用前复权数据，以减少除权除息造成的价格断层。

真实交易回测阶段建议使用不复权数据，以更接近实际成交价格。

对于 5分钟数据，可以先使用 BaoStock 获取长期样本，再使用 AkShare 近期数据进行字段校验和结果对比。

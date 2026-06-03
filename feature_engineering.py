from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=PerformanceWarning, module=__name__)

EPS = 1e-12
WINDOWS = [1, 3, 5, 10, 20, 48]
SHORT_WINDOWS = [3, 5, 10, 20]
MID_WINDOWS = [5, 10, 20, 48]


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"])
            out = out.set_index("datetime")
        else:
            out.index = pd.to_datetime(out.index)
    return out.sort_index()


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_datetime_index(df)
    out = out.copy()
    out["trade_date"] = out.index.normalize()

    out["ret_1"] = out["close"].pct_change(1)
    out["log_close"] = np.log(out["close"] + EPS)
    for w in WINDOWS:
        out[f"ret_{w}"] = out["close"].pct_change(w)
        out[f"log_ret_{w}"] = out["log_close"].diff(w)

    for w in MID_WINDOWS:
        out[f"ma_{w}"] = out["close"].rolling(w).mean()
        out[f"ma_dev_{w}"] = out["close"] / (out[f"ma_{w}"] + EPS) - 1
        out[f"ma_slope_{w}"] = out[f"ma_{w}"] / (out[f"ma_{w}"].shift(1) + EPS) - 1
    out["ma_gap_5_20"] = out["ma_5"] / (out["ma_20"] + EPS) - 1
    out["ma_gap_10_48"] = out["ma_10"] / (out["ma_48"] + EPS) - 1

    out["mom_3_10"] = out["ret_3"] - out["ret_10"]
    out["mom_5_20"] = out["ret_5"] - out["ret_20"]
    out["mom_10_48"] = out["ret_10"] - out["ret_48"]
    out["ret_recent_3"] = out["close"] / (out["close"].shift(3) + EPS) - 1
    out["ret_prev_3"] = out["close"].shift(3) / (out["close"].shift(6) + EPS) - 1
    out["ret_prev_6"] = out["close"].shift(3) / (out["close"].shift(9) + EPS) - 1
    out["mom_change_3"] = out["ret_recent_3"] - out["ret_prev_3"]
    out["mom_change_6"] = out["ret_recent_3"] - out["ret_prev_6"]
    out["up_bar"] = (out["close"] > out["close"].shift(1)).astype(int)
    out["down_bar"] = (out["close"] < out["close"].shift(1)).astype(int)
    for w in SHORT_WINDOWS:
        out[f"up_ratio_{w}"] = out["up_bar"].rolling(w).mean()
        out[f"down_ratio_{w}"] = out["down_bar"].rolling(w).mean()

    out["body"] = (out["close"] - out["open"]) / (out["open"] + EPS)
    out["bar_range"] = (out["high"] - out["low"]) / (out["close"] + EPS)
    out["upper_shadow"] = (out["high"] - out[["open", "close"]].max(axis=1)) / (out["close"] + EPS)
    out["lower_shadow"] = (out[["open", "close"]].min(axis=1) - out["low"]) / (out["close"] + EPS)
    out["close_position"] = (out["close"] - out["low"]) / (out["high"] - out["low"] + EPS)

    for w in MID_WINDOWS:
        out[f"vol_{w}"] = out["ret_1"].rolling(w).std()
        out[f"range_mean_{w}"] = out["bar_range"].rolling(w).mean()
        out[f"range_vol_{w}"] = out["bar_range"].rolling(w).std()
    hl = np.log((out["high"] + EPS) / (out["low"] + EPS)) ** 2
    for w in MID_WINDOWS:
        out[f"parkinson_vol_{w}"] = np.sqrt(hl.rolling(w).mean() / (4 * np.log(2)))
    out["vol_ratio_5_20"] = out["vol_5"] / (out["vol_20"] + EPS)
    out["vol_ratio_10_48"] = out["vol_10"] / (out["vol_48"] + EPS)
    out["mom_z_3_20"] = out["ret_3"] / (out["vol_20"] + EPS)
    out["mom_z_5_20"] = out["ret_5"] / (out["vol_20"] + EPS)
    out["mom_z_10_48"] = out["ret_10"] / (out["vol_48"] + EPS)
    for i in [5, 10, 20]:
        for w in [20, 48]:
            vol_mean = out[f"vol_{i}"].rolling(w).mean()
            vol_std = out[f"vol_{i}"].rolling(w).std()
            out[f"vol_zscore_{i}_{w}"] = (out[f"vol_{i}"] - vol_mean) / (vol_std + EPS)

    out["vwap_bar"] = out["amount"] / (out["volume"] + EPS)
    out["vwap_bar"] = out["vwap_bar"].fillna(out["close"])
    out["vwap_bar_dev"] = out["close"] / (out["vwap_bar"] + EPS) - 1
    out["amount_cum_day"] = out.groupby("trade_date")["amount"].cumsum()
    out["volume_cum_day"] = out.groupby("trade_date")["volume"].cumsum()
    out["vwap_cum_day"] = out["amount_cum_day"] / (out["volume_cum_day"] + EPS)
    out["vwap_cum_day"] = out["vwap_cum_day"].fillna(out["close"])
    out["vwap_dev_day"] = out["close"] / (out["vwap_cum_day"] + EPS) - 1
    out["vwap_cum_slope_3"] = out["vwap_cum_day"] / (out["vwap_cum_day"].shift(3) + EPS) - 1
    out["vwap_cum_slope_6"] = out["vwap_cum_day"] / (out["vwap_cum_day"].shift(6) + EPS) - 1
    for w in MID_WINDOWS:
        amount_roll = out["amount"].rolling(w).sum()
        volume_roll = out["volume"].rolling(w).sum()
        out[f"vwap_roll_{w}"] = amount_roll / (volume_roll + EPS)
        out[f"vwap_roll_{w}"] = out[f"vwap_roll_{w}"].fillna(out["close"])
        out[f"vwap_roll_dev_{w}"] = out["close"] / (out[f"vwap_roll_{w}"] + EPS) - 1
        out[f"vwap_roll_slope_{w}"] = out[f"vwap_roll_{w}"] / (out[f"vwap_roll_{w}"].shift(1) + EPS) - 1
    out["vwap_spread_20_day"] = out["vwap_roll_20"] / (out["vwap_cum_day"] + EPS) - 1
    out["vwap_spread_48_day"] = out["vwap_roll_48"] / (out["vwap_cum_day"] + EPS) - 1

    out["clv"] = ((out["close"] - out["low"]) - (out["high"] - out["close"])) / (out["high"] - out["low"] + EPS)
    out["clv"] = out["clv"].clip(-1, 1)
    for w in MID_WINDOWS:
        out[f"clv_mean_{w}"] = out["clv"].rolling(w).mean()
        out[f"clv_sum_{w}"] = out["clv"].rolling(w).sum()
    out["clv_volume"] = out["clv"] * np.log1p(out["volume"])
    for w in MID_WINDOWS:
        out[f"clv_volume_sum_{w}"] = out["clv_volume"].rolling(w).sum()
        out[f"clv_volume_mean_{w}"] = out["clv_volume"].rolling(w).mean()
    out["money_flow_volume"] = out["clv"] * out["volume"]
    for w in [10, 20, 48]:
        mfv_sum = out["money_flow_volume"].rolling(w).sum()
        vol_sum = out["volume"].rolling(w).sum()
        out[f"adl_roll_norm_{w}"] = mfv_sum / (vol_sum + EPS)

    for w in MID_WINDOWS:
        out[f"volume_ma_{w}"] = out["volume"].rolling(w).mean()
        out[f"amount_ma_{w}"] = out["amount"].rolling(w).mean()
        out[f"volume_ratio_{w}"] = out["volume"] / (out[f"volume_ma_{w}"] + EPS)
        out[f"amount_ratio_{w}"] = out["amount"] / (out[f"amount_ma_{w}"] + EPS)
        out[f"log_volume_ratio_{w}"] = np.log1p(out["volume"]) - np.log1p(out[f"volume_ma_{w}"])
        out[f"log_amount_ratio_{w}"] = np.log1p(out["amount"]) - np.log1p(out[f"amount_ma_{w}"])
    out["volume_change"] = out["volume"] / (out["volume"].shift(1) + EPS) - 1
    out["amount_change"] = out["amount"] / (out["amount"].shift(1) + EPS) - 1
    out["ret_volume"] = out["ret_1"] * np.log1p(out["volume"])
    out["ret_amount"] = out["ret_1"] * np.log1p(out["amount"])
    for w in [5, 10, 20]:
        out[f"ret_volume_sum_{w}"] = out["ret_volume"].rolling(w).sum()
        out[f"ret_amount_sum_{w}"] = out["ret_amount"].rolling(w).sum()

    out["signed_volume"] = np.sign(out["ret_1"]) * np.log1p(out["volume"])
    out["signed_amount"] = np.sign(out["ret_1"]) * np.log1p(out["amount"])
    for w in [5, 10, 20, 48]:
        out[f"signed_volume_sum_{w}"] = out["signed_volume"].rolling(w).sum()
        out[f"signed_amount_sum_{w}"] = out["signed_amount"].rolling(w).sum()
        out[f"signed_volume_mean_{w}"] = out["signed_volume"].rolling(w).mean()
        out[f"signed_amount_mean_{w}"] = out["signed_amount"].rolling(w).mean()
    out["up_volume_ratio"] = out["up_bar"] * out["volume_ratio_20"]
    out["down_volume_ratio"] = out["down_bar"] * out["volume_ratio_20"]
    out["up_amount_ratio"] = out["up_bar"] * out["amount_ratio_20"]
    out["down_amount_ratio"] = out["down_bar"] * out["amount_ratio_20"]
    for w in [5, 10, 20]:
        out[f"up_volume_ratio_mean_{w}"] = out["up_volume_ratio"].rolling(w).mean()
        out[f"down_volume_ratio_mean_{w}"] = out["down_volume_ratio"].rolling(w).mean()
        out[f"up_amount_ratio_mean_{w}"] = out["up_amount_ratio"].rolling(w).mean()
        out[f"down_amount_ratio_mean_{w}"] = out["down_amount_ratio"].rolling(w).mean()

    out["bar_in_day"] = out.groupby("trade_date").cumcount()
    out["bars_in_day"] = out.groupby("trade_date")["close"].transform("count")
    out["bar_in_day_pct"] = out["bar_in_day"] / (out["bars_in_day"] - 1 + EPS)
    out["is_morning"] = ((out.index.time >= pd.Timestamp("09:30").time()) & (out.index.time <= pd.Timestamp("11:30").time())).astype(int)
    out["is_afternoon"] = ((out.index.time >= pd.Timestamp("13:00").time()) & (out.index.time <= pd.Timestamp("15:00").time())).astype(int)
    out["is_open_30min"] = (out["bar_in_day"] < 6).astype(int)
    out["is_close_30min"] = (out["bar_in_day"] >= out["bars_in_day"] - 6).astype(int)
    out["is_after_lunch_open"] = ((out.index.time >= pd.Timestamp("13:00").time()) & (out.index.time <= pd.Timestamp("13:30").time())).astype(int)

    trade_dates = pd.Series(out["trade_date"].unique()).sort_values()
    calendar = pd.DataFrame({"trade_date": trade_dates})
    calendar["prev_trade_date"] = calendar["trade_date"].shift(1)
    calendar["gap_days"] = (calendar["trade_date"] - calendar["prev_trade_date"]).dt.days.fillna(1)
    calendar["is_normal_day"] = (calendar["gap_days"] == 1).astype(int)
    calendar["is_after_weekend"] = ((calendar["gap_days"] == 3) & (calendar["trade_date"].dt.dayofweek == 0)).astype(int)
    calendar["is_after_holiday"] = (calendar["gap_days"] > 3).astype(int)
    calendar_cols = ["gap_days", "is_normal_day", "is_after_weekend", "is_after_holiday"]
    out = out.drop(columns=[c for c in calendar_cols if c in out.columns])
    out = out.join(calendar.set_index("trade_date")[calendar_cols], on="trade_date")
    out[calendar_cols] = out[calendar_cols].fillna(0)

    out["volume_same_bar_mean_20d"] = out.groupby("bar_in_day")["volume"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    out["amount_same_bar_mean_20d"] = out.groupby("bar_in_day")["amount"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    out["volume_same_bar_ratio"] = out["volume"] / (out["volume_same_bar_mean_20d"] + EPS)
    out["amount_same_bar_ratio"] = out["amount"] / (out["amount_same_bar_mean_20d"] + EPS)
    out["log_volume_same_bar_ratio"] = np.log1p(out["volume"]) - np.log1p(out["volume_same_bar_mean_20d"])
    out["log_amount_same_bar_ratio"] = np.log1p(out["amount"]) - np.log1p(out["amount_same_bar_mean_20d"])

    for w in [10, 20, 48]:
        price_mean = out["close"].rolling(w).mean()
        price_std = out["close"].rolling(w).std()
        out[f"price_zscore_{w}"] = (out["close"] - price_mean) / (price_std + EPS)
    out["cum_high_day"] = out.groupby("trade_date")["high"].cummax()
    out["cum_low_day"] = out.groupby("trade_date")["low"].cummin()
    out["intraday_range_position"] = (out["close"] - out["cum_low_day"]) / (out["cum_high_day"] - out["cum_low_day"] + EPS)

    out["amihud"] = out["ret_1"].abs() / (out["amount"] + EPS)
    for w in [10, 20, 48]:
        out[f"amihud_{w}"] = out["amihud"].rolling(w).mean()
    out["range_per_volume"] = out["bar_range"] / (out["volume"] + EPS)
    for w in [10, 20, 48]:
        out[f"range_per_volume_{w}"] = out["range_per_volume"].rolling(w).mean()

    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out.copy()


BASE_EXCLUDE_COLS = [
    "open", "high", "low", "close", "volume", "amount", "trade_date",
    "buy_exit_reason", "sell_exit_reason",
    "log_close", "vwap_bar", "vwap_cum_day", "amount_cum_day", "volume_cum_day",
    "cum_high_day", "cum_low_day", "money_flow_volume",
]

TARGET_AND_LEAK_COLS = [
    "buy_label", "sell_label", "label_t0_action", "label_buy_signal", "label_sell_signal",
    "label_round_trip_net", "label_cost_net", "label", "label_cost_adjusted", "label_3class",
    "label_dynamic_3class", "label_dynamic_binary", "label_trade_success", "label_net_up",
    "label_path_up", "label_quality_up", "label_down", "label_path_down", "future_return",
    "future_log_return", "future_max_return", "future_min_return", "next_ret_1", "trade_return",
    "buy_gross_return", "sell_gross_return", "buy_net_return_est", "sell_net_return_est",
    "future_close", "future_entry_open", "future_max_close", "future_min_close",
    "future_max_price", "future_min_price", "hit_take_profit", "hit_stop_loss",
    "buy_exit_bar", "sell_exit_bar", "buy_exit_price", "sell_exit_price",
    "label_horizon", "label_tp", "label_sl", "label_cost", "target_vol",
    "dynamic_up_threshold", "dynamic_down_threshold",
]


def _is_absolute_level_feature(col: str) -> bool:
    if col.startswith(("volume_ma_", "amount_ma_", "volume_same_bar_mean_", "amount_same_bar_mean_")):
        return True
    if col.startswith("ma_") and not col.startswith(("ma_dev_", "ma_slope_", "ma_gap_")):
        return True
    if col.startswith("vwap_roll_") and not col.startswith(("vwap_roll_dev_", "vwap_roll_slope_")):
        return True
    return False


def select_feature_columns(df: pd.DataFrame, extra_exclude: Iterable[str] | None = None) -> tuple[list[str], dict]:
    exclude = set(BASE_EXCLUDE_COLS) | set(TARGET_AND_LEAK_COLS)
    if extra_exclude:
        exclude |= set(extra_exclude)

    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    absolute_level_cols = sorted({c for c in feature_cols if _is_absolute_level_feature(c)})
    feature_cols = [c for c in feature_cols if c not in absolute_level_cols]

    leak_keywords = ["future", "label", "target", "next", "hit"]
    leak_cols = sorted({c for c in feature_cols if any(k in c.lower() for k in leak_keywords)})
    feature_cols = [c for c in feature_cols if c not in leak_cols]

    nunique = df[feature_cols].nunique()
    constant_cols = sorted([c for c in feature_cols if nunique[c] <= 1])
    feature_cols = [c for c in feature_cols if c not in constant_cols]

    report = {
        "absolute_level_cols": absolute_level_cols,
        "leak_cols": leak_cols,
        "constant_cols": constant_cols,
        "feature_count": len(feature_cols),
    }
    return feature_cols, report


def split_time_series(df: pd.DataFrame, train_ratio: float, valid_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.copy().sort_index()
    n = len(data)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return data.iloc[:train_end].copy(), data.iloc[train_end:valid_end].copy(), data.iloc[valid_end:].copy()


def winsorize_by_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    lower_q: float = 0.001,
    upper_q: float = 0.999,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train = train_df.copy()
    valid = valid_df.copy()
    test = test_df.copy()
    bounds = {}
    for col in feature_cols:
        lower = train[col].quantile(lower_q)
        upper = train[col].quantile(upper_q)
        bounds[col] = (lower, upper)
        train[col] = train[col].clip(lower, upper)
        valid[col] = valid[col].clip(lower, upper)
        test[col] = test[col].clip(lower, upper)
    return train, valid, test, bounds


def standardize_by_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train = train_df.copy()
    valid = valid_df.copy()
    test = test_df.copy()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    valid[feature_cols] = scaler.transform(valid[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    return train, valid, test, scaler


def make_sequence_data(data: pd.DataFrame, feature_cols: list[str], target_col: str, lookback: int):
    x = data[feature_cols].values
    y = data[target_col].values
    x_seq, y_seq, index_seq = [], [], []
    for i in range(lookback, len(data)):
        x_seq.append(x[i - lookback:i])
        y_seq.append(y[i])
        index_seq.append(data.index[i])
    return np.asarray(x_seq), np.asarray(y_seq), np.asarray(index_seq)


def save_run_config(run_config: dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    pd.Series(run_config).to_json(path, force_ascii=False, indent=2)
    return path

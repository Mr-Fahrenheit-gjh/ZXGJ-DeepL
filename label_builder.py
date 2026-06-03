from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierConfig:
    horizon: int = 3
    tp: float = 0.003
    sl: float = 0.002
    cost: float = 0.0015
    same_bar_policy: str = "pessimistic"
    timeout_policy: str = "final_return"
    buy_label_col: str = "buy_label"
    sell_label_col: str = "sell_label"


def _validate_ohlc(df: pd.DataFrame) -> None:
    required_cols = ["open", "high", "low", "close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")


def _timeout_label(gross_return: float, cost: float, timeout_policy: str) -> int:
    if pd.isna(gross_return):
        return 0
    if timeout_policy == "final_return":
        return int(gross_return > cost)
    if timeout_policy == "negative":
        return 0
    raise ValueError(f"unsupported timeout_policy: {timeout_policy}")


def build_path_dependent_opportunity_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    tp: float = 0.003,
    sl: float = 0.002,
    cost: float = 0.0015,
    same_bar_policy: str = "pessimistic",
    timeout_policy: str = "final_return",
    buy_label_col: str = "buy_label",
    sell_label_col: str = "sell_label",
) -> pd.DataFrame:
    """Build path-dependent buy/sell opportunity labels using a triple-barrier style rule.

    Features should be built outside this function using only information available at t or earlier.
    This function uses t+1..t+horizon future bars only for label construction.
    """
    _validate_ohlc(df)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if same_bar_policy != "pessimistic":
        raise ValueError("only same_bar_policy='pessimistic' is currently supported")

    out = df.copy().sort_index()
    n = len(out)

    open_arr = out["open"].to_numpy(dtype=float)
    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    close_arr = out["close"].to_numpy(dtype=float)

    buy_label = np.zeros(n, dtype=float)
    sell_label = np.zeros(n, dtype=float)

    buy_reason = np.full(n, "no_future", dtype=object)
    sell_reason = np.full(n, "no_future", dtype=object)
    buy_exit_bar = np.full(n, np.nan)
    sell_exit_bar = np.full(n, np.nan)
    buy_exit_price = np.full(n, np.nan)
    sell_exit_price = np.full(n, np.nan)
    buy_gross_return = np.full(n, np.nan)
    sell_gross_return = np.full(n, np.nan)
    future_entry_open = np.full(n, np.nan)
    future_close = np.full(n, np.nan)

    for i in range(n):
        entry_i = i + 1
        exit_i = i + horizon
        if entry_i >= n or exit_i >= n:
            continue

        entry_price = open_arr[entry_i]
        final_close = close_arr[exit_i]
        if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(final_close):
            continue

        future_entry_open[i] = entry_price
        future_close[i] = final_close

        buy_tp_price = entry_price * (1 + tp + cost)
        buy_sl_price = entry_price * (1 - sl)
        sell_tp_price = entry_price * (1 - tp - cost)
        sell_sl_price = entry_price * (1 + sl)

        buy_done = False
        sell_done = False

        for j in range(entry_i, exit_i + 1):
            bar_offset = j - i

            # Pessimistic same-bar rule: if both barriers are touched in one OHLC bar,
            # assume the adverse barrier was hit first.
            if not buy_done:
                buy_stop_hit = low_arr[j] <= buy_sl_price
                buy_take_hit = high_arr[j] >= buy_tp_price
                if buy_stop_hit:
                    buy_label[i] = 0
                    buy_reason[i] = "stop_loss"
                    buy_exit_bar[i] = bar_offset
                    buy_exit_price[i] = buy_sl_price
                    buy_gross_return[i] = buy_exit_price[i] / entry_price - 1
                    buy_done = True
                elif buy_take_hit:
                    buy_label[i] = 1
                    buy_reason[i] = "take_profit"
                    buy_exit_bar[i] = bar_offset
                    buy_exit_price[i] = buy_tp_price
                    buy_gross_return[i] = buy_exit_price[i] / entry_price - 1
                    buy_done = True

            if not sell_done:
                sell_stop_hit = high_arr[j] >= sell_sl_price
                sell_take_hit = low_arr[j] <= sell_tp_price
                if sell_stop_hit:
                    sell_label[i] = 0
                    sell_reason[i] = "stop_loss"
                    sell_exit_bar[i] = bar_offset
                    sell_exit_price[i] = sell_sl_price
                    sell_gross_return[i] = entry_price / sell_exit_price[i] - 1
                    sell_done = True
                elif sell_take_hit:
                    sell_label[i] = 1
                    sell_reason[i] = "take_profit"
                    sell_exit_bar[i] = bar_offset
                    sell_exit_price[i] = sell_tp_price
                    sell_gross_return[i] = entry_price / sell_exit_price[i] - 1
                    sell_done = True

            if buy_done and sell_done:
                break

        if not buy_done:
            gross = final_close / entry_price - 1
            buy_label[i] = _timeout_label(gross, cost, timeout_policy)
            buy_reason[i] = "timeout_win" if buy_label[i] == 1 else "timeout_loss"
            buy_exit_bar[i] = horizon
            buy_exit_price[i] = final_close
            buy_gross_return[i] = gross

        if not sell_done:
            gross = entry_price / final_close - 1
            sell_label[i] = _timeout_label(gross, cost, timeout_policy)
            sell_reason[i] = "timeout_win" if sell_label[i] == 1 else "timeout_loss"
            sell_exit_bar[i] = horizon
            sell_exit_price[i] = final_close
            sell_gross_return[i] = gross

    out["future_entry_open"] = future_entry_open
    out["future_close"] = future_close
    out["trade_return"] = out["future_close"] / out["future_entry_open"] - 1
    out["future_return"] = out["future_close"] / out["close"] - 1

    out[buy_label_col] = buy_label
    out[sell_label_col] = sell_label
    out["buy_exit_reason"] = buy_reason
    out["sell_exit_reason"] = sell_reason
    out["buy_exit_bar"] = buy_exit_bar
    out["sell_exit_bar"] = sell_exit_bar
    out["buy_exit_price"] = buy_exit_price
    out["sell_exit_price"] = sell_exit_price
    out["buy_gross_return"] = buy_gross_return
    out["sell_gross_return"] = sell_gross_return
    out["buy_net_return_est"] = out["buy_gross_return"] - cost
    out["sell_net_return_est"] = out["sell_gross_return"] - cost
    out["label_horizon"] = int(horizon)
    out["label_tp"] = float(tp)
    out["label_sl"] = float(sl)
    out["label_cost"] = float(cost)
    return out


def summarize_opportunity_labels(
    df: pd.DataFrame,
    buy_label_col: str = "buy_label",
    sell_label_col: str = "sell_label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    reason_rows = []
    for side, label_col, reason_col, ret_col, bar_col in [
        ("buy", buy_label_col, "buy_exit_reason", "buy_gross_return", "buy_exit_bar"),
        ("sell", sell_label_col, "sell_exit_reason", "sell_gross_return", "sell_exit_bar"),
    ]:
        valid = df[label_col].dropna()
        pos = df.loc[valid.index, label_col] == 1
        rows.append(
            {
                "side": side,
                "label_col": label_col,
                "sample_count": int(len(valid)),
                "positive_count": int(pos.sum()),
                "positive_rate": float(pos.mean()) if len(valid) else np.nan,
                "gross_return_mean_all": float(df.loc[valid.index, ret_col].mean()),
                "gross_return_mean_positive": float(df.loc[valid.index[pos], ret_col].mean()) if pos.any() else np.nan,
                "gross_return_mean_negative": float(df.loc[valid.index[~pos], ret_col].mean()) if (~pos).any() else np.nan,
                "avg_exit_bar": float(df.loc[valid.index, bar_col].mean()),
            }
        )
        if reason_col in df.columns:
            vc = df.loc[valid.index, reason_col].value_counts(dropna=False)
            for reason, count in vc.items():
                reason_rows.append(
                    {
                        "side": side,
                        "reason": reason,
                        "count": int(count),
                        "ratio": float(count / len(valid)) if len(valid) else np.nan,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(reason_rows)


def run_label_parameter_stability(
    df: pd.DataFrame,
    horizons=(3, 6, 9, 12),
    tps=(0.002, 0.003, 0.004),
    sls=(0.0015, 0.002, 0.003),
    cost: float = 0.0015,
    same_bar_policy: str = "pessimistic",
    timeout_policy: str = "final_return",
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    rows = []
    for horizon, tp, sl in product(horizons, tps, sls):
        labeled = build_path_dependent_opportunity_labels(
            df,
            horizon=horizon,
            tp=tp,
            sl=sl,
            cost=cost,
            same_bar_policy=same_bar_policy,
            timeout_policy=timeout_policy,
        )
        summary, _ = summarize_opportunity_labels(labeled)
        row = {
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
            "cost": cost,
        }
        for _, r in summary.iterrows():
            side = r["side"]
            row[f"{side}_positive_rate"] = r["positive_rate"]
            row[f"{side}_gross_return_mean_all"] = r["gross_return_mean_all"]
            row[f"{side}_avg_exit_bar"] = r["avg_exit_bar"]
        rows.append(row)

    result = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "label_parameter_stability.csv", index=False)
    return result


def config_to_dict(config: TripleBarrierConfig) -> dict:
    return asdict(config)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from risk_management import (
    DEFAULT_PERIODS_PER_YEAR,
    probability_scaled_position_pct,
    summarize_equity_curve,
)


OUTPUT_DIR = Path("outputs/diagnostics/rule_compare_from_existing_signals")
ASSUMED_MODEL_DIR = Path(
    "outputs/diagnostics/basic_deep_full_1fold/walk_forward/fold_000/models/auc_weighted_ensemble"
)
DATA_CANDIDATES = [
    Path("688981_5min_20200716-20260602.parquet"),
    Path("688981_5min_20200716-20260602-not_adjust.parquet"),
]

BACKTEST_CONFIG = {
    "initial_capital": 1_000_000,
    "base_position_pct": 0.6,
    "max_t0_trade_pct": 0.3,
    "min_position_pct": 0.25,
    "max_position_pct": 1.0,
    "position_sizing_high_quantile": 0.99,
    "commission_rate": 0.001,
    "slippage_rate": 0.0005,
    "tp": 0.003,
    "sl": 0.002,
    "min_lot_size": 100,
    "max_participation_rate": 0.05,
    "a_share_t0_mode": "inventory",
    "periods_per_year": DEFAULT_PERIODS_PER_YEAR,
}

RULES = [
    {
        "rule_name": "both_q95_q95",
        "trade_direction_mode": "both",
        "buy_quantile": 0.95,
        "sell_quantile": 0.95,
    },
    {
        "rule_name": "both_q98_q98",
        "trade_direction_mode": "both",
        "buy_quantile": 0.98,
        "sell_quantile": 0.98,
    },
    {
        "rule_name": "sell_only_q95",
        "trade_direction_mode": "sell_only",
        "buy_quantile": None,
        "sell_quantile": 0.95,
    },
    {
        "rule_name": "sell_only_q98",
        "trade_direction_mode": "sell_only",
        "buy_quantile": None,
        "sell_quantile": 0.98,
    },
    {
        "rule_name": "buy_only_q95",
        "trade_direction_mode": "buy_only",
        "buy_quantile": 0.95,
        "sell_quantile": None,
    },
    {
        "rule_name": "buy_only_q98",
        "trade_direction_mode": "buy_only",
        "buy_quantile": 0.98,
        "sell_quantile": None,
    },
]


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def round_lot_shares(raw_shares: float, lot_size: int) -> int:
    return int(max(raw_shares, 0) // lot_size * lot_size)


def volume_cap_shares(data: pd.DataFrame, row_pos: int, max_participation_rate: float, lot_size: int) -> int | None:
    if "volume" not in data.columns or pd.isna(data["volume"].iloc[row_pos]):
        return None
    return round_lot_shares(float(data["volume"].iloc[row_pos]) * max_participation_rate, lot_size)


def locate_signal_pair() -> tuple[Path, Path]:
    assumed_valid = ASSUMED_MODEL_DIR / "valid_signals.csv"
    assumed_test = ASSUMED_MODEL_DIR / "test_signals.csv"
    if assumed_valid.exists() and assumed_test.exists():
        return assumed_valid, assumed_test

    candidates = []
    for test_path in Path("outputs/diagnostics").rglob("test_signals.csv"):
        if "auc_weighted_ensemble" not in test_path.parts:
            continue
        valid_path = test_path.with_name("valid_signals.csv")
        if not valid_path.exists():
            continue
        path_text = str(test_path).replace("\\", "/")
        score = 0
        if "basic_deep_full_1fold" in path_text:
            score += 100
        if "basic_bucket_test_1fold" in path_text:
            score += 80
        if "1fold" in path_text:
            score += 40
        if "fold_000" in path_text:
            score += 10
        candidates.append((score, test_path.stat().st_mtime, valid_path, test_path))

    if not candidates:
        raise FileNotFoundError("No auc_weighted_ensemble valid/test signal pair found under outputs/diagnostics.")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, valid_path, test_path = candidates[0]
    return valid_path, test_path


def read_signals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "datetime"
    df = df.sort_index()
    required = {"buy_prob", "sell_prob"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required signal columns: {sorted(missing)}")
    return df


def load_price_data() -> pd.DataFrame:
    for path in DATA_CANDIDATES:
        if path.exists():
            price = pd.read_parquet(path)
            break
    else:
        raise FileNotFoundError(f"No price parquet found. Tried: {[str(p) for p in DATA_CANDIDATES]}")

    if not isinstance(price.index, pd.DatetimeIndex):
        if "datetime" in price.columns:
            price["datetime"] = pd.to_datetime(price["datetime"])
            price = price.set_index("datetime")
        else:
            raise ValueError(f"{path} must have a DatetimeIndex or datetime column.")

    keep = [col for col in ["open", "high", "low", "close", "volume"] if col in price.columns]
    if not {"open", "close"}.issubset(keep):
        raise ValueError(f"{path} must contain open and close columns.")
    price = price[keep].sort_index().copy()
    for col in keep:
        price[col] = pd.to_numeric(price[col], errors="coerce")
    return price


def attach_price(signals: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    merged = signals.join(price, how="left")
    missing_price = merged[["open", "close"]].isna().any(axis=1)
    if missing_price.any():
        missing_count = int(missing_price.sum())
        raise ValueError(f"{missing_count} signal rows could not be matched to price bars.")
    return merged


def run_inventory_t0_backtest_with_mode(
    signal_df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    config: dict,
    trade_direction_mode: str,
    price_col: str = "close",
    entry_price_col: str = "open",
    no_overnight: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = signal_df.sort_index().copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("signal_df index must be a DatetimeIndex")
    required = {"buy_prob", "sell_prob", price_col, entry_price_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if trade_direction_mode not in {"both", "sell_only", "buy_only"}:
        raise ValueError(f"unsupported trade_direction_mode: {trade_direction_mode}")

    allow_sell_first = trade_direction_mode in {"both", "sell_only"}
    allow_buy_first = trade_direction_mode in {"both", "buy_only"}

    initial_capital = float(config.get("initial_capital", 1_000_000))
    lot_size = int(config.get("min_lot_size", 100))
    base_position_pct = float(config.get("base_position_pct", 0.6))
    max_t0_trade_pct = float(config.get("max_t0_trade_pct", 0.3))
    max_participation_rate = float(config.get("max_participation_rate", 0.05))
    commission_rate = float(config.get("commission_rate", 0.001))
    slippage_rate = float(config.get("slippage_rate", 0.0005))
    tp = float(config.get("tp", 0.003))
    sl = float(config.get("sl", 0.002))
    min_position_pct = float(config.get("min_position_pct", 0.25))
    max_position_pct = float(config.get("max_position_pct", 1.0))
    high_quantile = float(config.get("position_sizing_high_quantile", 0.99))
    buy_high_threshold = float(data["buy_prob"].quantile(high_quantile))
    sell_high_threshold = float(data["sell_prob"].quantile(high_quantile))

    first_open = float(data[entry_price_col].iloc[0])
    if pd.isna(first_open) or first_open <= 0:
        raise ValueError("first open price must be positive")
    base_shares = round_lot_shares(initial_capital * base_position_pct / first_open, lot_size)
    base_cost = base_shares * first_open
    cash = initial_capital - base_cost
    shares = base_shares
    tradable_shares = base_shares
    active_leg = None
    trades = []
    equity_values = []
    benchmark_values = []
    alpha_values = []
    bar_returns = []
    alpha_returns = []
    prev_equity = initial_capital
    prev_alpha_equity = initial_capital
    current_day = None

    for i in range(len(data)):
        now = data.index[i]
        if current_day is None or now.normalize() != current_day:
            current_day = now.normalize()
            tradable_shares = shares

        close_price = float(data[price_col].iloc[i])
        equity = cash + shares * close_price
        benchmark_equity = initial_capital - base_cost + base_shares * close_price
        alpha_equity = initial_capital + (equity - benchmark_equity)
        equity_values.append((now, equity))
        benchmark_values.append((now, benchmark_equity))
        alpha_values.append((now, alpha_equity))
        bar_returns.append(equity / prev_equity - 1 if prev_equity > 0 else 0.0)
        alpha_returns.append(alpha_equity / prev_alpha_equity - 1 if prev_alpha_equity > 0 else 0.0)
        prev_equity = equity
        prev_alpha_equity = alpha_equity

        if i >= len(data) - 1:
            continue

        next_time = data.index[i + 1]
        next_open = float(data[entry_price_col].iloc[i + 1])
        if pd.isna(next_open) or next_open <= 0:
            continue
        same_day_next = now.normalize() == next_time.normalize()
        volume_cap = volume_cap_shares(data, i + 1, max_participation_rate, lot_size)

        def cap_qty(raw_qty: float) -> int:
            qty = round_lot_shares(raw_qty, lot_size)
            if volume_cap is not None:
                qty = min(qty, volume_cap)
            return int(qty)

        if active_leg is None:
            if no_overnight and not same_day_next:
                continue

            buy_prob = float(data["buy_prob"].iloc[i])
            sell_prob = float(data["sell_prob"].iloc[i])
            buy_position_pct = probability_scaled_position_pct(
                buy_prob,
                buy_threshold,
                buy_high_threshold,
                min_position_pct,
                max_position_pct,
            )
            sell_position_pct = probability_scaled_position_pct(
                sell_prob,
                sell_threshold,
                sell_high_threshold,
                min_position_pct,
                max_position_pct,
            )

            if (
                allow_sell_first
                and sell_prob >= sell_threshold
                and tradable_shares >= lot_size
                and sell_position_pct >= buy_position_pct
            ):
                qty_budget = initial_capital * max_t0_trade_pct * sell_position_pct / next_open
                qty = cap_qty(min(qty_budget, tradable_shares))
                if qty <= 0:
                    continue
                sell_price = next_open * (1 - slippage_rate)
                notional = qty * sell_price
                commission = notional * commission_rate
                cash += notional - commission
                shares -= qty
                tradable_shares -= qty
                active_leg = {
                    "type": "sell_first",
                    "entry_time": next_time,
                    "entry_price": sell_price,
                    "qty": qty,
                    "entry_prob": sell_prob,
                    "position_pct": sell_position_pct,
                }
                continue

            if allow_buy_first and buy_prob >= buy_threshold and cash > 0 and tradable_shares >= lot_size:
                qty_budget = initial_capital * max_t0_trade_pct * buy_position_pct / next_open
                cash_budget = cash / (next_open * (1 + slippage_rate) * (1 + commission_rate))
                qty = cap_qty(min(qty_budget, cash_budget, tradable_shares))
                if qty <= 0:
                    continue
                buy_price = next_open * (1 + slippage_rate)
                notional = qty * buy_price
                commission = notional * commission_rate
                cash -= notional + commission
                shares += qty
                active_leg = {
                    "type": "buy_first",
                    "entry_time": next_time,
                    "entry_price": buy_price,
                    "qty": qty,
                    "entry_prob": buy_prob,
                    "position_pct": buy_position_pct,
                }
                continue

        else:
            leg_type = active_leg["type"]
            qty = int(active_leg["qty"])
            entry_price = float(active_leg["entry_price"])
            exit_reason = None

            if leg_type == "sell_first":
                gross_return_if_exit = entry_price / (next_open * (1 + slippage_rate)) - 1
                if gross_return_if_exit >= tp:
                    exit_reason = "take_profit_buyback"
                elif gross_return_if_exit <= -sl:
                    exit_reason = "stop_loss_buyback"
                elif float(data["buy_prob"].iloc[i]) >= buy_threshold:
                    exit_reason = "buy_signal_buyback"
                elif no_overnight and not same_day_next:
                    exit_reason = "end_of_day_buyback"
                if exit_reason is None:
                    continue
                buy_price = next_open * (1 + slippage_rate)
                notional = qty * buy_price
                commission = notional * commission_rate
                if cash < notional + commission:
                    continue
                cash -= notional + commission
                shares += qty
                gross_return = entry_price / buy_price - 1
                net_return = gross_return - 2 * commission_rate - 2 * slippage_rate
                trades.append(
                    {
                        "leg_type": leg_type,
                        "entry_time": active_leg["entry_time"],
                        "exit_time": next_time,
                        "entry_price": entry_price,
                        "exit_price": buy_price,
                        "shares": qty,
                        "position_pct": float(active_leg["position_pct"]),
                        "entry_prob": float(active_leg["entry_prob"]),
                        "exit_prob": float(data["buy_prob"].iloc[i]),
                        "gross_return": float(gross_return),
                        "net_return": float(net_return),
                        "exit_reason": exit_reason,
                    }
                )
                active_leg = None
                continue

            gross_return_if_exit = (next_open * (1 - slippage_rate)) / entry_price - 1
            if gross_return_if_exit >= tp:
                exit_reason = "take_profit_sell"
            elif gross_return_if_exit <= -sl:
                exit_reason = "stop_loss_sell"
            elif float(data["sell_prob"].iloc[i]) >= sell_threshold:
                exit_reason = "sell_signal_sell"
            elif no_overnight and not same_day_next:
                exit_reason = "end_of_day_sell"
            if exit_reason is None:
                continue
            qty = min(qty, tradable_shares)
            qty = cap_qty(qty)
            if qty <= 0:
                continue
            sell_price = next_open * (1 - slippage_rate)
            notional = qty * sell_price
            commission = notional * commission_rate
            cash += notional - commission
            shares -= qty
            tradable_shares -= qty
            gross_return = sell_price / entry_price - 1
            net_return = gross_return - 2 * commission_rate - 2 * slippage_rate
            trades.append(
                {
                    "leg_type": leg_type,
                    "entry_time": active_leg["entry_time"],
                    "exit_time": next_time,
                    "entry_price": entry_price,
                    "exit_price": sell_price,
                    "shares": qty,
                    "position_pct": float(active_leg["position_pct"]),
                    "entry_prob": float(active_leg["entry_prob"]),
                    "exit_prob": float(data["sell_prob"].iloc[i]),
                    "gross_return": float(gross_return),
                    "net_return": float(net_return),
                    "exit_reason": exit_reason,
                }
            )
            active_leg = None

    equity = pd.Series(dict(equity_values), name="equity").sort_index()
    benchmark = pd.Series(dict(benchmark_values), name="benchmark_equity").sort_index()
    alpha_equity = pd.Series(dict(alpha_values), name="alpha_equity").sort_index()
    bar_returns = pd.Series(bar_returns, index=equity.index, name="bar_return")
    alpha_returns = pd.Series(alpha_returns, index=alpha_equity.index, name="alpha_bar_return")
    trades_df = pd.DataFrame(trades)

    stats = summarize_equity_curve(
        equity,
        bar_returns,
        trades_df,
        periods_per_year=int(config.get("periods_per_year", DEFAULT_PERIODS_PER_YEAR)),
        risk_free_rate=float(config.get("risk_free_rate", 0.0)),
    )
    alpha_stats = summarize_equity_curve(
        alpha_equity,
        alpha_returns,
        trades_df,
        periods_per_year=int(config.get("periods_per_year", DEFAULT_PERIODS_PER_YEAR)),
        risk_free_rate=float(config.get("risk_free_rate", 0.0)),
    )
    stats.update(
        {
            "backtest_mode": "a_share_inventory_t0",
            "trade_direction_mode": trade_direction_mode,
            "buy_threshold": float(buy_threshold),
            "sell_threshold": float(sell_threshold),
            "initial_capital": initial_capital,
            "final_equity": float(equity.iloc[-1]) if len(equity) else initial_capital,
            "final_benchmark_equity": float(benchmark.iloc[-1]) if len(benchmark) else initial_capital,
            "final_alpha_equity": float(alpha_equity.iloc[-1]) if len(alpha_equity) else initial_capital,
            "alpha_total_return": alpha_stats["total_return"],
            "alpha_sharpe": alpha_stats["sharpe"],
            "alpha_sortino": alpha_stats["sortino"],
            "alpha_calmar": alpha_stats["calmar"],
            "alpha_max_drawdown": alpha_stats["max_drawdown"],
            "base_shares": int(base_shares),
            "ending_shares": int(shares),
            "base_position_pct": base_position_pct,
            "max_t0_trade_pct": max_t0_trade_pct,
            "max_participation_rate": max_participation_rate,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
            "tp": tp,
            "sl": sl,
            "no_overnight": bool(no_overnight),
        }
    )
    equity_frame = pd.concat([equity, benchmark, alpha_equity], axis=1)
    return trades_df, equity_frame, stats


def export_backtest_result(output_dir: Path, trades: pd.DataFrame, equity: pd.DataFrame, stats: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / "t0_trades.csv", index=False)
    equity.to_csv(output_dir / "t0_equity_curve.csv")
    with open(output_dir / "t0_backtest_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=json_default)


def threshold_from_valid(valid_signals: pd.DataFrame, prob_col: str, quantile: float | None) -> float:
    if quantile is None:
        quantile = 0.95
    return float(valid_signals[prob_col].quantile(float(quantile)))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    valid_path, test_path = locate_signal_pair()
    valid_signals = read_signals(valid_path)
    test_signals = read_signals(test_path)
    price = load_price_data()
    valid_data = attach_price(valid_signals, price)
    test_data = attach_price(test_signals, price)

    rows = []
    for rule in RULES:
        rule_name = rule["rule_name"]
        mode = rule["trade_direction_mode"]
        buy_quantile = rule["buy_quantile"]
        sell_quantile = rule["sell_quantile"]
        buy_threshold = threshold_from_valid(valid_data, "buy_prob", buy_quantile)
        sell_threshold = threshold_from_valid(valid_data, "sell_prob", sell_quantile)

        trades, equity, stats = run_inventory_t0_backtest_with_mode(
            test_data,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            config=BACKTEST_CONFIG,
            trade_direction_mode=mode,
        )
        stats.update(
            {
                "rule_name": rule_name,
                "valid_signals_path": str(valid_path),
                "test_signals_path": str(test_path),
                "buy_quantile": np.nan if buy_quantile is None else float(buy_quantile),
                "sell_quantile": np.nan if sell_quantile is None else float(sell_quantile),
                "buy_threshold": float(buy_threshold),
                "sell_threshold": float(sell_threshold),
                "buy_open_enabled": bool(mode in {"both", "buy_only"}),
                "sell_open_enabled": bool(mode in {"both", "sell_only"}),
            }
        )
        rule_dir = OUTPUT_DIR / rule_name
        export_backtest_result(rule_dir, trades, equity, stats)
        rows.append(
            {
                "rule_name": rule_name,
                "total_return": stats.get("total_return"),
                "alpha_total_return": stats.get("alpha_total_return"),
                "alpha_max_drawdown": stats.get("alpha_max_drawdown"),
                "alpha_sharpe": stats.get("alpha_sharpe"),
                "alpha_sortino": stats.get("alpha_sortino"),
                "alpha_calmar": stats.get("alpha_calmar"),
                "trade_count": stats.get("trade_count"),
                "win_rate": stats.get("win_rate"),
                "avg_net_return_per_trade": stats.get("avg_net_return_per_trade"),
                "median_net_return_per_trade": stats.get("median_net_return_per_trade"),
                "buy_threshold": stats.get("buy_threshold"),
                "sell_threshold": stats.get("sell_threshold"),
                "buy_quantile": stats.get("buy_quantile"),
                "sell_quantile": stats.get("sell_quantile"),
                "trade_direction_mode": mode,
                "output_dir": str(rule_dir),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["alpha_total_return", "alpha_sharpe"], ascending=[False, False])
    summary_path = OUTPUT_DIR / "rule_compare_summary.csv"
    json_path = OUTPUT_DIR / "rule_compare_summary.json"
    summary.to_csv(summary_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=json_default)

    display_cols = [
        "rule_name",
        "trade_direction_mode",
        "alpha_total_return",
        "alpha_max_drawdown",
        "alpha_sharpe",
        "trade_count",
        "win_rate",
        "buy_threshold",
        "sell_threshold",
    ]
    printable = summary[display_cols].copy()
    for col in [
        "alpha_total_return",
        "alpha_max_drawdown",
        "alpha_sharpe",
        "win_rate",
        "buy_threshold",
        "sell_threshold",
    ]:
        printable[col] = printable[col].astype(float).round(6)
    print(printable.to_string(index=False))


if __name__ == "__main__":
    main()

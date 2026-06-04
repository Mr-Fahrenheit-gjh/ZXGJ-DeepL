from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PERIODS_PER_YEAR = 48 * 242


def calc_max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return np.nan
    equity = pd.Series(equity).astype(float)
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def calc_sortino(returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR, risk_free_rate: float = 0.0) -> float:
    returns = pd.Series(returns).dropna().astype(float)
    if len(returns) == 0:
        return np.nan
    per_period_rf = risk_free_rate / periods_per_year
    excess = returns - per_period_rf
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or pd.isna(downside_std):
        return np.nan
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def calc_sharpe(returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR, risk_free_rate: float = 0.0) -> float:
    returns = pd.Series(returns).dropna().astype(float)
    if len(returns) == 0:
        return np.nan
    per_period_rf = risk_free_rate / periods_per_year
    excess = returns - per_period_rf
    std = excess.std(ddof=1)
    if std == 0 or pd.isna(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def calc_annualized_return(total_return: float, start_time, end_time) -> float:
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    years = (end_time - start_time).total_seconds() / (365.25 * 24 * 60 * 60)
    if years <= 0:
        return np.nan
    return float((1 + total_return) ** (1 / years) - 1)


def summarize_equity_curve(
    equity: pd.Series,
    bar_returns: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> dict:
    if len(equity) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "trade_count": 0,
            "win_rate": np.nan,
        }

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    annualized_return = calc_annualized_return(total_return, equity.index.min(), equity.index.max())
    max_drawdown = calc_max_drawdown(equity)
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    trade_count = int(len(trades))
    win_rate = float((trades["net_return"] > 0).mean()) if trade_count else np.nan
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe": calc_sharpe(bar_returns, periods_per_year, risk_free_rate),
        "sortino": calc_sortino(bar_returns, periods_per_year, risk_free_rate),
        "calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "avg_net_return_per_trade": float(trades["net_return"].mean()) if trade_count else np.nan,
        "median_net_return_per_trade": float(trades["net_return"].median()) if trade_count else np.nan,
    }


def probability_scaled_position_pct(
    prob: float,
    threshold: float,
    high_threshold: float,
    min_position_pct: float,
    max_position_pct: float,
) -> float:
    if pd.isna(prob) or prob < threshold:
        return 0.0
    if high_threshold <= threshold:
        return float(max_position_pct)
    scale = (float(prob) - float(threshold)) / (float(high_threshold) - float(threshold))
    scale = min(max(scale, 0.0), 1.0)
    return float(min_position_pct + scale * (max_position_pct - min_position_pct))


def run_t0_dual_signal_backtest(
    signal_df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    config: dict,
    price_col: str = "close",
    entry_price_col: str = "open",
    no_overnight: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Long-only A-share T+0 style state machine driven by buy/sell probabilities.

    This is still a research backtest, but it is intentionally event based:
    signal at bar t, execute at t+1 open, one position at a time, round lots,
    commission + slippage, tp/sl, sell signal, and optional same-day flat.
    """
    data = signal_df.sort_index().copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("signal_df index must be a DatetimeIndex")
    required = {"buy_prob", "sell_prob", price_col, entry_price_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    initial_capital = float(config.get("initial_capital", 1_000_000))
    cash = initial_capital
    shares = 0
    entry_price = np.nan
    entry_time = None
    entry_position_pct = 0.0
    entry_buy_prob = np.nan
    min_lot_size = int(config.get("min_lot_size", 100))
    max_position_pct = float(config.get("max_position_pct", 1.0))
    min_position_pct = float(config.get("min_position_pct", max_position_pct))
    position_sizing_mode = config.get("position_sizing_mode", "probability_scaled")
    high_quantile = float(config.get("position_sizing_high_quantile", 0.99))
    buy_high_threshold = float(data["buy_prob"].quantile(high_quantile))
    commission_rate = float(config.get("commission_rate", 0.001))
    slippage_rate = float(config.get("slippage_rate", 0.0005))
    tp = float(config.get("tp", 0.003))
    sl = float(config.get("sl", 0.002))

    trades = []
    equity_values = []
    bar_returns = []
    prev_equity = initial_capital
    n = len(data)

    for i in range(n):
        now = data.index[i]
        close_price = float(data[price_col].iloc[i])
        equity = cash + shares * close_price
        equity_values.append((now, equity))
        bar_returns.append(equity / prev_equity - 1 if prev_equity > 0 else 0.0)
        prev_equity = equity

        if i >= n - 1:
            continue

        next_time = data.index[i + 1]
        next_open = float(data[entry_price_col].iloc[i + 1])
        if pd.isna(next_open) or next_open <= 0:
            continue
        same_day_next = now.normalize() == next_time.normalize()

        if shares == 0:
            if no_overnight and not same_day_next:
                continue
            buy_prob = float(data["buy_prob"].iloc[i])
            if buy_prob < buy_threshold:
                continue
            if position_sizing_mode == "probability_scaled":
                position_pct = probability_scaled_position_pct(
                    buy_prob,
                    buy_threshold,
                    buy_high_threshold,
                    min_position_pct,
                    max_position_pct,
                )
            else:
                position_pct = max_position_pct
            if position_pct <= 0:
                continue
            gross_budget = equity * position_pct
            executable_price = next_open * (1 + slippage_rate)
            lot_count = int(gross_budget // (executable_price * min_lot_size))
            buy_shares = lot_count * min_lot_size
            if buy_shares <= 0:
                continue
            notional = buy_shares * executable_price
            commission = notional * commission_rate
            if notional + commission > cash:
                continue
            cash -= notional + commission
            shares = buy_shares
            entry_price = executable_price
            entry_time = next_time
            entry_position_pct = position_pct
            entry_buy_prob = buy_prob
            continue

        unrealized = next_open / entry_price - 1
        exit_reason = None
        if unrealized >= tp:
            exit_reason = "take_profit"
        elif unrealized <= -sl:
            exit_reason = "stop_loss"
        elif float(data["sell_prob"].iloc[i]) >= sell_threshold:
            exit_reason = "sell_signal"
        elif no_overnight and not same_day_next:
            exit_reason = "end_of_day_flat"

        if exit_reason is None:
            continue

        executable_price = next_open * (1 - slippage_rate)
        notional = shares * executable_price
        commission = notional * commission_rate
        cash += notional - commission
        gross_return = executable_price / entry_price - 1
        net_return = gross_return - 2 * commission_rate - 2 * slippage_rate
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": next_time,
                "entry_price": float(entry_price),
                "exit_price": float(executable_price),
                "shares": int(shares),
                "position_pct": float(entry_position_pct),
                "entry_buy_prob": float(entry_buy_prob),
                "exit_sell_prob": float(data["sell_prob"].iloc[i]),
                "gross_return": float(gross_return),
                "net_return": float(net_return),
                "exit_reason": exit_reason,
                "entry_buy_threshold": float(buy_threshold),
                "exit_sell_threshold": float(sell_threshold),
            }
        )
        shares = 0
        entry_price = np.nan
        entry_time = None
        entry_position_pct = 0.0
        entry_buy_prob = np.nan

    if shares > 0:
        final_time = data.index[-1]
        final_price = float(data[price_col].iloc[-1]) * (1 - slippage_rate)
        notional = shares * final_price
        commission = notional * commission_rate
        cash += notional - commission
        gross_return = final_price / entry_price - 1
        net_return = gross_return - 2 * commission_rate - 2 * slippage_rate
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": final_time,
                "entry_price": float(entry_price),
                "exit_price": float(final_price),
                "shares": int(shares),
                "position_pct": float(entry_position_pct),
                "entry_buy_prob": float(entry_buy_prob),
                "exit_sell_prob": np.nan,
                "gross_return": float(gross_return),
                "net_return": float(net_return),
                "exit_reason": "final_flat",
                "entry_buy_threshold": float(buy_threshold),
                "exit_sell_threshold": float(sell_threshold),
            }
        )
        if equity_values:
            equity_values[-1] = (final_time, cash)

    equity = pd.Series(dict(equity_values), name="equity").sort_index()
    bar_returns = pd.Series(bar_returns, index=equity.index, name="bar_return")
    trades_df = pd.DataFrame(trades)
    stats = summarize_equity_curve(
        equity,
        bar_returns,
        trades_df,
        periods_per_year=int(config.get("periods_per_year", DEFAULT_PERIODS_PER_YEAR)),
        risk_free_rate=float(config.get("risk_free_rate", 0.0)),
    )
    stats.update(
        {
            "buy_threshold": float(buy_threshold),
            "sell_threshold": float(sell_threshold),
            "initial_capital": initial_capital,
            "final_equity": float(equity.iloc[-1]) if len(equity) else initial_capital,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
            "tp": tp,
            "sl": sl,
            "no_overnight": bool(no_overnight),
            "position_sizing_mode": position_sizing_mode,
            "min_position_pct": min_position_pct,
            "max_position_pct": max_position_pct,
            "position_sizing_high_quantile": high_quantile,
            "buy_high_threshold": buy_high_threshold,
        }
    )
    return trades_df, equity, stats


def _round_lot_shares(raw_shares: float, lot_size: int) -> int:
    return int(max(raw_shares, 0) // lot_size * lot_size)


def _volume_cap_shares(data: pd.DataFrame, row_pos: int, max_participation_rate: float, lot_size: int) -> int | None:
    if "volume" not in data.columns or pd.isna(data["volume"].iloc[row_pos]):
        return None
    return _round_lot_shares(float(data["volume"].iloc[row_pos]) * max_participation_rate, lot_size)


def run_a_share_inventory_t0_backtest(
    signal_df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    config: dict,
    price_col: str = "close",
    entry_price_col: str = "open",
    no_overnight: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A-share inventory-based T+0 research backtest.

    A-share cash equities are T+1 for newly bought shares. This simulator starts
    with a base inventory and only sells inventory that was already tradable at
    the start of the day. It supports:
    - sell_first: sell existing inventory, then buy back later;
    - buy_first: buy extra shares, then sell the same quantity from old inventory.
    """
    data = signal_df.sort_index().copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("signal_df index must be a DatetimeIndex")
    required = {"buy_prob", "sell_prob", price_col, entry_price_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

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
    trade_direction_mode = str(config.get("trade_direction_mode", "both"))
    if trade_direction_mode not in {"both", "sell_only", "buy_only"}:
        raise ValueError(f"unsupported trade_direction_mode: {trade_direction_mode}")
    allow_sell_first = trade_direction_mode in {"both", "sell_only"}
    allow_buy_first = trade_direction_mode in {"both", "buy_only"}

    first_open = float(data[entry_price_col].iloc[0])
    if pd.isna(first_open) or first_open <= 0:
        raise ValueError("first open price must be positive")
    base_shares = _round_lot_shares(initial_capital * base_position_pct / first_open, lot_size)
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

    n = len(data)
    for i in range(n):
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

        if i >= n - 1:
            continue

        next_time = data.index[i + 1]
        next_open = float(data[entry_price_col].iloc[i + 1])
        if pd.isna(next_open) or next_open <= 0:
            continue
        same_day_next = now.normalize() == next_time.normalize()
        volume_cap = _volume_cap_shares(data, i + 1, max_participation_rate, lot_size)

        def cap_qty(raw_qty):
            qty = _round_lot_shares(raw_qty, lot_size)
            if volume_cap is not None:
                qty = min(qty, volume_cap)
            return int(qty)

        has_same_day_exit_bar = i + 2 < n and data.index[i + 2].normalize() == next_time.normalize()

        if active_leg is None:
            if no_overnight and (not same_day_next or not has_same_day_exit_bar):
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

            if no_overnight and not same_day_next:
                if leg_type == "sell_first":
                    buy_price = close_price * (1 + slippage_rate)
                    notional = qty * buy_price
                    commission = notional * commission_rate
                    if cash >= notional + commission:
                        cash -= notional + commission
                        shares += qty
                        gross_return = entry_price / buy_price - 1
                        net_return = gross_return - 2 * commission_rate - 2 * slippage_rate
                        trades.append(
                            {
                                "leg_type": leg_type,
                                "entry_time": active_leg["entry_time"],
                                "exit_time": now,
                                "entry_price": entry_price,
                                "exit_price": buy_price,
                                "shares": qty,
                                "position_pct": float(active_leg["position_pct"]),
                                "entry_prob": float(active_leg["entry_prob"]),
                                "exit_prob": float(data["buy_prob"].iloc[i]),
                                "gross_return": float(gross_return),
                                "net_return": float(net_return),
                                "exit_reason": "end_of_day_buyback",
                            }
                        )
                        active_leg = None
                else:
                    qty = min(qty, tradable_shares)
                    qty = cap_qty(qty)
                    if qty > 0:
                        sell_price = close_price * (1 - slippage_rate)
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
                                "exit_time": now,
                                "entry_price": entry_price,
                                "exit_price": sell_price,
                                "shares": qty,
                                "position_pct": float(active_leg["position_pct"]),
                                "entry_prob": float(active_leg["entry_prob"]),
                                "exit_prob": float(data["sell_prob"].iloc[i]),
                                "gross_return": float(gross_return),
                                "net_return": float(net_return),
                                "exit_reason": "end_of_day_sell",
                            }
                        )
                        active_leg = None
                continue

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

    if active_leg is not None and len(data):
        last_time = data.index[-1]
        last_close = float(data[price_col].iloc[-1])
        leg_type = active_leg["type"]
        qty = int(active_leg["qty"])
        entry_price = float(active_leg["entry_price"])
        if leg_type == "sell_first":
            buy_price = last_close * (1 + slippage_rate)
            notional = qty * buy_price
            commission = notional * commission_rate
            if cash >= notional + commission:
                cash -= notional + commission
                shares += qty
                gross_return = entry_price / buy_price - 1
                net_return = gross_return - 2 * commission_rate - 2 * slippage_rate
                trades.append(
                    {
                        "leg_type": leg_type,
                        "entry_time": active_leg["entry_time"],
                        "exit_time": last_time,
                        "entry_price": entry_price,
                        "exit_price": buy_price,
                        "shares": qty,
                        "position_pct": float(active_leg["position_pct"]),
                        "entry_prob": float(active_leg["entry_prob"]),
                        "exit_prob": float(data["buy_prob"].iloc[-1]),
                        "gross_return": float(gross_return),
                        "net_return": float(net_return),
                        "exit_reason": "final_buyback",
                    }
                )
                active_leg = None
        else:
            qty = min(qty, tradable_shares)
            qty = _round_lot_shares(qty, lot_size)
            if qty > 0:
                sell_price = last_close * (1 - slippage_rate)
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
                        "exit_time": last_time,
                        "entry_price": entry_price,
                        "exit_price": sell_price,
                        "shares": qty,
                        "position_pct": float(active_leg["position_pct"]),
                        "entry_prob": float(active_leg["entry_prob"]),
                        "exit_prob": float(data["sell_prob"].iloc[-1]),
                        "gross_return": float(gross_return),
                        "net_return": float(net_return),
                        "exit_reason": "final_sell",
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


def export_t0_backtest_result(
    output_dir: str | Path,
    trades: pd.DataFrame,
    equity: pd.Series,
    stats: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / "t0_trades.csv", index=False)
    equity.to_csv(output_dir / "t0_equity_curve.csv")
    with open(output_dir / "t0_backtest_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def run_inventory_t0_stress_grid(
    signal_df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    config: dict,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Stress the inventory T+0 simulator under higher costs and tighter liquidity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_commission = float(config.get("commission_rate", 0.001))
    base_slippage = float(config.get("slippage_rate", 0.0005))
    commission_multipliers = config.get("stress_commission_multipliers", [1.0, 1.5])
    slippage_multipliers = config.get("stress_slippage_multipliers", [1.0, 1.5, 2.0])
    participation_rates = config.get("stress_participation_rates", [config.get("max_participation_rate", 0.05)])

    rows = []
    for commission_mult in commission_multipliers:
        for slippage_mult in slippage_multipliers:
            for participation_rate in participation_rates:
                scenario_config = dict(config)
                scenario_config["commission_rate"] = base_commission * float(commission_mult)
                scenario_config["slippage_rate"] = base_slippage * float(slippage_mult)
                scenario_config["max_participation_rate"] = float(participation_rate)
                trades, equity, stats = run_a_share_inventory_t0_backtest(
                    signal_df,
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                    config=scenario_config,
                )
                scenario_name = (
                    f"comm{commission_mult:g}_slip{slippage_mult:g}_part{float(participation_rate):g}"
                    .replace(".", "p")
                )
                scenario_dir = output_dir / scenario_name
                export_t0_backtest_result(scenario_dir, trades, equity, stats)
                rows.append(
                    {
                        "scenario": scenario_name,
                        "commission_multiplier": float(commission_mult),
                        "slippage_multiplier": float(slippage_mult),
                        "max_participation_rate": float(participation_rate),
                        **stats,
                        "output_dir": str(scenario_dir),
                    }
                )

    stress_df = pd.DataFrame(rows)
    stress_df.to_csv(output_dir / "stress_test_summary.csv", index=False)
    aggregate = {
        "scenario_count": int(len(stress_df)),
        "all_alpha_positive": bool((stress_df["alpha_total_return"] > 0).all()) if "alpha_total_return" in stress_df and len(stress_df) else False,
        "min_alpha_total_return": float(stress_df["alpha_total_return"].min()) if "alpha_total_return" in stress_df and len(stress_df) else np.nan,
        "worst_alpha_max_drawdown": float(stress_df["alpha_max_drawdown"].min()) if "alpha_max_drawdown" in stress_df and len(stress_df) else np.nan,
        "min_trade_count": int(stress_df["trade_count"].min()) if "trade_count" in stress_df and len(stress_df) else 0,
    }
    with open(output_dir / "stress_test_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    return stress_df


def run_sell_only_threshold_grid(
    valid_signals: pd.DataFrame,
    test_signals: pd.DataFrame,
    config: dict,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Evaluate sell-only inventory T+0 over sell probability quantiles.

    Thresholds are selected from valid_signals and evaluated on test_signals to
    avoid using test outcomes or test probability distribution for selection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sell_quantiles = config.get("sell_threshold_grid_quantiles", [0.90, 0.92, 0.94, 0.95, 0.96, 0.97])
    buy_quantile = float(config.get("buy_threshold_quantile", config.get("fixed_threshold_quantile", 0.95)))
    buy_threshold = float(valid_signals["buy_prob"].quantile(buy_quantile))
    rows = []
    for sell_quantile in sell_quantiles:
        sell_quantile = float(sell_quantile)
        sell_threshold = float(valid_signals["sell_prob"].quantile(sell_quantile))
        scenario_config = dict(config)
        scenario_config["trade_direction_mode"] = "sell_only"
        trades, equity, stats = run_a_share_inventory_t0_backtest(
            test_signals,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            config=scenario_config,
        )
        scenario_name = f"sell_only_q{int(round(sell_quantile * 100)):02d}"
        scenario_dir = output_dir / scenario_name
        export_t0_backtest_result(scenario_dir, trades, equity, stats)
        row = {
            "rule_name": scenario_name,
            "trade_direction_mode": "sell_only",
            "buy_quantile": buy_quantile,
            "sell_quantile": sell_quantile,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            **stats,
            "output_dir": str(scenario_dir),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    if len(result):
        sort_cols = [c for c in ["alpha_total_return", "alpha_sharpe"] if c in result.columns]
        if sort_cols:
            result = result.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    result.to_csv(output_dir / "sell_only_threshold_grid.csv", index=False)
    with open(output_dir / "sell_only_threshold_grid.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=str)
    return result

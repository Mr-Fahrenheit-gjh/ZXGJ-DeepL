from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd


def check_vnpy_available() -> tuple[bool, dict]:
    info = {}
    try:
        import vnpy_ctastrategy  # noqa: F401
        info["vnpy_ctastrategy"] = True
    except Exception as exc:
        info["vnpy_ctastrategy"] = False
        info["vnpy_ctastrategy_error"] = repr(exc)

    try:
        from vnpy.trader.constant import Exchange, Interval
        from vnpy.trader.database import get_database
        from vnpy.trader.object import BarData

        info.update({
            "core": True,
            "Exchange": Exchange,
            "Interval": Interval,
            "get_database": get_database,
            "BarData": BarData,
        })
    except Exception as exc:
        info["core"] = False
        info["core_error"] = repr(exc)

    return bool(info.get("vnpy_ctastrategy") and info.get("core")), info


def save_df_to_vnpy_database(
    df: pd.DataFrame,
    symbol: str = "688981",
    exchange=None,
    gateway_name: str = "LOCAL_5MIN",
    vnpy_info: dict | None = None,
) -> int:
    if vnpy_info is None:
        available, vnpy_info = check_vnpy_available()
    else:
        available = bool(vnpy_info.get("core"))
    if not available:
        print("vn.py unavailable, skip saving bars")
        return 0

    Exchange = vnpy_info["Exchange"]
    Interval = vnpy_info["Interval"]
    BarData = vnpy_info["BarData"]
    get_database = vnpy_info["get_database"]
    exchange = exchange or Exchange.SSE

    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        if "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"])
            data = data.set_index("datetime")
        else:
            raise ValueError("df must have a DatetimeIndex or a datetime column")

    num_cols = ["open", "high", "low", "close", "volume", "amount"]
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close", "volume"]).sort_index()
    data["amount"] = data["amount"].fillna(0.0)

    bars = []
    for dt, row in data.iterrows():
        bars.append(
            BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt.to_pydatetime(),
                interval=Interval.MINUTE,
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=float(row["volume"]),
                turnover=float(row.get("amount", 0.0)),
                open_interest=0,
                gateway_name=gateway_name,
            )
        )

    database = get_database()
    database.save_bar_data(bars)
    return len(bars)


def export_signal_file(backtest_result: pd.DataFrame, signal_path: str | Path, prob_col: str = "pred_prob") -> tuple[Path, pd.DataFrame]:
    signal_df = backtest_result.copy()
    if not isinstance(signal_df.index, pd.DatetimeIndex):
        if "datetime" in signal_df.columns:
            signal_df["datetime"] = pd.to_datetime(signal_df["datetime"])
        else:
            raise ValueError("backtest_result must have a DatetimeIndex or a datetime column")
    else:
        signal_df["datetime"] = signal_df.index

    keep_cols = ["datetime", prob_col]
    optional = [c for c in ["pred_prob_buy", "pred_prob_sell", "pred_prob_hold"] if c in signal_df.columns and c not in keep_cols]
    signal_df = signal_df[keep_cols + optional].dropna(subset=[prob_col]).copy()
    signal_df["datetime"] = pd.to_datetime(signal_df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    signal_df = signal_df.rename(columns={prob_col: "pred_prob"})

    signal_path = Path(signal_path)
    signal_df.to_csv(signal_path, index=False)
    return signal_path, signal_df


def export_dual_signal_file(
    signal_df: pd.DataFrame,
    signal_path: str | Path,
    buy_prob_col: str = "buy_prob",
    sell_prob_col: str = "sell_prob",
) -> tuple[Path, pd.DataFrame]:
    data = signal_df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        if "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"])
        else:
            raise ValueError("signal_df must have a DatetimeIndex or a datetime column")
    else:
        data["datetime"] = data.index
    required = ["datetime", buy_prob_col, sell_prob_col]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"missing columns for dual signal export: {missing}")
    out = data[required].dropna(subset=[buy_prob_col, sell_prob_col]).copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out.rename(columns={buy_prob_col: "buy_prob", sell_prob_col: "sell_prob"})
    signal_path = Path(signal_path)
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(signal_path, index=False)
    return signal_path, out


def build_ml_signal_long_only_strategy(vnpy_info: dict):
    from vnpy_ctastrategy import CtaTemplate

    BarData = vnpy_info["BarData"]

    class MLSignalLongOnlyStrategy(CtaTemplate):
        author = "mvp"

        signal_file = "mvp_pred_signal.csv"
        threshold = 0.95
        fixed_size = 100
        exit_at_close = True
        stop_loss_pct = 0.02

        parameters = ["signal_file", "threshold", "fixed_size", "exit_at_close", "stop_loss_pct"]
        variables = ["entry_price"]

        def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
            super().__init__(cta_engine, strategy_name, vt_symbol, setting)
            self.signal_map = {}
            self.entry_price = 0.0

        def on_init(self):
            signal = pd.read_csv(self.signal_file)
            signal["datetime"] = pd.to_datetime(signal["datetime"])
            self.signal_map = dict(zip(signal["datetime"].dt.to_pydatetime(), signal["pred_prob"]))
            self.write_log(f"Loaded {len(self.signal_map)} ML buy-probability signals")

        def on_start(self):
            self.write_log("Strategy started")

        def on_stop(self):
            self.write_log("Strategy stopped")

        def on_bar(self, bar: BarData):
            self.cancel_all()
            if self.exit_at_close and bar.datetime.time() >= time(15, 0):
                if self.pos > 0:
                    self.sell(bar.close_price * 0.99, abs(self.pos))
                return

            if self.pos > 0 and self.entry_price > 0:
                if bar.close_price / self.entry_price - 1 <= -self.stop_loss_pct:
                    self.sell(bar.close_price * 0.99, abs(self.pos))
                    return

            prob = self.signal_map.get(bar.datetime.replace(tzinfo=None))
            if prob is None:
                return
            if prob >= self.threshold and self.pos <= 0:
                self.buy(bar.close_price * 1.01, self.fixed_size)
                self.entry_price = bar.close_price
            elif prob < self.threshold and self.pos > 0:
                self.sell(bar.close_price * 0.99, abs(self.pos))
                self.entry_price = 0.0

    return MLSignalLongOnlyStrategy


def build_dual_signal_inventory_t0_strategy(vnpy_info: dict):
    from vnpy_ctastrategy import CtaTemplate

    BarData = vnpy_info["BarData"]

    class DualSignalInventoryT0Strategy(CtaTemplate):
        """Event-driven approximation of the research dual-signal inventory T+0 logic.

        vn.py CTA backtesting is not an A-share exchange simulator, so the
        strategy explicitly maintains a base inventory target and only uses
        sell-first/buy-first legs sized by fixed lots. It is intended as an
        event-driven cross-check of signal timing and order flow, not a
        replacement for broker-side A-share settlement controls.
        """

        author = "mvp"

        signal_file = "dual_signal.csv"
        buy_threshold = 0.95
        sell_threshold = 0.95
        fixed_size = 100
        base_size = 100
        stop_loss_pct = 0.002
        take_profit_pct = 0.003
        restore_at_close = True

        parameters = [
            "signal_file",
            "buy_threshold",
            "sell_threshold",
            "fixed_size",
            "base_size",
            "stop_loss_pct",
            "take_profit_pct",
            "restore_at_close",
        ]
        variables = ["active_leg", "leg_entry_price"]

        def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
            super().__init__(cta_engine, strategy_name, vt_symbol, setting)
            self.signal_map = {}
            self.active_leg = ""
            self.leg_entry_price = 0.0

        def on_init(self):
            signal = pd.read_csv(self.signal_file)
            signal["datetime"] = pd.to_datetime(signal["datetime"])
            self.signal_map = {
                row.datetime.to_pydatetime(): (float(row.buy_prob), float(row.sell_prob))
                for row in signal.itertuples(index=False)
            }
            self.write_log(f"Loaded {len(self.signal_map)} dual ML signals")

        def on_start(self):
            if self.pos < self.base_size:
                self.buy(1e9, self.base_size - self.pos)
            self.write_log("Dual signal inventory T+0 strategy started")

        def on_stop(self):
            self.write_log("Dual signal inventory T+0 strategy stopped")

        def _restore_base(self, bar: BarData):
            if self.pos > self.base_size:
                self.sell(bar.close_price * 0.99, self.pos - self.base_size)
            elif self.pos < self.base_size:
                self.buy(bar.close_price * 1.01, self.base_size - self.pos)
            self.active_leg = ""
            self.leg_entry_price = 0.0

        def on_bar(self, bar: BarData):
            self.cancel_all()
            if self.restore_at_close and bar.datetime.time() >= time(14, 55):
                self._restore_base(bar)
                return

            signal = self.signal_map.get(bar.datetime.replace(tzinfo=None))
            buy_prob, sell_prob = signal if signal is not None else (0.0, 0.0)

            if self.active_leg == "buy_first":
                gross = bar.close_price / self.leg_entry_price - 1 if self.leg_entry_price > 0 else 0
                if gross >= self.take_profit_pct or gross <= -self.stop_loss_pct or sell_prob >= self.sell_threshold:
                    qty = min(self.fixed_size, max(self.pos - self.base_size, 0))
                    if qty > 0:
                        self.sell(bar.close_price * 0.99, qty)
                    self.active_leg = ""
                    self.leg_entry_price = 0.0
                return

            if self.active_leg == "sell_first":
                gross = self.leg_entry_price / bar.close_price - 1 if bar.close_price > 0 else 0
                if gross >= self.take_profit_pct or gross <= -self.stop_loss_pct or buy_prob >= self.buy_threshold:
                    if self.pos < self.base_size:
                        self.buy(bar.close_price * 1.01, min(self.fixed_size, self.base_size - self.pos))
                    self.active_leg = ""
                    self.leg_entry_price = 0.0
                return

            if sell_prob >= self.sell_threshold and self.pos >= self.base_size:
                self.sell(bar.close_price * 0.99, self.fixed_size)
                self.active_leg = "sell_first"
                self.leg_entry_price = bar.close_price
                return

            if buy_prob >= self.buy_threshold and self.pos <= self.base_size:
                self.buy(bar.close_price * 1.01, self.fixed_size)
                self.active_leg = "buy_first"
                self.leg_entry_price = bar.close_price

    return DualSignalInventoryT0Strategy


def run_vnpy_backtest(
    vt_symbol: str,
    signal_path: str | Path,
    backtest_result: pd.DataFrame,
    threshold: float,
    vnpy_info: dict,
    fixed_size: int = 100,
    capital: float = 1_000_000,
) -> tuple[object | None, pd.DataFrame, dict]:
    if not (vnpy_info.get("vnpy_ctastrategy") and vnpy_info.get("core")):
        print("vn.py unavailable, backtest skipped")
        return None, pd.DataFrame(), {}

    from vnpy_ctastrategy.backtesting import BacktestingEngine

    Interval = vnpy_info["Interval"]
    Strategy = build_ml_signal_long_only_strategy(vnpy_info)

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=Interval.MINUTE,
        start=backtest_result.index.min().to_pydatetime(),
        end=backtest_result.index.max().to_pydatetime(),
        rate=0.001,
        slippage=0.05,
        size=1,
        pricetick=0.01,
        capital=capital,
    )
    engine.add_strategy(
        Strategy,
        {
            "signal_file": str(signal_path),
            "threshold": threshold,
            "fixed_size": fixed_size,
            "exit_at_close": True,
            "stop_loss_pct": 0.02,
        },
    )
    engine.load_data()
    engine.run_backtesting()
    daily_result = engine.calculate_result()
    stats = engine.calculate_statistics(output=False)
    return engine, daily_result, stats


def run_vnpy_dual_signal_t0_backtest(
    vt_symbol: str,
    signal_path: str | Path,
    signal_df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    vnpy_info: dict,
    fixed_size: int = 100,
    base_size: int | None = None,
    capital: float = 1_000_000,
    commission_rate: float = 0.001,
    slippage_abs: float = 0.05,
    stop_loss_pct: float = 0.002,
    take_profit_pct: float = 0.003,
) -> tuple[object | None, pd.DataFrame, dict]:
    if not (vnpy_info.get("vnpy_ctastrategy") and vnpy_info.get("core")):
        print("vn.py unavailable, backtest skipped")
        return None, pd.DataFrame(), {}

    from vnpy_ctastrategy.backtesting import BacktestingEngine

    Interval = vnpy_info["Interval"]
    Strategy = build_dual_signal_inventory_t0_strategy(vnpy_info)
    data = signal_df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    base_size = int(base_size or fixed_size)

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=Interval.MINUTE,
        start=data.index.min().to_pydatetime(),
        end=data.index.max().to_pydatetime(),
        rate=float(commission_rate),
        slippage=float(slippage_abs),
        size=1,
        pricetick=0.01,
        capital=float(capital),
    )
    engine.add_strategy(
        Strategy,
        {
            "signal_file": str(signal_path),
            "buy_threshold": float(buy_threshold),
            "sell_threshold": float(sell_threshold),
            "fixed_size": int(fixed_size),
            "base_size": int(base_size),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "restore_at_close": True,
        },
    )
    engine.load_data()
    engine.run_backtesting()
    daily_result = engine.calculate_result()
    stats = engine.calculate_statistics(output=False)
    return engine, daily_result, stats

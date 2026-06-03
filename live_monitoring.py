from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def _safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """Compute PSI using expected-sample quantile bins.

    PSI is a coarse drift alarm. Around 0.1 is often treated as noticeable,
    around 0.2 as material enough to investigate before trusting live signals.
    """
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(expected) < bins * 2 or len(actual) < bins:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(expected.quantile(quantiles).to_numpy())
    if len(edges) <= 2:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_pct = pd.cut(expected, edges, include_lowest=True).value_counts(sort=False) / len(expected)
    actual_pct = pd.cut(actual, edges, include_lowest=True).value_counts(sort=False) / len(actual)
    expected_pct = expected_pct.clip(lower=EPS)
    actual_pct = actual_pct.clip(lower=EPS)
    psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
    return float(psi)


def compute_drift_table(
    baseline_df: pd.DataFrame,
    shadow_df: pd.DataFrame,
    columns: list[str],
    bins: int = 10,
    kind: str = "feature",
) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in baseline_df.columns or col not in shadow_df.columns:
            continue
        rows.append(
            {
                "kind": kind,
                "column": col,
                "baseline_count": int(pd.Series(baseline_df[col]).dropna().shape[0]),
                "shadow_count": int(pd.Series(shadow_df[col]).dropna().shape[0]),
                "baseline_mean": _safe_float(pd.Series(baseline_df[col]).mean()),
                "shadow_mean": _safe_float(pd.Series(shadow_df[col]).mean()),
                "psi": population_stability_index(baseline_df[col], shadow_df[col], bins=bins),
            }
        )
    return pd.DataFrame(rows)


def _consecutive_loss_count(trades: pd.DataFrame) -> int:
    if len(trades) == 0 or "net_return" not in trades:
        return 0
    max_run = 0
    current = 0
    for value in trades["net_return"].fillna(0):
        if float(value) < 0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def _daily_alpha_returns(equity: pd.DataFrame) -> pd.Series:
    if len(equity) == 0:
        return pd.Series(dtype=float)
    frame = equity.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    col = "alpha_equity" if "alpha_equity" in frame.columns else "equity"
    daily = frame[col].astype(float).resample("1D").last().dropna()
    return daily.pct_change().dropna()


def build_shadow_monitoring_report(
    baseline_df: pd.DataFrame,
    shadow_df: pd.DataFrame,
    feature_cols: list[str],
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    config: dict,
    output_dir: str | Path | None = None,
) -> dict:
    """Build a paper/shadow-trading gate report.

    The caller supplies a baseline research sample, a later shadow sample,
    generated shadow trades, and equity curve. This function only evaluates
    whether the shadow evidence is strong enough to keep considering live use.
    """
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    shadow = shadow_df.copy()
    if not isinstance(shadow.index, pd.DatetimeIndex):
        shadow.index = pd.to_datetime(shadow.index)

    prob_cols = [c for c in ["buy_prob", "sell_prob"] if c in baseline_df.columns and c in shadow.columns]
    feature_drift = compute_drift_table(
        baseline_df=baseline_df,
        shadow_df=shadow,
        columns=feature_cols,
        bins=int(config.get("psi_bins", 10)),
        kind="feature",
    )
    prob_drift = compute_drift_table(
        baseline_df=baseline_df,
        shadow_df=shadow,
        columns=prob_cols,
        bins=int(config.get("psi_bins", 10)),
        kind="probability",
    )

    trades = trades.copy()
    if "entry_time" in trades:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    daily_alpha = _daily_alpha_returns(equity.copy())
    shadow_days = int(shadow.index.normalize().nunique()) if len(shadow) else 0
    trade_count = int(len(trades))
    alpha_col = "alpha_equity" if "alpha_equity" in equity.columns else "equity"
    alpha_return = (
        float(equity[alpha_col].iloc[-1] / equity[alpha_col].iloc[0] - 1)
        if len(equity) and alpha_col in equity
        else np.nan
    )
    running_peak = equity[alpha_col].cummax() if len(equity) and alpha_col in equity else pd.Series(dtype=float)
    drawdown = equity[alpha_col] / running_peak - 1 if len(running_peak) else pd.Series(dtype=float)
    max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan
    win_rate = float((trades["net_return"] > 0).mean()) if trade_count and "net_return" in trades else np.nan
    avg_net_return = float(trades["net_return"].mean()) if trade_count and "net_return" in trades else np.nan
    max_daily_loss = float(daily_alpha.min()) if len(daily_alpha) else np.nan
    max_feature_psi = _safe_float(feature_drift["psi"].max()) if len(feature_drift) else np.nan
    max_probability_psi = _safe_float(prob_drift["psi"].max()) if len(prob_drift) else np.nan
    require_probability_drift = bool(config.get("require_probability_drift", True))

    checks = {
        "enough_shadow_days": shadow_days >= int(config.get("shadow_min_days", 20)),
        "enough_shadow_trades": trade_count >= int(config.get("shadow_min_trades", 30)),
        "alpha_return_positive": pd.notna(alpha_return)
        and alpha_return >= float(config.get("shadow_min_alpha_return", 0.0)),
        "drawdown_within_limit": pd.notna(max_drawdown)
        and max_drawdown >= float(config.get("shadow_max_drawdown", -0.03)),
        "daily_loss_within_limit": pd.notna(max_daily_loss)
        and max_daily_loss >= float(config.get("shadow_max_daily_loss", -0.01)),
        "win_rate_above_floor": pd.notna(win_rate)
        and win_rate >= float(config.get("shadow_min_win_rate", 0.45)),
        "consecutive_losses_within_limit": _consecutive_loss_count(trades)
        <= int(config.get("shadow_max_consecutive_losses", 5)),
        "feature_drift_within_limit": pd.notna(max_feature_psi)
        and max_feature_psi <= float(config.get("drift_max_feature_psi", 0.20)),
        "probability_drift_within_limit": (
            (not require_probability_drift and len(prob_drift) == 0)
            or (pd.notna(max_probability_psi) and max_probability_psi <= float(config.get("drift_max_probability_psi", 0.20)))
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed": {
            "shadow_days": shadow_days,
            "trade_count": trade_count,
            "alpha_total_return": alpha_return,
            "max_drawdown": max_drawdown,
            "max_daily_loss": max_daily_loss,
            "win_rate": win_rate,
            "avg_net_return_per_trade": avg_net_return,
            "max_consecutive_losses": _consecutive_loss_count(trades),
            "max_feature_psi": max_feature_psi,
            "max_probability_psi": max_probability_psi,
        },
        "thresholds": {
            "shadow_min_days": int(config.get("shadow_min_days", 20)),
            "shadow_min_trades": int(config.get("shadow_min_trades", 30)),
            "shadow_min_alpha_return": float(config.get("shadow_min_alpha_return", 0.0)),
            "shadow_max_drawdown": float(config.get("shadow_max_drawdown", -0.03)),
            "shadow_max_daily_loss": float(config.get("shadow_max_daily_loss", -0.01)),
            "shadow_min_win_rate": float(config.get("shadow_min_win_rate", 0.45)),
            "shadow_max_consecutive_losses": int(config.get("shadow_max_consecutive_losses", 5)),
            "drift_max_feature_psi": float(config.get("drift_max_feature_psi", 0.20)),
            "drift_max_probability_psi": float(config.get("drift_max_probability_psi", 0.20)),
            "require_probability_drift": require_probability_drift,
        },
        "note": "Shadow monitoring is a pre-live gate. It is not proof of live profitability.",
    }

    if output_dir is not None:
        feature_drift.to_csv(output_dir / "shadow_feature_drift.csv", index=False)
        prob_drift.to_csv(output_dir / "shadow_probability_drift.csv", index=False)
        pd.DataFrame(
            [{"check": key, "passed": value} for key, value in checks.items()]
        ).to_csv(output_dir / "shadow_monitoring_checks.csv", index=False)
        with (output_dir / "shadow_monitoring_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def load_shadow_monitoring_report(output_dir: str | Path) -> dict | None:
    path = Path(output_dir) / "shadow_monitoring_report.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

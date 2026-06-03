from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
OPTIONAL_EXECUTION_COLUMNS = ["amount"]


def _safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"])
            out = out.set_index("datetime")
        else:
            out.index = pd.to_datetime(out.index)
    return out.sort_index()


def clean_market_data_for_execution(
    df: pd.DataFrame,
    config: dict,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Remove bars/days that cannot support realistic execution.

    The default policy drops an entire trade date if any bar on that date has
    nonpositive OHLC, nonpositive volume, missing OHLCV, inconsistent ranges, or
    is outside the continuous auction session. This is conservative, but it
    prevents sequence models from learning through suspension/data-vendor gaps.
    """
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    data = _to_datetime_index(df)
    numeric_cols = [c for c in REQUIRED_OHLCV_COLUMNS + OPTIONAL_EXECUTION_COLUMNS if c in data.columns]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    missing_required_cols = [c for c in REQUIRED_OHLCV_COLUMNS if c not in data.columns]
    if missing_required_cols:
        raise ValueError(f"missing required OHLCV columns: {missing_required_cols}")

    invalid_mask = pd.Series(False, index=data.index)
    invalid_reasons = pd.DataFrame(index=data.index)
    invalid_reasons["null_ohlcv"] = data[REQUIRED_OHLCV_COLUMNS].isna().any(axis=1)
    invalid_reasons["nonpositive_price"] = (data[["open", "high", "low", "close"]] <= 0).any(axis=1)
    invalid_reasons["nonpositive_volume"] = data["volume"].fillna(0) <= 0
    invalid_reasons["high_low_inconsistent"] = data["high"] < data["low"]
    invalid_reasons["open_outside_range"] = (data["open"] > data["high"]) | (data["open"] < data["low"])
    invalid_reasons["close_outside_range"] = (data["close"] > data["high"]) | (data["close"] < data["low"])
    invalid_reasons["outside_session"] = ~_session_mask(data.index)
    for col in invalid_reasons.columns:
        invalid_mask |= invalid_reasons[col].fillna(True)

    drop_entire_invalid_days = bool(config.get("drop_entire_invalid_days", True))
    invalid_days = pd.Index(data.index.normalize()[invalid_mask]).unique()
    if drop_entire_invalid_days:
        drop_mask = pd.Series(data.index.normalize(), index=data.index).isin(invalid_days).to_numpy()
    else:
        drop_mask = invalid_mask.to_numpy()
    cleaned = data.loc[~drop_mask].copy()

    bad_bars = data.loc[invalid_mask].copy()
    reason_counts = {col: int(invalid_reasons[col].sum()) for col in invalid_reasons.columns}
    report = {
        "status": "PASS" if len(cleaned) > 0 and len(missing_required_cols) == 0 else "FAIL",
        "policy": {
            "drop_entire_invalid_days": drop_entire_invalid_days,
            "invalid_bar_rules": list(invalid_reasons.columns),
        },
        "observed": {
            "raw_rows": int(len(data)),
            "cleaned_rows": int(len(cleaned)),
            "dropped_rows": int(drop_mask.sum()),
            "raw_trade_days": int(pd.Index(data.index.normalize()).nunique()),
            "cleaned_trade_days": int(pd.Index(cleaned.index.normalize()).nunique()) if len(cleaned) else 0,
            "dropped_trade_days": int(len(invalid_days)) if drop_entire_invalid_days else 0,
            "invalid_bar_count": int(invalid_mask.sum()),
            "invalid_reason_counts": reason_counts,
            "first_invalid_timestamp": str(bad_bars.index.min()) if len(bad_bars) else None,
            "last_invalid_timestamp": str(bad_bars.index.max()) if len(bad_bars) else None,
        },
        "hard_rule": "Invalid execution bars must be removed or independently justified before live deployment.",
    }

    if output_dir is not None:
        bad_export = bad_bars.copy()
        for col in invalid_reasons.columns:
            bad_export[f"invalid_{col}"] = invalid_reasons.loc[bad_export.index, col].astype(int)
        bad_export.to_csv(output_dir / "execution_invalid_bars.csv")
        with (output_dir / "market_data_cleaning_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return cleaned, report


def _session_mask(index: pd.DatetimeIndex) -> pd.Series:
    t = pd.Series(index.time, index=index)
    morning = (t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())
    afternoon = (t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time())
    return morning | afternoon


def build_daily_execution_summary(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = _to_datetime_index(df)
    day = data.index.normalize()
    summary = pd.DataFrame(index=pd.Index(sorted(day.unique()), name="trade_date"))
    grouped = data.groupby(day)
    summary["bar_count"] = grouped["close"].size()
    summary["zero_volume_bars"] = grouped["volume"].apply(lambda x: int((pd.to_numeric(x, errors="coerce") <= 0).sum()))
    summary["nan_ohlcv_bars"] = grouped[REQUIRED_OHLCV_COLUMNS].apply(
        lambda x: int(x.apply(pd.to_numeric, errors="coerce").isna().any(axis=1).sum())
    )
    summary["first_bar"] = grouped.apply(lambda x: str(x.index.min().time()))
    summary["last_bar"] = grouped.apply(lambda x: str(x.index.max().time()))

    expected_bars = int(config.get("expected_bars_per_day", 48))
    min_bars = int(config.get("min_bars_per_day", max(1, int(expected_bars * 0.85))))
    summary["expected_bars_per_day"] = expected_bars
    summary["bar_count_shortfall"] = (expected_bars - summary["bar_count"]).clip(lower=0)
    summary["short_day"] = summary["bar_count"] < min_bars
    return summary.reset_index()


def audit_execution_feasibility(
    df: pd.DataFrame,
    config: dict,
    output_dir: str | Path | None = None,
) -> dict:
    """Audit whether OHLCV data is usable for realistic A-share execution tests.

    This is a pre-live hard gate. It does not prove that trades are profitable;
    it catches cases where the backtest may be trading through bad bars, zero
    volume, inconsistent OHLC, duplicate timestamps, or incomplete sessions.
    """
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    data = _to_datetime_index(df)
    missing_required_cols = [c for c in REQUIRED_OHLCV_COLUMNS if c not in data.columns]
    missing_optional_cols = [c for c in OPTIONAL_EXECUTION_COLUMNS if c not in data.columns]
    numeric_cols = [c for c in REQUIRED_OHLCV_COLUMNS + OPTIONAL_EXECUTION_COLUMNS if c in data.columns]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    checks = {}
    if missing_required_cols:
        checks["required_columns_present"] = False
        report = {
            "status": "FAIL",
            "checks": checks,
            "observed": {
                "missing_required_cols": missing_required_cols,
                "missing_optional_cols": missing_optional_cols,
            },
            "hard_rule": "No live deployment if execution feasibility audit fails.",
        }
        if output_dir is not None:
            with (output_dir / "execution_feasibility_audit.json").open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    duplicate_timestamps = int(data.index.duplicated().sum())
    monotonic_index = bool(data.index.is_monotonic_increasing)
    session_valid = _session_mask(data.index)
    outside_session_bars = int((~session_valid).sum())
    null_ohlcv_bars = int(data[REQUIRED_OHLCV_COLUMNS].isna().any(axis=1).sum())
    nonpositive_price_bars = int((data[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    zero_volume_bars = int((data["volume"].fillna(0) <= 0).sum())
    high_low_inconsistent = int((data["high"] < data["low"]).sum())
    open_outside_range = int(((data["open"] > data["high"]) | (data["open"] < data["low"])).sum())
    close_outside_range = int(((data["close"] > data["high"]) | (data["close"] < data["low"])).sum())

    daily_summary = build_daily_execution_summary(data, config)
    short_day_count = int(daily_summary["short_day"].sum()) if len(daily_summary) else 0
    max_bar_shortfall = int(daily_summary["bar_count_shortfall"].max()) if len(daily_summary) else 0

    daily_close = data["close"].groupby(data.index.normalize()).last().astype(float)
    daily_ret = daily_close.pct_change().dropna()
    daily_gap_days = pd.Series(daily_close.index, index=daily_close.index).diff().dt.days
    price_limit_check_max_gap_days = int(config.get("price_limit_check_max_gap_days", 3))
    limit_check_ret = daily_ret[daily_gap_days.loc[daily_ret.index] <= price_limit_check_max_gap_days]
    estimated_limit_rate = float(config.get("a_share_price_limit_rate", 0.20))
    limit_tolerance = float(config.get("price_limit_tolerance", 0.005))
    extreme_daily_move_count = int((limit_check_ret.abs() > estimated_limit_rate + limit_tolerance).sum())

    row_count = int(len(data))
    zero_volume_ratio = zero_volume_bars / row_count if row_count else np.nan
    outside_session_ratio = outside_session_bars / row_count if row_count else np.nan
    short_day_ratio = short_day_count / len(daily_summary) if len(daily_summary) else np.nan

    checks = {
        "required_columns_present": len(missing_required_cols) == 0,
        "datetime_index_monotonic": monotonic_index,
        "no_duplicate_timestamps": duplicate_timestamps == 0,
        "no_null_ohlcv_bars": null_ohlcv_bars == 0,
        "no_nonpositive_price_bars": nonpositive_price_bars == 0,
        "ohlc_ranges_consistent": high_low_inconsistent == 0 and open_outside_range == 0 and close_outside_range == 0,
        "zero_volume_ratio_within_limit": pd.notna(zero_volume_ratio)
        and zero_volume_ratio <= float(config.get("max_zero_volume_bar_ratio", 0.001)),
        "outside_session_ratio_within_limit": pd.notna(outside_session_ratio)
        and outside_session_ratio <= float(config.get("max_outside_session_bar_ratio", 0.0)),
        "short_day_ratio_within_limit": pd.notna(short_day_ratio)
        and short_day_ratio <= float(config.get("max_short_day_ratio", 0.02)),
        "extreme_daily_moves_within_limit": extreme_daily_move_count
        <= int(config.get("max_extreme_daily_move_count", 0)),
    }
    observed = {
        "row_count": row_count,
        "trade_day_count": int(len(daily_summary)),
        "missing_required_cols": missing_required_cols,
        "missing_optional_cols": missing_optional_cols,
        "duplicate_timestamps": duplicate_timestamps,
        "outside_session_bars": outside_session_bars,
        "outside_session_ratio": outside_session_ratio,
        "null_ohlcv_bars": null_ohlcv_bars,
        "nonpositive_price_bars": nonpositive_price_bars,
        "zero_volume_bars": zero_volume_bars,
        "zero_volume_ratio": zero_volume_ratio,
        "high_low_inconsistent": high_low_inconsistent,
        "open_outside_range": open_outside_range,
        "close_outside_range": close_outside_range,
        "short_day_count": short_day_count,
        "short_day_ratio": short_day_ratio,
        "max_bar_shortfall": max_bar_shortfall,
        "extreme_daily_move_count": extreme_daily_move_count,
        "price_limit_checked_day_count": int(len(limit_check_ret)),
        "price_limit_skipped_gap_day_count": int(len(daily_ret) - len(limit_check_ret)),
        "max_abs_daily_return": _safe_float(daily_ret.abs().max()),
        "max_abs_price_limit_checked_return": _safe_float(limit_check_ret.abs().max()),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed": observed,
        "thresholds": {
            "expected_bars_per_day": int(config.get("expected_bars_per_day", 48)),
            "min_bars_per_day": int(config.get("min_bars_per_day", 40)),
            "max_zero_volume_bar_ratio": float(config.get("max_zero_volume_bar_ratio", 0.001)),
            "max_outside_session_bar_ratio": float(config.get("max_outside_session_bar_ratio", 0.0)),
            "max_short_day_ratio": float(config.get("max_short_day_ratio", 0.02)),
            "a_share_price_limit_rate": estimated_limit_rate,
            "price_limit_tolerance": limit_tolerance,
            "price_limit_check_max_gap_days": price_limit_check_max_gap_days,
            "max_extreme_daily_move_count": int(config.get("max_extreme_daily_move_count", 0)),
        },
        "hard_rule": "No live deployment if execution feasibility audit fails.",
    }

    if output_dir is not None:
        daily_summary.to_csv(output_dir / "execution_daily_summary.csv", index=False)
        pd.DataFrame([{"check": k, "passed": v} for k, v in checks.items()]).to_csv(
            output_dir / "execution_feasibility_checks.csv",
            index=False,
        )
        with (output_dir / "execution_feasibility_audit.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report

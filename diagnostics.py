from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from label_builder import run_label_parameter_stability, summarize_opportunity_labels


def feature_scale_report(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    data = df[feature_cols].copy()
    report = pd.DataFrame({
        "dtype": data.dtypes.astype(str),
        "missing_rate": data.isna().mean(),
        "inf_count": np.isinf(data.replace([np.inf, -np.inf], np.nan)).sum(),
        "mean": data.mean(numeric_only=True),
        "std": data.std(numeric_only=True),
        "min": data.min(numeric_only=True),
        "p01": data.quantile(0.01),
        "p50": data.quantile(0.50),
        "p99": data.quantile(0.99),
        "max": data.max(numeric_only=True),
        "abs_max": data.abs().max(numeric_only=True),
        "nunique": data.nunique(),
    })
    report["range_p99_p01"] = report["p99"] - report["p01"]
    report["cv_abs"] = report["std"] / (report["mean"].abs() + 1e-12)
    return report


def scaled_feature_report(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    data = df[feature_cols].copy()
    report = pd.DataFrame({
        "mean": data.mean(),
        "std": data.std(),
        "min": data.min(),
        "p01": data.quantile(0.01),
        "p50": data.quantile(0.50),
        "p99": data.quantile(0.99),
        "max": data.max(),
        "abs_max": data.abs().max(),
        "missing_rate": data.isna().mean(),
    })
    report["mean_abs"] = report["mean"].abs()
    report["std_dev_from_1"] = (report["std"] - 1).abs()
    return report


def sequence_scale_report(x_seq: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    flat = x_seq.reshape(-1, x_seq.shape[-1])
    report = pd.DataFrame({
        "feature": feature_cols,
        "seq_mean": flat.mean(axis=0),
        "seq_std": flat.std(axis=0),
        "seq_abs_mean": np.abs(flat).mean(axis=0),
        "seq_abs_max": np.abs(flat).max(axis=0),
    })
    report["std_dev_from_1"] = np.abs(report["seq_std"] - 1)
    return report


def gradient_importance_report(grad: np.ndarray, feature_cols: list[str], input_array: np.ndarray) -> pd.DataFrame:
    feature_grad_importance = np.abs(grad).mean(axis=(0, 1))
    input_abs_mean = np.abs(input_array).mean(axis=(0, 1))
    report = pd.DataFrame({
        "feature": feature_cols,
        "grad_abs_mean": feature_grad_importance,
        "input_abs_mean": input_abs_mean,
    })
    report["grad_share"] = report["grad_abs_mean"] / (report["grad_abs_mean"].sum() + 1e-12)
    return report.sort_values("grad_abs_mean", ascending=False)


def export_label_diagnostics(
    split_frames: dict[str, pd.DataFrame],
    buy_label_col: str,
    sell_label_col: str,
    config: dict,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, reason_rows, return_rows = [], [], []
    for split_name, part in split_frames.items():
        split_summary, split_reasons = summarize_opportunity_labels(
            part,
            buy_label_col=buy_label_col,
            sell_label_col=sell_label_col,
        )
        split_summary.insert(0, "split", split_name)
        split_reasons.insert(0, "split", split_name)
        summary_rows.append(split_summary)
        reason_rows.append(split_reasons)

        for side, label_col, ret_col in [
            ("buy", buy_label_col, "buy_gross_return"),
            ("sell", sell_label_col, "sell_gross_return"),
        ]:
            for cls in [0, 1]:
                cls_ret = part.loc[part[label_col] == cls, ret_col]
                return_rows.append({
                    "split": split_name,
                    "side": side,
                    "label_col": label_col,
                    "class": cls,
                    "count": int((part[label_col] == cls).sum()),
                    "ratio": float((part[label_col] == cls).mean()),
                    "gross_return_mean": float(cls_ret.mean()),
                    "gross_return_median": float(cls_ret.median()),
                    "gross_return_std": float(cls_ret.std()),
                })

    label_summary = pd.concat(summary_rows, ignore_index=True)
    label_reason_distribution = pd.concat(reason_rows, ignore_index=True)
    label_return_by_class = pd.DataFrame(return_rows)

    label_summary.to_csv(output_dir / "label_summary.csv", index=False)
    label_reason_distribution.to_csv(output_dir / "label_reason_distribution.csv", index=False)
    label_return_by_class.to_csv(output_dir / "label_return_by_class.csv", index=False)

    all_df = split_frames["all"]
    stability = run_label_parameter_stability(
        all_df[["open", "high", "low", "close"]].copy(),
        horizons=config.get("label_stability_horizons", [3, 6, 9, 12]),
        tps=config.get("label_stability_tps", [0.002, 0.003, 0.004]),
        sls=config.get("label_stability_sls", [0.0015, 0.002, 0.003]),
        cost=config["label_cost"],
        same_bar_policy=config["same_bar_policy"],
        timeout_policy=config["timeout_policy"],
        output_dir=output_dir,
    )

    summary = {
        "current_target_col": config["target_col"],
        "sell_target_col": config["sell_target_col"],
        "label_mode": config["label_mode"],
        "horizon": int(config["horizon"]),
        "tp": float(config["tp"]),
        "sl": float(config["sl"]),
        "cost": float(config["label_cost"]),
        "same_bar_policy": config["same_bar_policy"],
        "timeout_policy": config["timeout_policy"],
        "output_files": {
            "label_summary": str(output_dir / "label_summary.csv"),
            "label_reason_distribution": str(output_dir / "label_reason_distribution.csv"),
            "label_return_by_class": str(output_dir / "label_return_by_class.csv"),
            "label_parameter_stability": str(output_dir / "label_parameter_stability.csv"),
        },
    }
    with open(output_dir / "label_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return label_summary, label_reason_distribution, label_return_by_class, stability, summary


def export_signal_diagnostics(
    backtest_result: pd.DataFrame,
    group_analysis: pd.DataFrame,
    target_col: str,
    label_threshold: float,
    round_trip_cost: float,
    test_auc: float | None,
    best_auc: float | None,
    output_dir: str | Path,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_describe = backtest_result["pred_prob"].describe(
        percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]
    )
    prob_describe.to_csv(output_dir / "signal_probability_describe.csv")
    group_analysis.to_csv(output_dir / "signal_group_analysis.csv")

    rows = []
    for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        threshold = backtest_result["pred_prob"].quantile(q)
        signal = backtest_result["pred_prob"] >= threshold
        active = backtest_result[signal]
        rows.append({
            "quantile": q,
            "threshold": float(threshold),
            "signal_count": int(signal.sum()),
            "signal_ratio": float(signal.mean()),
            "active_label_mean": float(active["is_buy_label"].mean()) if len(active) and "is_buy_label" in active.columns else (float(active["y_true"].mean()) if len(active) else np.nan),
            "active_trade_return_mean": float(active["trade_return"].mean()) if len(active) else np.nan,
            "active_trade_return_median": float(active["trade_return"].median()) if len(active) else np.nan,
        })
    threshold_diagnostics = pd.DataFrame(rows)
    threshold_diagnostics.to_csv(output_dir / "signal_threshold_diagnostics.csv", index=False)
    pd.DataFrame().to_csv(output_dir / "signal_fixed_threshold_diagnostics.csv", index=False)

    summary = {
        "test_auc": float(test_auc) if test_auc is not None else None,
        "best_valid_auc": float(best_auc) if best_auc is not None else None,
        "sample_count": int(len(backtest_result)),
        "positive_rate": float(backtest_result["is_buy_label"].mean()) if "is_buy_label" in backtest_result.columns else float(backtest_result["y_true"].mean()),
        "target_col": target_col,
        "label_threshold": float(label_threshold),
        "round_trip_cost": float(round_trip_cost),
        "pred_prob_mean": float(backtest_result["pred_prob"].mean()),
        "pred_prob_std": float(backtest_result["pred_prob"].std()),
        "pred_prob_min": float(backtest_result["pred_prob"].min()),
        "pred_prob_max": float(backtest_result["pred_prob"].max()),
        "note": "This file is signal diagnostics only. Real trading PnL is in outputs/diagnostics/realistic_backtest/.",
        "output_files": {
            "probability_describe": str(output_dir / "signal_probability_describe.csv"),
            "group_analysis": str(output_dir / "signal_group_analysis.csv"),
            "threshold_diagnostics": str(output_dir / "signal_threshold_diagnostics.csv"),
        },
    }
    with open(output_dir / "signal_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return prob_describe, threshold_diagnostics, summary


from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_signals import (
    binary_signal_metrics,
    build_probability_bucket_analysis,
    build_signal_group_analysis,
    build_signal_threshold_diagnostics,
)


def auc_edge_weight(auc_value, floor: float = 0.5) -> float:
    if auc_value is None or pd.isna(auc_value):
        return 0.0
    return float(max(float(auc_value) - floor, 0.0))


def build_auc_weight_table(model_results: dict[str, dict], side: str) -> pd.DataFrame:
    metric = f"valid_{side}_auc"
    rows = []
    for model_name, result in model_results.items():
        summary = result.get("summary", {})
        rows.append(
            {
                "model": model_name,
                "side": side,
                "valid_auc": summary.get(metric, np.nan),
                "raw_weight": auc_edge_weight(summary.get(metric, np.nan)),
            }
        )
    weights = pd.DataFrame(rows)
    total = weights["raw_weight"].sum()
    if total <= 0 or pd.isna(total):
        weights["weight"] = 1.0 / len(weights) if len(weights) else np.nan
        weights["weight_method"] = "equal_fallback_no_positive_auc_edge"
    else:
        weights["weight"] = weights["raw_weight"] / total
        weights["weight_method"] = "valid_auc_edge_over_0p5"
    return weights


def _weighted_average_frames(model_results: dict[str, dict], split: str, prob_col: str, weights: pd.DataFrame) -> np.ndarray:
    prob_matrix = []
    model_order = []
    for model_name, result in model_results.items():
        frame = result[f"{split}_signals"]
        prob_matrix.append(frame[prob_col].to_numpy(dtype=float))
        model_order.append(model_name)
    probs = np.vstack(prob_matrix)
    weight_vector = (
        weights.set_index("model")
        .loc[model_order, "weight"]
        .to_numpy(dtype=float)
        .reshape(-1, 1)
    )
    return (probs * weight_vector).sum(axis=0)


def build_ensemble_signal_result(
    model_results: dict[str, dict],
    config: dict,
    output_dir: str | Path | None = None,
    ensemble_name: str = "auc_weighted_ensemble",
) -> dict:
    if not model_results:
        raise ValueError("model_results cannot be empty")
    output_dir = Path(output_dir or Path(config.get("diagnostics_dir", "outputs/diagnostics")) / "model_signals" / ensemble_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    buy_weights = build_auc_weight_table(model_results, "buy")
    sell_weights = build_auc_weight_table(model_results, "sell")

    first_result = next(iter(model_results.values()))
    valid_signals = first_result["valid_signals"].copy()
    test_signals = first_result["test_signals"].copy()

    valid_signals["buy_prob"] = _weighted_average_frames(model_results, "valid", "buy_prob", buy_weights)
    valid_signals["sell_prob"] = _weighted_average_frames(model_results, "valid", "sell_prob", sell_weights)
    test_signals["buy_prob"] = _weighted_average_frames(model_results, "test", "buy_prob", buy_weights)
    test_signals["sell_prob"] = _weighted_average_frames(model_results, "test", "sell_prob", sell_weights)

    if "buy_label_true" not in valid_signals:
        valid_signals["buy_label_true"] = first_result["valid_signals"]["buy_label_true"]
    if "sell_label_true" not in valid_signals:
        valid_signals["sell_label_true"] = first_result["valid_signals"]["sell_label_true"]
    if "buy_label_true" not in test_signals:
        test_signals["buy_label_true"] = first_result["test_signals"]["buy_label_true"]
    if "sell_label_true" not in test_signals:
        test_signals["sell_label_true"] = first_result["test_signals"]["sell_label_true"]

    summary = {
        "model": ensemble_name,
        "member_models": list(model_results.keys()),
        "buy_weight_method": buy_weights["weight_method"].iloc[0] if len(buy_weights) else None,
        "sell_weight_method": sell_weights["weight_method"].iloc[0] if len(sell_weights) else None,
        "feature_count": first_result.get("summary", {}).get("feature_count"),
        "lookback": first_result.get("summary", {}).get("lookback"),
    }
    summary.update(binary_signal_metrics(valid_signals["buy_label_true"], valid_signals["buy_prob"], "valid_buy"))
    summary.update(binary_signal_metrics(test_signals["buy_label_true"], test_signals["buy_prob"], "test_buy"))
    summary.update(binary_signal_metrics(valid_signals["sell_label_true"], valid_signals["sell_prob"], "valid_sell"))
    summary.update(binary_signal_metrics(test_signals["sell_label_true"], test_signals["sell_prob"], "test_sell"))

    group_analysis = build_signal_group_analysis(test_signals)
    threshold_diagnostics = build_signal_threshold_diagnostics(test_signals)
    probability_bucket_analysis = build_probability_bucket_analysis(test_signals)

    signal_cols = [
        "buy_prob",
        "sell_prob",
        "buy_label_true",
        "sell_label_true",
        "trade_return",
        "future_return",
        "buy_gross_return",
        "sell_gross_return",
        "buy_net_return_est",
        "sell_net_return_est",
    ]
    valid_signals[[c for c in signal_cols if c in valid_signals.columns]].to_csv(output_dir / "valid_signals.csv")
    test_signals[[c for c in signal_cols if c in test_signals.columns]].to_csv(output_dir / "test_signals.csv")
    buy_weights.to_csv(output_dir / "buy_weights.csv", index=False)
    sell_weights.to_csv(output_dir / "sell_weights.csv", index=False)
    group_analysis.to_csv(output_dir / "signal_group_analysis.csv", index=False)
    threshold_diagnostics.to_csv(output_dir / "signal_threshold_diagnostics.csv", index=False)
    probability_bucket_analysis.to_csv(output_dir / "probability_bucket_analysis.csv", index=False)
    with open(output_dir / "signal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "valid_signals": valid_signals,
        "test_signals": test_signals,
        "group_analysis": group_analysis,
        "threshold_diagnostics": threshold_diagnostics,
        "probability_bucket_analysis": probability_bucket_analysis,
        "summary": summary,
        "buy_weights": buy_weights,
        "sell_weights": sell_weights,
        "output_dir": output_dir,
    }

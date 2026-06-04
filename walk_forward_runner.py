from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import build_ensemble_signal_result
from feature_engineering import build_normalization_audit, make_sequence_data, standardize_by_train, winsorize_by_train
from explainability import (
    compute_permutation_importance_binary,
    compute_torch_gradient_importance,
    export_model_explainability,
    try_compute_shap_summary,
)
from model_signals import (
    train_dual_cnn_signals,
    train_dual_logistic_signals,
    train_dual_lstm_signals,
    train_dual_mlp_signals,
    train_dual_random_forest_signals,
    train_dual_transformer_lstm_signals,
)
from risk_management import (
    export_t0_backtest_result,
    run_a_share_inventory_t0_backtest,
    run_inventory_t0_stress_grid,
    run_sell_only_threshold_grid,
    run_t0_dual_signal_backtest,
)
from validation import build_walk_forward_splits, export_walk_forward_splits
from project_paths import display_project_path


MODEL_TRAINERS = {
    "logistic_regression": "sklearn_logit",
    "random_forest": "sklearn_rf",
    "transformer_lstm": train_dual_transformer_lstm_signals,
    "lstm": train_dual_lstm_signals,
    "cnn": train_dual_cnn_signals,
    "mlp": train_dual_mlp_signals,
}

SKLEARN_MODEL_NAMES = {"logistic_regression", "random_forest"}
TORCH_MODEL_NAMES = {"transformer_lstm", "lstm", "cnn", "mlp"}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_walk_forward_progress(
    output_dir: Path,
    rows: list[dict],
    total_folds: int,
    current_status: str,
    current_fold: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_df = pd.DataFrame(rows)
    progress_df.to_csv(output_dir / "walk_forward_progress.csv", index=False)
    completed = int(sum(1 for row in rows if row.get("event") == "fold_completed"))
    payload = {
        "status": current_status,
        "total_folds": int(total_folds),
        "completed_folds": completed,
        "current_fold": current_fold,
        "updated_at": pd.Timestamp.now().isoformat(),
        "progress_file": display_project_path(output_dir / "walk_forward_progress.csv"),
        "last_event": rows[-1] if rows else None,
    }
    with open(output_dir / "walk_forward_progress.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _required_columns(feature_cols: list[str], buy_target_col: str, sell_target_col: str) -> list[str]:
    required = list(feature_cols) + [
        buy_target_col,
        sell_target_col,
        "open",
        "close",
        "trade_return",
        "future_return",
    ]
    return list(dict.fromkeys(required))


def _split_train_calibration(train_window: pd.DataFrame, calibration_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < calibration_ratio < 0.5:
        raise ValueError("calibration_ratio should be in (0, 0.5) for stable walk-forward validation")
    cut = int(len(train_window) * (1 - calibration_ratio))
    if cut <= 0 or cut >= len(train_window):
        raise ValueError("train_window is too small for calibration split")
    return train_window.iloc[:cut].copy(), train_window.iloc[cut:].copy()


def _prepare_fold_data(
    train_window: pd.DataFrame,
    test_window: pd.DataFrame,
    feature_cols: list[str],
    buy_target_col: str,
    sell_target_col: str,
    config: dict,
) -> dict:
    calibration_ratio = float(config.get("walk_forward_calibration_ratio", 0.2))
    subtrain_df, calibration_df = _split_train_calibration(train_window, calibration_ratio)

    required = _required_columns(feature_cols, buy_target_col, sell_target_col)
    subtrain_df = subtrain_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    calibration_df = calibration_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    test_window = test_window.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()

    subtrain_w, calibration_w, test_w, clip_bounds = winsorize_by_train(
        subtrain_df,
        calibration_df,
        test_window,
        feature_cols,
        lower_q=float(config.get("winsor_lower_q", 0.001)),
        upper_q=float(config.get("winsor_upper_q", 0.999)),
    )
    subtrain_scaled, calibration_scaled, test_scaled, scaler = standardize_by_train(
        subtrain_w,
        calibration_w,
        test_w,
        feature_cols,
    )

    lookback = int(config.get("lookback", 32))
    x_train_seq, y_train_buy_seq, train_index_seq = make_sequence_data(
        subtrain_scaled, feature_cols, buy_target_col, lookback
    )
    x_calib_seq, y_calib_buy_seq, calib_index_seq = make_sequence_data(
        calibration_scaled, feature_cols, buy_target_col, lookback
    )
    x_test_seq, y_test_buy_seq, test_index_seq = make_sequence_data(
        test_scaled, feature_cols, buy_target_col, lookback
    )
    _, y_train_sell_seq, train_sell_index_seq = make_sequence_data(
        subtrain_scaled, feature_cols, sell_target_col, lookback
    )
    _, y_calib_sell_seq, calib_sell_index_seq = make_sequence_data(
        calibration_scaled, feature_cols, sell_target_col, lookback
    )
    _, y_test_sell_seq, test_sell_index_seq = make_sequence_data(
        test_scaled, feature_cols, sell_target_col, lookback
    )

    if not (
        np.array_equal(train_index_seq, train_sell_index_seq)
        and np.array_equal(calib_index_seq, calib_sell_index_seq)
        and np.array_equal(test_index_seq, test_sell_index_seq)
    ):
        raise ValueError("buy/sell sequence index alignment failed")

    return {
        "subtrain_df": subtrain_df,
        "calibration_df": calibration_df,
        "test_df": test_window,
        "subtrain_scaled": subtrain_scaled,
        "calibration_scaled": calibration_scaled,
        "test_scaled": test_scaled,
        "clip_bounds": clip_bounds,
        "scaler": scaler,
        "x_train_seq": x_train_seq,
        "y_train_buy_seq": y_train_buy_seq,
        "y_train_sell_seq": y_train_sell_seq,
        "train_index_seq": train_index_seq,
        "x_calib_seq": x_calib_seq,
        "y_calib_buy_seq": y_calib_buy_seq,
        "y_calib_sell_seq": y_calib_sell_seq,
        "calib_index_seq": calib_index_seq,
        "x_test_seq": x_test_seq,
        "y_test_buy_seq": y_test_buy_seq,
        "y_test_sell_seq": y_test_sell_seq,
        "test_index_seq": test_index_seq,
    }


def _train_fold_model(
    model_name: str,
    fold_data: dict,
    feature_cols: list[str],
    buy_target_col: str,
    sell_target_col: str,
    config: dict,
    output_dir: Path,
    device: str | None,
) -> dict:
    if model_name == "logistic_regression":
        return train_dual_logistic_signals(
            train_scaled=fold_data["subtrain_scaled"],
            valid_scaled=fold_data["calibration_scaled"],
            test_scaled=fold_data["test_scaled"],
            valid_df=fold_data["calibration_df"],
            test_df=fold_data["test_df"],
            feature_cols=feature_cols,
            buy_target_col=buy_target_col,
            sell_target_col=sell_target_col,
            valid_index_seq=fold_data["calib_index_seq"],
            test_index_seq=fold_data["test_index_seq"],
            config=config,
            output_dir=output_dir / model_name,
        )
    if model_name == "random_forest":
        return train_dual_random_forest_signals(
            train_scaled=fold_data["subtrain_scaled"],
            valid_scaled=fold_data["calibration_scaled"],
            test_scaled=fold_data["test_scaled"],
            valid_df=fold_data["calibration_df"],
            test_df=fold_data["test_df"],
            feature_cols=feature_cols,
            buy_target_col=buy_target_col,
            sell_target_col=sell_target_col,
            valid_index_seq=fold_data["calib_index_seq"],
            test_index_seq=fold_data["test_index_seq"],
            config=config,
            output_dir=output_dir / model_name,
        )

    trainer = MODEL_TRAINERS.get(model_name)
    if trainer is None or isinstance(trainer, str):
        raise ValueError(f"Unsupported model_name: {model_name}")
    return trainer(
        x_train_seq=fold_data["x_train_seq"],
        y_train_buy_seq=fold_data["y_train_buy_seq"],
        y_train_sell_seq=fold_data["y_train_sell_seq"],
        x_valid_seq=fold_data["x_calib_seq"],
        y_valid_buy_seq=fold_data["y_calib_buy_seq"],
        y_valid_sell_seq=fold_data["y_calib_sell_seq"],
        x_test_seq=fold_data["x_test_seq"],
        y_test_buy_seq=fold_data["y_test_buy_seq"],
        y_test_sell_seq=fold_data["y_test_sell_seq"],
        valid_df=fold_data["calibration_df"],
        test_df=fold_data["test_df"],
        valid_index_seq=fold_data["calib_index_seq"],
        test_index_seq=fold_data["test_index_seq"],
        config=config,
        output_dir=output_dir / model_name,
        device=device,
    )


def evaluate_quality_gates(fold_summary: pd.DataFrame, config: dict) -> dict:
    trade_direction_mode = str(config.get("trade_direction_mode", "both"))
    gates = {
        "min_folds": int(config.get("quality_min_folds", 3)),
        "min_total_trades": int(config.get("quality_min_total_trades", 30)),
        "min_median_test_buy_auc": float(config.get("quality_min_median_test_buy_auc", 0.52)),
        "min_median_test_sell_auc": float(config.get("quality_min_median_test_sell_auc", 0.52)),
        "min_median_alpha_return": float(config.get("quality_min_median_alpha_return", 0.0)),
        "max_median_drawdown": float(config.get("quality_max_median_drawdown", -0.08)),
        "trade_direction_mode": trade_direction_mode,
    }
    drawdown_col = "alpha_max_drawdown" if "alpha_max_drawdown" in fold_summary else "max_drawdown"
    alpha_return_col = "alpha_total_return" if "alpha_total_return" in fold_summary else "total_return"
    observed = {
        "fold_count": int(len(fold_summary)),
        "total_trades": int(fold_summary["trade_count"].sum()) if "trade_count" in fold_summary else 0,
        "median_test_buy_auc": float(fold_summary["test_buy_auc"].median()) if "test_buy_auc" in fold_summary and len(fold_summary) else np.nan,
        "median_test_sell_auc": float(fold_summary["test_sell_auc"].median()) if "test_sell_auc" in fold_summary and len(fold_summary) else np.nan,
        "median_alpha_return": float(fold_summary[alpha_return_col].median()) if alpha_return_col in fold_summary and len(fold_summary) else np.nan,
        "median_max_drawdown": float(fold_summary[drawdown_col].median()) if drawdown_col in fold_summary and len(fold_summary) else np.nan,
        "drawdown_metric": drawdown_col,
        "return_metric": alpha_return_col,
    }
    buy_auc_ok = (
        pd.notna(observed["median_test_buy_auc"])
        and observed["median_test_buy_auc"] >= gates["min_median_test_buy_auc"]
    )
    sell_auc_ok = (
        pd.notna(observed["median_test_sell_auc"])
        and observed["median_test_sell_auc"] >= gates["min_median_test_sell_auc"]
    )
    if trade_direction_mode == "sell_only":
        direction_auc_ok = sell_auc_ok
    elif trade_direction_mode == "buy_only":
        direction_auc_ok = buy_auc_ok
    else:
        direction_auc_ok = buy_auc_ok and sell_auc_ok
    checks = {
        "enough_folds": observed["fold_count"] >= gates["min_folds"],
        "enough_trades": observed["total_trades"] >= gates["min_total_trades"],
        "buy_auc_above_gate": buy_auc_ok,
        "sell_auc_above_gate": sell_auc_ok,
        "direction_auc_above_gate": direction_auc_ok,
        "alpha_return_above_gate": pd.notna(observed["median_alpha_return"])
        and observed["median_alpha_return"] >= gates["min_median_alpha_return"],
        "drawdown_within_gate": pd.notna(observed["median_max_drawdown"])
        and observed["median_max_drawdown"] >= gates["max_median_drawdown"],
    }
    return {
        "gates": gates,
        "observed": observed,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
        "note": "Research gate only. Passing this does not prove live-trading readiness.",
    }


def _export_fold_explainability(
    model_results: dict[str, dict],
    fold_data: dict,
    feature_cols: list[str],
    config: dict,
    output_dir: Path,
    device: str | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    top_n = int(config.get("explainability_top_features", 30))
    n_repeats = int(config.get("permutation_repeats", 3))
    max_shap_samples = int(config.get("shap_max_samples", 512))
    max_gradient_samples = int(config.get("gradient_max_samples", 1024))

    permutation_results = {}
    shap_results = {}
    gradient_results = {}

    for model_name, result in model_results.items():
        safe_name = model_name.lower().replace(" ", "_")
        if model_name in SKLEARN_MODEL_NAMES:
            test_index = pd.to_datetime(fold_data["test_index_seq"])
            x_test = fold_data["test_scaled"].loc[test_index, feature_cols]
            y_test_buy = fold_data["test_scaled"].loc[test_index, config.get("buy_label_col", "buy_label")]
            y_test_sell = fold_data["test_scaled"].loc[test_index, config.get("sell_label_col", "sell_label")]
            permutation_results[f"{safe_name}_buy"] = compute_permutation_importance_binary(
                result["buy_model"],
                x_test,
                y_test_buy,
                feature_cols,
                n_repeats=n_repeats,
                random_state=int(config.get("random_state", 42)),
            ).head(top_n)
            permutation_results[f"{safe_name}_sell"] = compute_permutation_importance_binary(
                result["sell_model"],
                x_test,
                y_test_sell,
                feature_cols,
                n_repeats=n_repeats,
                random_state=int(config.get("random_state", 42)),
            ).head(top_n)
            shap_results[f"{safe_name}_buy"] = try_compute_shap_summary(
                result["buy_model"],
                x_test,
                feature_cols,
                max_samples=max_shap_samples,
            )
            shap_results[f"{safe_name}_sell"] = try_compute_shap_summary(
                result["sell_model"],
                x_test,
                feature_cols,
                max_samples=max_shap_samples,
            )
        elif model_name in TORCH_MODEL_NAMES:
            gradient_results[f"{safe_name}_buy"] = compute_torch_gradient_importance(
                result["buy_model"],
                fold_data["x_test_seq"],
                feature_cols,
                device=device or result["summary"].get("buy_fit", {}).get("device", "cpu"),
                max_samples=max_gradient_samples,
            ).head(top_n)
            gradient_results[f"{safe_name}_sell"] = compute_torch_gradient_importance(
                result["sell_model"],
                fold_data["x_test_seq"],
                feature_cols,
                device=device or result["summary"].get("sell_fit", {}).get("device", "cpu"),
                max_samples=max_gradient_samples,
            ).head(top_n)

    return export_model_explainability(
        output_dir=output_dir,
        summary={
            "model_count": len(model_results),
            "models": list(model_results.keys()),
            "top_features": top_n,
            "permutation_repeats": n_repeats,
            "note": "Permutation and SHAP are computed for sklearn models; gradient importance is computed for torch sequence models.",
        },
        permutation_results=permutation_results,
        gradient_results=gradient_results,
        shap_results=shap_results,
    )
def run_walk_forward_signal_research(
    data: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
    output_dir: str | Path | None = None,
    model_names: list[str] | None = None,
    device: str | None = None,
) -> dict:
    output_dir = Path(output_dir or Path(config.get("diagnostics_dir", "outputs/diagnostics")) / "walk_forward")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = data.sort_index().copy()
    buy_target_col = config.get("buy_label_col", config.get("target_col", "buy_label"))
    sell_target_col = config.get("sell_label_col", config.get("sell_target_col", "sell_label"))
    required = _required_columns(feature_cols, buy_target_col, sell_target_col)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()

    splits = build_walk_forward_splits(
        data.index,
        train_bars=int(config.get("walk_forward_train_bars", 12000)),
        valid_bars=int(config.get("walk_forward_valid_bars", 2400)),
        step_bars=int(config.get("walk_forward_step_bars", 2400)),
        expanding=bool(config.get("walk_forward_expanding", False)),
    )
    max_folds = config.get("walk_forward_max_folds")
    if max_folds is not None:
        splits = splits[: int(max_folds)]
    split_df = export_walk_forward_splits(splits, output_dir)

    model_names = model_names or list(config.get("walk_forward_model_names", ["logistic_regression", "random_forest"]))
    fold_rows = []
    fold_artifacts = []
    progress_rows = []
    total_folds = len(splits)
    _write_walk_forward_progress(
        output_dir,
        progress_rows,
        total_folds=total_folds,
        current_status="started",
        current_fold=None,
    )

    for split in splits:
        fold = int(split["fold"])
        fold_start_ts = time.time()
        fold_dir = output_dir / f"fold_{fold:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        progress_rows.append(
            {
                "event": "fold_started",
                "fold": fold,
                "timestamp": pd.Timestamp.now().isoformat(),
                "selected_model": None,
                "elapsed_seconds": np.nan,
                "output_dir": display_project_path(fold_dir),
            }
        )
        _write_walk_forward_progress(
            output_dir,
            progress_rows,
            total_folds=total_folds,
            current_status="running",
            current_fold=fold,
        )

        train_window = data.iloc[split["train_start_pos"] : split["train_end_pos"]].copy()
        test_window = data.iloc[split["valid_start_pos"] : split["valid_end_pos"]].copy()
        fold_data = _prepare_fold_data(
            train_window=train_window,
            test_window=test_window,
            feature_cols=feature_cols,
            buy_target_col=buy_target_col,
            sell_target_col=sell_target_col,
            config=config,
        )
        normalization_audit, normalization_summary = build_normalization_audit(
            fold_data["subtrain_scaled"],
            fold_data["calibration_scaled"],
            fold_data["test_scaled"],
            feature_cols,
        )
        normalization_audit.to_csv(fold_dir / "normalization_audit.csv", index=False)
        with open(fold_dir / "normalization_summary.json", "w", encoding="utf-8") as f:
            json.dump(normalization_summary, f, ensure_ascii=False, indent=2, default=_json_default)

        model_results = {}
        for model_name in model_names:
            model_start_ts = time.time()
            model_results[model_name] = _train_fold_model(
                model_name=model_name,
                fold_data=fold_data,
                feature_cols=feature_cols,
                buy_target_col=buy_target_col,
                sell_target_col=sell_target_col,
                config=config,
                output_dir=fold_dir / "models",
                device=device,
            )
            progress_rows.append(
                {
                    "event": "model_completed",
                    "fold": fold,
                    "model": model_name,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "valid_buy_auc": model_results[model_name]["summary"].get("valid_buy_auc", np.nan),
                    "valid_sell_auc": model_results[model_name]["summary"].get("valid_sell_auc", np.nan),
                    "test_buy_auc": model_results[model_name]["summary"].get("test_buy_auc", np.nan),
                    "test_sell_auc": model_results[model_name]["summary"].get("test_sell_auc", np.nan),
                    "elapsed_seconds": round(time.time() - model_start_ts, 3),
                    "output_dir": display_project_path(model_results[model_name].get("output_dir", "")),
                }
            )
            _write_walk_forward_progress(
                output_dir,
                progress_rows,
                total_folds=total_folds,
                current_status="running",
                current_fold=fold,
            )

        if len(model_results) >= 2:
            signal_result = build_ensemble_signal_result(
                model_results,
                config=config,
                output_dir=fold_dir / "models" / "auc_weighted_ensemble",
            )
            selected_model_name = "auc_weighted_ensemble"
        else:
            selected_model_name, signal_result = next(iter(model_results.items()))

        explainability_manifest = None
        if config.get("run_explainability", False):
            explainability_manifest = _export_fold_explainability(
                model_results=model_results,
                fold_data=fold_data,
                feature_cols=feature_cols,
                config=config,
                output_dir=fold_dir / "explainability",
                device=device,
            )

        buy_quantile = float(config.get("buy_threshold_quantile", config.get("fixed_threshold_quantile", 0.95)))
        sell_quantile = float(config.get("sell_threshold_quantile", config.get("fixed_threshold_quantile", 0.95)))
        buy_threshold = float(signal_result["valid_signals"]["buy_prob"].quantile(buy_quantile))
        sell_threshold = float(signal_result["valid_signals"]["sell_prob"].quantile(sell_quantile))
        if config.get("a_share_t0_mode", "inventory") == "inventory":
            trades, equity, stats = run_a_share_inventory_t0_backtest(
                signal_result["test_signals"],
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                config=config,
            )
        else:
            trades, equity, stats = run_t0_dual_signal_backtest(
                signal_result["test_signals"],
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                config=config,
            )
        export_t0_backtest_result(fold_dir / "t0_backtest", trades, equity, stats)
        sell_threshold_grid_stats = {}
        if config.get("run_sell_threshold_grid", False) and config.get("a_share_t0_mode", "inventory") == "inventory":
            sell_threshold_grid = run_sell_only_threshold_grid(
                valid_signals=signal_result["valid_signals"],
                test_signals=signal_result["test_signals"],
                config=config,
                output_dir=fold_dir / "sell_only_threshold_grid",
            )
            if len(sell_threshold_grid):
                best_grid_row = sell_threshold_grid.iloc[0].to_dict()
                sell_threshold_grid_stats = {
                    "sell_grid_best_rule": best_grid_row.get("rule_name"),
                    "sell_grid_best_quantile": best_grid_row.get("sell_quantile"),
                    "sell_grid_best_alpha_total_return": best_grid_row.get("alpha_total_return"),
                    "sell_grid_best_trade_count": best_grid_row.get("trade_count"),
                    "sell_grid_result_count": int(len(sell_threshold_grid)),
                }
        if config.get("a_share_t0_mode", "inventory") == "inventory":
            stress_df = run_inventory_t0_stress_grid(
                signal_result["test_signals"],
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                config=config,
                output_dir=fold_dir / "stress_tests",
            )
            stress_stats = {
                "stress_scenario_count": int(len(stress_df)),
                "stress_min_alpha_total_return": float(stress_df["alpha_total_return"].min()) if "alpha_total_return" in stress_df and len(stress_df) else np.nan,
                "stress_worst_alpha_drawdown": float(stress_df["alpha_max_drawdown"].min()) if "alpha_max_drawdown" in stress_df and len(stress_df) else np.nan,
                "stress_min_trade_count": int(stress_df["trade_count"].min()) if "trade_count" in stress_df and len(stress_df) else 0,
            }
        else:
            stress_df = pd.DataFrame()
            stress_stats = {
                "stress_scenario_count": 0,
                "stress_min_alpha_total_return": np.nan,
                "stress_worst_alpha_drawdown": np.nan,
                "stress_min_trade_count": 0,
            }

        row = {
            "fold": fold,
            "selected_model": selected_model_name,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "test_start": split["valid_start"],
            "test_end": split["valid_end"],
            "subtrain_rows": int(len(fold_data["subtrain_df"])),
            "calibration_rows": int(len(fold_data["calibration_df"])),
            "test_rows": int(len(fold_data["test_df"])),
            "test_buy_auc": signal_result["summary"].get("test_buy_auc", np.nan),
            "test_sell_auc": signal_result["summary"].get("test_sell_auc", np.nan),
            "valid_buy_auc": signal_result["summary"].get("valid_buy_auc", np.nan),
            "valid_sell_auc": signal_result["summary"].get("valid_sell_auc", np.nan),
            **stats,
            **stress_stats,
            **sell_threshold_grid_stats,
            "output_dir": display_project_path(fold_dir),
        }
        fold_rows.append(row)
        fold_artifacts.append(
            {
                "fold": fold,
                "model_summaries": {name: result["summary"] for name, result in model_results.items()},
                "selected_model_summary": signal_result["summary"],
                "t0_stats": stats,
                "stress_stats": stress_stats,
                "sell_threshold_grid_stats": sell_threshold_grid_stats,
                "normalization_summary": normalization_summary,
                "explainability_manifest": explainability_manifest,
                "output_dir": display_project_path(fold_dir),
            }
        )
        with open(fold_dir / "fold_manifest.json", "w", encoding="utf-8") as f:
            json.dump(fold_artifacts[-1], f, ensure_ascii=False, indent=2, default=_json_default)
        progress_rows.append(
            {
                "event": "fold_completed",
                "fold": fold,
                "timestamp": pd.Timestamp.now().isoformat(),
                "selected_model": selected_model_name,
                "test_buy_auc": row.get("test_buy_auc", np.nan),
                "test_sell_auc": row.get("test_sell_auc", np.nan),
                "alpha_total_return": row.get("alpha_total_return", np.nan),
                "trade_count": row.get("trade_count", 0),
                "elapsed_seconds": round(time.time() - fold_start_ts, 3),
                "output_dir": display_project_path(fold_dir),
            }
        )
        _write_walk_forward_progress(
            output_dir,
            progress_rows,
            total_folds=total_folds,
            current_status="running",
            current_fold=fold,
        )

    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(output_dir / "walk_forward_fold_summary.csv", index=False)
    quality_report = evaluate_quality_gates(fold_summary, config)
    aggregate_summary = {
        "model_names": model_names,
        "split_count": int(len(split_df)),
        "executed_fold_count": int(len(fold_summary)),
        "mean_total_return": float(fold_summary["total_return"].mean()) if len(fold_summary) else np.nan,
        "median_total_return": float(fold_summary["total_return"].median()) if len(fold_summary) else np.nan,
        "mean_alpha_total_return": float(fold_summary["alpha_total_return"].mean()) if "alpha_total_return" in fold_summary and len(fold_summary) else np.nan,
        "median_alpha_total_return": float(fold_summary["alpha_total_return"].median()) if "alpha_total_return" in fold_summary and len(fold_summary) else np.nan,
        "mean_test_buy_auc": float(fold_summary["test_buy_auc"].mean()) if len(fold_summary) else np.nan,
        "median_test_buy_auc": float(fold_summary["test_buy_auc"].median()) if len(fold_summary) else np.nan,
        "mean_test_sell_auc": float(fold_summary["test_sell_auc"].mean()) if len(fold_summary) else np.nan,
        "median_test_sell_auc": float(fold_summary["test_sell_auc"].median()) if len(fold_summary) else np.nan,
        "total_trades": int(fold_summary["trade_count"].sum()) if len(fold_summary) else 0,
        "stress_min_alpha_total_return": float(fold_summary["stress_min_alpha_total_return"].min()) if "stress_min_alpha_total_return" in fold_summary and len(fold_summary) else np.nan,
        "stress_worst_alpha_drawdown": float(fold_summary["stress_worst_alpha_drawdown"].min()) if "stress_worst_alpha_drawdown" in fold_summary and len(fold_summary) else np.nan,
        "quality_report": quality_report,
        "methodology": {
            "split_policy": "chronological walk-forward by bar position; no random shuffle",
            "fold_policy": "each train window is split into subtrain and calibration; fold validation is treated as out-of-sample test",
            "preprocessing_policy": "winsorization and standardization are fit on subtrain only, then applied to calibration and test",
            "threshold_policy": "buy/sell thresholds are calibration-set fixed quantiles, never selected from the fold test window",
            "ensemble_policy": "model weights are proportional to validation AUC edge over 0.5; equal fallback if no model beats random",
            "execution_policy": "signal at bar t, execute at next bar open with commission, slippage, lot size, no-overlap state machine, tp/sl and no-overnight handling",
            "a_share_t0_policy": "default inventory mode starts with tradable base shares; same-day sells use existing inventory, buybacks restore inventory, and buy-first exits sell old inventory rather than same-day purchases",
            "liquidity_policy": "single trade size is capped by max_participation_rate of next-bar volume when volume is available",
            "explainability_policy": "optional per-fold permutation/SHAP for sklearn models and input-gradient importance for torch sequence models",
        },
    }
    with open(output_dir / "walk_forward_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, ensure_ascii=False, indent=2, default=_json_default)
    with open(output_dir / "methodology.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_summary["methodology"], f, ensure_ascii=False, indent=2)
    with open(output_dir / "walk_forward_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(fold_artifacts, f, ensure_ascii=False, indent=2, default=_json_default)
    _write_walk_forward_progress(
        output_dir,
        progress_rows,
        total_folds=total_folds,
        current_status="completed",
        current_fold=None,
    )

    return {
        "splits": split_df,
        "fold_summary": fold_summary,
        "aggregate_summary": aggregate_summary,
        "quality_report": quality_report,
        "output_dir": output_dir,
    }

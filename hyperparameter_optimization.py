from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from project_paths import display_project_path


def suggest_sequence_config(trial, base_config: dict, model_name: str) -> dict:
    config = dict(base_config)
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    config["weight_decay"] = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    config["early_stop_patience"] = int(base_config.get("early_stop_patience", 8))

    if model_name == "transformer_lstm":
        d_model = trial.suggest_categorical("transformer_d_model", [64, 128, 192, 256])
        nhead_candidates = [h for h in [2, 4, 8] if d_model % h == 0]
        config["transformer_d_model"] = d_model
        config["transformer_nhead"] = trial.suggest_categorical("transformer_nhead", nhead_candidates)
        config["transformer_num_layers"] = trial.suggest_int("transformer_num_layers", 1, 3)
        config["transformer_dropout"] = trial.suggest_float("transformer_dropout", 0.05, 0.5)
        config["transformer_lstm_hidden_dim"] = trial.suggest_categorical("transformer_lstm_hidden_dim", [64, 128, 256])
        config["transformer_lstm_num_layers"] = trial.suggest_int("transformer_lstm_num_layers", 1, 2)
    elif model_name == "lstm":
        config["lstm_hidden_dim"] = trial.suggest_categorical("lstm_hidden_dim", [64, 128, 256])
        config["lstm_num_layers"] = trial.suggest_int("lstm_num_layers", 1, 3)
        config["lstm_dropout"] = trial.suggest_float("lstm_dropout", 0.05, 0.5)
    elif model_name == "cnn":
        config["cnn_channels"] = trial.suggest_categorical("cnn_channels", [32, 64, 128])
        config["cnn_kernel_size"] = trial.suggest_categorical("cnn_kernel_size", [3, 5, 7])
        config["cnn_dropout"] = trial.suggest_float("cnn_dropout", 0.05, 0.5)
    elif model_name == "mlp":
        config["mlp_hidden_dim"] = trial.suggest_categorical("mlp_hidden_dim", [128, 256, 512])
        config["mlp_dropout"] = trial.suggest_float("mlp_dropout", 0.05, 0.5)
    else:
        raise ValueError(f"unsupported model_name: {model_name}")
    return config


def run_optuna_sequence_search(
    trainer: Callable,
    model_name: str,
    base_config: dict,
    trainer_kwargs: dict,
    output_dir: str | Path,
    n_trials: int = 20,
    timeout: int | None = None,
    direction: str = "maximize",
) -> object:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install optuna to run hyperparameter optimization.") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        config = suggest_sequence_config(trial, base_config, model_name)
        trial_output = output_dir / f"trial_{trial.number:04d}"
        result = trainer(config=config, output_dir=trial_output, **trainer_kwargs)
        score = result["summary"].get("valid_buy_auc", np.nan)
        if not np.isfinite(score):
            score = -1.0
        trial.set_user_attr("valid_sell_auc", result["summary"].get("valid_sell_auc", np.nan))
        trial.set_user_attr("test_buy_auc", result["summary"].get("test_buy_auc", np.nan))
        trial.set_user_attr("test_sell_auc", result["summary"].get("test_sell_auc", np.nan))
        trial.set_user_attr("output_dir", display_project_path(trial_output))
        return float(score)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    best_payload = {
        "model_name": model_name,
        "direction": direction,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    with open(output_dir / "optuna_best.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)
    return study


def suggest_walk_forward_config(trial, base_config: dict, model_names: list[str]) -> dict:
    """Suggest parameters that affect validation-to-trading conversion and model fit."""
    config = dict(base_config)
    names = set(model_names)

    config["trade_direction_mode"] = trial.suggest_categorical("trade_direction_mode", ["sell_only"])
    config["buy_threshold_quantile"] = trial.suggest_categorical("buy_threshold_quantile", [0.95])
    config["sell_threshold_quantile"] = trial.suggest_categorical(
        "sell_threshold_quantile",
        [0.90, 0.92, 0.94, 0.95, 0.96, 0.97],
    )
    config["position_sizing_high_quantile"] = trial.suggest_categorical(
        "position_sizing_high_quantile",
        [0.97, 0.985, 0.99, 0.995],
    )
    config["min_position_pct"] = trial.suggest_categorical("min_position_pct", [0.10, 0.20, 0.25, 0.30])
    config["max_t0_trade_pct"] = trial.suggest_categorical("max_t0_trade_pct", [0.10, 0.20, 0.30])
    config["max_participation_rate"] = trial.suggest_categorical("max_participation_rate", [0.02, 0.03, 0.05])

    if "random_forest" in names:
        config["rf_n_estimators"] = trial.suggest_categorical("rf_n_estimators", [100, 200, 300, 500])
        config["rf_max_depth"] = trial.suggest_categorical("rf_max_depth", [4, 6, 8, 10, None])
        config["rf_min_samples_leaf"] = trial.suggest_categorical("rf_min_samples_leaf", [20, 50, 100, 150])
        config["rf_max_features"] = trial.suggest_categorical("rf_max_features", ["sqrt", "log2"])

    if "logistic_regression" in names:
        config["logit_c"] = trial.suggest_float("logit_c", 0.01, 10.0, log=True)
        config["logit_class_weight"] = trial.suggest_categorical("logit_class_weight", ["balanced", None])

    if names.intersection({"transformer_lstm", "lstm", "cnn", "mlp"}):
        config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 2e-3, log=True)
        config["weight_decay"] = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
        config["batch_size_train"] = trial.suggest_categorical("batch_size_train", [64, 128, 256])
        config["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [0.5, 1.0, 2.0])

    if "lstm" in names:
        config["lstm_hidden_dim"] = trial.suggest_categorical("lstm_hidden_dim", [64, 128, 256])
        config["lstm_num_layers"] = trial.suggest_int("lstm_num_layers", 1, 3)
        config["lstm_dropout"] = trial.suggest_float("lstm_dropout", 0.05, 0.5)

    if "transformer_lstm" in names:
        d_model = trial.suggest_categorical("transformer_d_model", [64, 128, 192])
        config["transformer_d_model"] = d_model
        config["transformer_nhead"] = trial.suggest_categorical(
            "transformer_nhead",
            [h for h in [2, 4, 8] if d_model % h == 0],
        )
        config["transformer_num_layers"] = trial.suggest_int("transformer_num_layers", 1, 3)
        config["transformer_dropout"] = trial.suggest_float("transformer_dropout", 0.05, 0.5)
        config["transformer_lstm_hidden_dim"] = trial.suggest_categorical("transformer_lstm_hidden_dim", [64, 128, 256])
        config["transformer_lstm_num_layers"] = trial.suggest_int("transformer_lstm_num_layers", 1, 2)

    if "cnn" in names:
        config["cnn_channels"] = trial.suggest_categorical("cnn_channels", [32, 64, 128])
        config["cnn_kernel_size"] = trial.suggest_categorical("cnn_kernel_size", [3, 5, 7])
        config["cnn_dropout"] = trial.suggest_float("cnn_dropout", 0.05, 0.5)

    if "mlp" in names:
        config["mlp_hidden_dim"] = trial.suggest_categorical("mlp_hidden_dim", [128, 256, 512])
        config["mlp_dropout"] = trial.suggest_float("mlp_dropout", 0.05, 0.5)

    return config


def score_walk_forward_summary(summary: dict, min_trades: int = 1) -> float:
    """Trading-first objective: prefer positive alpha, robustness, and enough trades."""
    median_alpha = float(summary.get("median_alpha_total_return", np.nan))
    mean_alpha = float(summary.get("mean_alpha_total_return", np.nan))
    stress_alpha = float(summary.get("stress_min_alpha_total_return", np.nan))
    trades = int(summary.get("total_trades", 0) or 0)
    buy_auc = float(summary.get("median_test_buy_auc", np.nan))
    sell_auc = float(summary.get("median_test_sell_auc", np.nan))

    if not np.isfinite(median_alpha):
        median_alpha = -1.0
    if not np.isfinite(mean_alpha):
        mean_alpha = median_alpha
    if not np.isfinite(stress_alpha):
        stress_alpha = median_alpha
    if not np.isfinite(buy_auc):
        buy_auc = 0.5
    if not np.isfinite(sell_auc):
        sell_auc = 0.5

    trade_penalty = 0.0
    if trades < min_trades:
        trade_penalty = -0.02 * (min_trades - trades) / max(min_trades, 1)

    auc_edge = max(0.0, (buy_auc + sell_auc) / 2 - 0.5)
    return float(median_alpha + 0.25 * mean_alpha + 0.25 * stress_alpha + 0.02 * auc_edge + trade_penalty)


def run_walk_forward_config_search(
    evaluator: Callable[[dict, Path], dict],
    base_config: dict,
    model_names: list[str],
    output_dir: str | Path,
    n_trials: int = 20,
    timeout: int | None = None,
    direction: str = "maximize",
) -> dict:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install optuna to run walk-forward hyperparameter optimization.") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_trades = int(base_config.get("quality_min_total_trades", 1))

    def objective(trial):
        config = suggest_walk_forward_config(trial, base_config, model_names)
        trial_output = output_dir / f"trial_{trial.number:04d}"
        result = evaluator(config, trial_output)
        summary = result.get("aggregate_summary", {})
        score = score_walk_forward_summary(summary, min_trades=min_trades)
        for key in [
            "median_alpha_total_return",
            "mean_alpha_total_return",
            "stress_min_alpha_total_return",
            "total_trades",
            "median_test_buy_auc",
            "median_test_sell_auc",
        ]:
            trial.set_user_attr(key, summary.get(key))
        trial.set_user_attr("output_dir", display_project_path(trial_output))
        return score

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "walk_forward_optuna_trials.csv", index=False)
    best_config = dict(base_config)
    best_config.update(study.best_params)
    best_payload = {
        "status": "PASS",
        "direction": direction,
        "model_names": model_names,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_trial_number": int(study.best_trial.number),
        "best_config_overrides": study.best_params,
        "objective": "median_alpha + robustness + small AUC edge, with low-trade penalty",
    }
    with open(output_dir / "walk_forward_optuna_best.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)
    return {
        "study": study,
        "report": best_payload,
        "best_config": best_config,
        "trials": trials_df,
        "output_dir": output_dir,
    }

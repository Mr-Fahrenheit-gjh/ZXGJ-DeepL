from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


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
        trial.set_user_attr("output_dir", str(trial_output))
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

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


BASE_MVP_CONFIG: dict = {
    "symbol": "688981",
    "bar_minutes": 5,
    "horizon": 3,
    "lookback": 32,
    "commission_rate": 0.001,
    "slippage_rate": 0.0005,
    "tp": 0.003,
    "sl": 0.002,
    "same_bar_policy": "pessimistic",
    "timeout_policy": "final_return",
    "target_col": "buy_label",
    "sell_target_col": "sell_label",
    "buy_label_col": "buy_label",
    "sell_label_col": "sell_label",
    "label_mode": "path_dependent_opportunity",
    "drop_incomplete_labels": True,
    "num_classes": 2,
    "buy_class": 1,
    "fixed_threshold_quantile": 0.95,
    "sequence_mode": "continuous",
    "train_ratio": 0.70,
    "valid_ratio": 0.15,
    "winsor_lower_q": 0.001,
    "winsor_upper_q": 0.999,
    "batch_size_train": 128,
    "batch_size_eval": 256,
    "lstm_hidden_dim": 256,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.3,
    "cnn_channels": 64,
    "cnn_kernel_size": 3,
    "mlp_hidden_dim": 256,
    "mlp_dropout": 0.3,
    "learning_rate": 5e-4,
    "weight_decay": 1e-6,
    "max_epochs": 100,
    "early_stop_patience": 10,
    "grad_clip_norm": 1.0,
    "rf_n_estimators": 300,
    "rf_max_depth": 8,
    "rf_min_samples_leaf": 50,
    "random_state": 42,
    "diagnostics_dir": "outputs/diagnostics",
    "vnpy_signal_path": "mvp_pred_signal.csv",
}


def build_mvp_config(overrides: dict | None = None) -> dict:
    config = deepcopy(BASE_MVP_CONFIG)
    if overrides:
        config.update(overrides)

    one_side_cost = config["commission_rate"] + config["slippage_rate"]
    config["one_side_cost"] = one_side_cost
    config["round_trip_cost"] = 2 * one_side_cost
    config["label_cost"] = config.get("label_cost") or one_side_cost
    config["label_threshold"] = config["label_cost"]
    config["total_cost"] = one_side_cost

    config["diagnostics_dir"] = str(Path(config["diagnostics_dir"]))
    return config


MVP_CONFIG = build_mvp_config()


from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from project_paths import PROJECT_ROOT


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
    "transformer_d_model": 128,
    "transformer_nhead": 4,
    "transformer_num_layers": 2,
    "transformer_dropout": 0.2,
    "transformer_lstm_hidden_dim": 128,
    "transformer_lstm_num_layers": 1,
    "cnn_channels": 64,
    "cnn_kernel_size": 3,
    "cnn_dropout": 0.3,
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
    "rf_max_features": "sqrt",
    "rf_class_weight": "balanced_subsample",
    "rf_n_jobs": -1,
    "logit_max_iter": 2000,
    "logit_c": 1.0,
    "logit_class_weight": "balanced",
    "sequence_signal_models": ["transformer_lstm", "lstm", "cnn", "mlp"],
    "primary_signal_model": "transformer_lstm",
    "use_ensemble_primary": True,
    "run_explainability": False,
    "explainability_top_features": 30,
    "permutation_repeats": 3,
    "shap_max_samples": 512,
    "gradient_max_samples": 1024,
    "walk_forward_train_bars": 12000,
    "walk_forward_valid_bars": 2400,
    "walk_forward_step_bars": 2400,
    "walk_forward_calibration_ratio": 0.2,
    "walk_forward_model_names": ["logistic_regression", "random_forest"],
    "walk_forward_max_folds": None,
    "run_walk_forward_in_notebook": False,
    "optuna_n_trials": 20,
    "optuna_timeout_seconds": None,
    "quality_min_folds": 3,
    "quality_min_total_trades": 30,
    "quality_min_median_test_buy_auc": 0.52,
    "quality_min_median_alpha_return": 0.0,
    "quality_max_median_drawdown": -0.08,
    "live_min_folds": 8,
    "live_min_total_trades": 200,
    "live_min_buy_auc": 0.53,
    "live_min_sell_auc": 0.53,
    "live_max_median_drawdown": -0.05,
    "stress_slippage_multipliers": [1.0, 1.5, 2.0],
    "stress_commission_multipliers": [1.0, 1.5],
    "stress_participation_rates": [0.02, 0.05],
    "require_shadow_monitoring": True,
    "shadow_min_days": 20,
    "shadow_min_trades": 30,
    "shadow_min_alpha_return": 0.0,
    "shadow_max_drawdown": -0.03,
    "shadow_max_daily_loss": -0.01,
    "shadow_max_consecutive_losses": 5,
    "shadow_min_win_rate": 0.45,
    "drift_max_feature_psi": 0.20,
    "drift_max_probability_psi": 0.20,
    "require_probability_drift": True,
    "expected_bars_per_day": 48,
    "min_bars_per_day": 40,
    "drop_entire_invalid_days": True,
    "max_zero_volume_bar_ratio": 0.001,
    "max_outside_session_bar_ratio": 0.0,
    "max_short_day_ratio": 0.02,
    "a_share_price_limit_rate": 0.20,
    "price_limit_tolerance": 0.005,
    "price_limit_check_max_gap_days": 3,
    "max_extreme_daily_move_count": 0,
    "risk_free_rate": 0.0,
    "initial_capital": 1_000_000,
    "max_position_pct": 1.0,
    "min_position_pct": 0.25,
    "position_sizing_mode": "probability_scaled",
    "position_sizing_high_quantile": 0.99,
    "min_lot_size": 100,
    "base_position_pct": 0.6,
    "max_t0_trade_pct": 0.3,
    "max_participation_rate": 0.05,
    "a_share_t0_mode": "inventory",
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

    diagnostics_dir = Path(config["diagnostics_dir"])
    if not diagnostics_dir.is_absolute():
        diagnostics_dir = PROJECT_ROOT / diagnostics_dir
    config["diagnostics_dir"] = str(diagnostics_dir)
    return config


MVP_CONFIG = build_mvp_config()

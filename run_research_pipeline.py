from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build_feature_set, select_feature_columns
from feature_engineering import make_sequence_data, split_time_series, standardize_by_train, winsorize_by_train
from label_builder import build_path_dependent_opportunity_labels, summarize_opportunity_labels
from execution_audit import audit_execution_feasibility, clean_market_data_for_execution
from hyperparameter_optimization import run_optuna_sequence_search, run_walk_forward_config_search
from mvp_config import build_mvp_config
from production_readiness import (
    audit_feature_leakage,
    build_live_readiness_report,
    collect_reproducibility_manifest,
)
from research_report import export_walk_forward_markdown_report
from model_signals import (
    train_dual_cnn_signals,
    train_dual_lstm_signals,
    train_dual_mlp_signals,
    train_dual_transformer_lstm_signals,
)
from vnpy_backtest import export_dual_signal_file, run_vnpy_dual_signal_t0_backtest, check_vnpy_available
from walk_forward_runner import run_walk_forward_signal_research
from project_paths import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, display_project_path, resolve_project_path


NUMERIC_OHLCV_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_pipeline_status(
    output_dir: Path,
    stage: str,
    status: str,
    details: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "status": status,
        "updated_at": pd.Timestamp.now().isoformat(),
        "details": details or {},
    }
    with (output_dir / "pipeline_status.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def write_final_run_summary(
    output_dir: Path,
    args: argparse.Namespace,
    config: dict,
    prepared_rows: int,
    feature_count: int,
    walk_forward_result: dict | None = None,
    readiness_report: dict | None = None,
    vnpy_report: dict | None = None,
    optuna_report: dict | None = None,
    report_path: Path | None = None,
) -> dict:
    summary = {
        "status": "COMPLETED",
        "completed_at": pd.Timestamp.now().isoformat(),
        "output_dir": display_project_path(output_dir),
        "data_path": display_project_path(args.data_path),
        "prepared_rows": int(prepared_rows),
        "feature_count": int(feature_count),
        "run_options": {
            "run_walk_forward": bool(args.run_walk_forward),
            "full_model_suite": bool(args.full_model_suite),
            "run_explainability": bool(args.run_explainability),
            "run_vnpy_backtest": bool(args.run_vnpy_backtest),
            "run_optuna": bool(args.run_optuna),
            "run_wf_optuna": bool(args.run_wf_optuna),
            "max_folds": args.max_folds,
            "device": args.device,
            "model_names": args.model_names,
        },
        "config_core": {
            "feature_engineering_set": config.get("feature_engineering_set"),
            "lookback": config.get("lookback"),
            "horizon": config.get("horizon"),
            "tp": config.get("tp"),
            "sl": config.get("sl"),
            "commission_rate": config.get("commission_rate"),
            "slippage_rate": config.get("slippage_rate"),
            "fixed_threshold_quantile": config.get("fixed_threshold_quantile"),
            "buy_threshold_quantile": config.get("buy_threshold_quantile"),
            "sell_threshold_quantile": config.get("sell_threshold_quantile"),
            "trade_direction_mode": config.get("trade_direction_mode"),
            "run_sell_threshold_grid": config.get("run_sell_threshold_grid"),
        },
        "walk_forward_summary": walk_forward_result.get("aggregate_summary") if walk_forward_result else None,
        "live_readiness": readiness_report,
        "vnpy_backtest": vnpy_report,
        "optuna": optuna_report,
        "walk_forward_optuna": config.get("walk_forward_optuna_report"),
        "walk_forward_report": display_project_path(report_path) if report_path else None,
        "progress_files": {
            "pipeline_status": display_project_path(output_dir / "pipeline_status.json"),
            "walk_forward_progress": display_project_path(output_dir / "walk_forward" / "walk_forward_progress.json"),
            "walk_forward_progress_csv": display_project_path(output_dir / "walk_forward" / "walk_forward_progress.csv"),
        },
    }
    with (output_dir / "final_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 688981 T+0 research pipeline.")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Input 5-minute OHLCV parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Pipeline output directory.",
    )
    parser.add_argument(
        "--run-walk-forward",
        action="store_true",
        help="Run walk-forward model training and T+0 research backtest.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small windows and lightweight models for a fast smoke run.",
    )
    parser.add_argument(
        "--save-prepared-data",
        action="store_true",
        help="Save the fully prepared feature/label dataframe as parquet.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help="Model names for walk-forward. Main suite: transformer_lstm lstm cnn mlp.",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        choices=["basic", "ta", "basic_ta"],
        help="Feature engineering set: built-in basic features, TA-library features, or both.",
    )
    parser.add_argument("--max-folds", type=int, default=None, help="Limit walk-forward folds.")
    parser.add_argument("--device", default=None, help="Torch device for deep sequence models.")
    parser.add_argument(
        "--full-model-suite",
        action="store_true",
        help="Run Transformer-LSTM, LSTM, CNN, MLP, random forest, and logistic regression.",
    )
    parser.add_argument(
        "--run-optuna",
        action="store_true",
        help="Run optional Optuna hyperparameter search before walk-forward.",
    )
    parser.add_argument(
        "--run-wf-optuna",
        action="store_true",
        help="Run Optuna search over walk-forward trading objective and apply the best config to the final run.",
    )
    parser.add_argument(
        "--optuna-model",
        default="transformer_lstm",
        choices=["transformer_lstm", "lstm", "cnn", "mlp"],
        help="Sequence model optimized by Optuna.",
    )
    parser.add_argument("--optuna-trials", type=int, default=None, help="Override Optuna trial count.")
    parser.add_argument("--wf-optuna-trials", type=int, default=None, help="Override walk-forward Optuna trial count.")
    parser.add_argument("--wf-optuna-timeout", type=int, default=None, help="Optional walk-forward Optuna timeout in seconds.")
    parser.add_argument(
        "--run-explainability",
        action="store_true",
        help="Export permutation, SHAP when available, and gradient importance inside walk-forward folds.",
    )
    parser.add_argument(
        "--run-vnpy-backtest",
        action="store_true",
        help="Run optional vn.py event-driven backtest from selected walk-forward signals.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full JSON reports to terminal. By default, full reports are written to files.",
    )
    return parser.parse_args()


def load_market_data(path: str | Path) -> pd.DataFrame:
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for col in NUMERIC_OHLCV_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_pipeline_config(args: argparse.Namespace) -> dict:
    overrides = {}
    if args.quick:
        overrides.update(
            {
                "lookback": 8,
                "walk_forward_train_bars": 800,
                "walk_forward_valid_bars": 160,
                "walk_forward_step_bars": 160,
                "walk_forward_calibration_ratio": 0.25,
                "walk_forward_max_folds": args.max_folds or 2,
                "rf_n_estimators": 50,
                "rf_max_depth": 5,
                "rf_min_samples_leaf": 5,
                "logit_max_iter": 500,
                "max_epochs": 3,
                "early_stop_patience": 2,
                "optuna_n_trials": 2,
                "quality_min_folds": 1,
                "quality_min_total_trades": 1,
            }
        )
    if args.max_folds is not None:
        overrides["walk_forward_max_folds"] = args.max_folds
    if args.model_names:
        overrides["walk_forward_model_names"] = args.model_names
    if args.feature_set:
        overrides["feature_engineering_set"] = args.feature_set
    if args.full_model_suite:
        overrides["walk_forward_model_names"] = [
            "transformer_lstm",
            "lstm",
            "cnn",
            "mlp",
        ]
    if args.run_explainability:
        overrides["run_explainability"] = True
    if args.optuna_trials is not None:
        overrides["optuna_n_trials"] = args.optuna_trials
    if args.wf_optuna_trials is not None:
        overrides["wf_optuna_n_trials"] = args.wf_optuna_trials
    if args.wf_optuna_timeout is not None:
        overrides["wf_optuna_timeout_seconds"] = args.wf_optuna_timeout
    config = build_mvp_config(overrides)
    config["diagnostics_dir"] = str(Path(args.output_dir))
    return config


def run_optuna_from_pipeline(
    data: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
    model_name: str,
    output_dir: Path,
    device: str | None = None,
):
    train_df, valid_df, test_df = split_time_series(
        data,
        train_ratio=float(config.get("train_ratio", 0.70)),
        valid_ratio=float(config.get("valid_ratio", 0.15)),
    )
    buy_target_col = config.get("buy_label_col", "buy_label")
    sell_target_col = config.get("sell_label_col", "sell_label")
    required = feature_cols + [buy_target_col, sell_target_col]
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    valid_df = valid_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    train_w, valid_w, test_w, _ = winsorize_by_train(
        train_df,
        valid_df,
        test_df,
        feature_cols,
        lower_q=float(config.get("winsor_lower_q", 0.001)),
        upper_q=float(config.get("winsor_upper_q", 0.999)),
    )
    train_scaled, valid_scaled, test_scaled, _ = standardize_by_train(train_w, valid_w, test_w, feature_cols)

    lookback = int(config.get("lookback", 32))
    x_train_seq, y_train_buy_seq, train_index_seq = make_sequence_data(train_scaled, feature_cols, buy_target_col, lookback)
    x_valid_seq, y_valid_buy_seq, valid_index_seq = make_sequence_data(valid_scaled, feature_cols, buy_target_col, lookback)
    x_test_seq, y_test_buy_seq, test_index_seq = make_sequence_data(test_scaled, feature_cols, buy_target_col, lookback)
    _, y_train_sell_seq, _ = make_sequence_data(train_scaled, feature_cols, sell_target_col, lookback)
    _, y_valid_sell_seq, _ = make_sequence_data(valid_scaled, feature_cols, sell_target_col, lookback)
    _, y_test_sell_seq, _ = make_sequence_data(test_scaled, feature_cols, sell_target_col, lookback)

    trainers = {
        "transformer_lstm": train_dual_transformer_lstm_signals,
        "lstm": train_dual_lstm_signals,
        "cnn": train_dual_cnn_signals,
        "mlp": train_dual_mlp_signals,
    }
    trainer = trainers[model_name]
    trainer_kwargs = {
        "x_train_seq": x_train_seq,
        "y_train_buy_seq": y_train_buy_seq,
        "y_train_sell_seq": y_train_sell_seq,
        "x_valid_seq": x_valid_seq,
        "y_valid_buy_seq": y_valid_buy_seq,
        "y_valid_sell_seq": y_valid_sell_seq,
        "x_test_seq": x_test_seq,
        "y_test_buy_seq": y_test_buy_seq,
        "y_test_sell_seq": y_test_sell_seq,
        "valid_df": valid_scaled,
        "test_df": test_scaled,
        "valid_index_seq": valid_index_seq,
        "test_index_seq": test_index_seq,
        "device": device,
    }
    optuna_dir = output_dir / model_name
    try:
        study = run_optuna_sequence_search(
            trainer=trainer,
            model_name=model_name,
            base_config=config,
            trainer_kwargs=trainer_kwargs,
            output_dir=optuna_dir,
            n_trials=int(config.get("optuna_n_trials", 20)),
            timeout=config.get("optuna_timeout_seconds"),
        )
    except ImportError as exc:
        optuna_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "SKIPPED",
            "model_name": model_name,
            "reason": str(exc),
            "install_hint": "pip install optuna",
        }
        with (optuna_dir / "optuna_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
        return report
    report = {
        "status": "PASS",
        "model_name": model_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_trial_number": int(study.best_trial.number),
    }
    with (optuna_dir / "optuna_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    return report


def run_walk_forward_optuna_from_pipeline(
    data: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
    output_dir: Path,
    device: str | None = None,
) -> dict:
    optuna_dir = output_dir / "walk_forward_optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(config.get("walk_forward_model_names", ["transformer_lstm", "lstm", "cnn", "mlp"]))

    def evaluator(trial_config: dict, trial_output: Path) -> dict:
        trial_config = dict(trial_config)
        trial_config["run_explainability"] = False
        return run_walk_forward_signal_research(
            data=data,
            feature_cols=feature_cols,
            config=trial_config,
            output_dir=trial_output,
            model_names=model_names,
            device=device,
        )

    try:
        result = run_walk_forward_config_search(
            evaluator=evaluator,
            base_config=config,
            model_names=model_names,
            output_dir=optuna_dir,
            n_trials=int(config.get("wf_optuna_n_trials", config.get("optuna_n_trials", 20))),
            timeout=config.get("wf_optuna_timeout_seconds", config.get("optuna_timeout_seconds")),
        )
    except ImportError as exc:
        report = {
            "status": "SKIPPED",
            "reason": str(exc),
            "install_hint": "pip install optuna",
        }
        with (optuna_dir / "walk_forward_optuna_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
        return {"report": report, "best_config": config}

    report = dict(result["report"])
    report["output_dir"] = display_project_path(result["output_dir"])
    with (optuna_dir / "walk_forward_optuna_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    return {"report": report, "best_config": result["best_config"]}


def run_optional_vnpy_stage(
    walk_forward_result: dict,
    config: dict,
    output_dir: Path,
) -> dict:
    vnpy_dir = output_dir / "vnpy_backtest"
    vnpy_dir.mkdir(parents=True, exist_ok=True)
    available, vnpy_info = check_vnpy_available()
    report = {"available": bool(available), "vnpy_info": {k: str(v) for k, v in vnpy_info.items() if k not in {"Exchange", "Interval", "BarData", "get_database"}}}
    if not available:
        report["status"] = "SKIPPED"
        report["reason"] = "vn.py core or vnpy_ctastrategy is unavailable in this environment"
        with (vnpy_dir / "vnpy_backtest_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
        return report

    fold_summary = walk_forward_result["fold_summary"]
    if not len(fold_summary):
        report.update({"status": "SKIPPED", "reason": "no walk-forward folds"})
        with (vnpy_dir / "vnpy_backtest_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
        return report

    best_row = fold_summary.sort_values("alpha_total_return", ascending=False).iloc[0]
    fold_dir = resolve_project_path(best_row["output_dir"])
    model_name = str(best_row["selected_model"])
    signal_path = fold_dir / "models" / model_name / "test_signals.csv"
    if not signal_path.exists():
        report.update({"status": "SKIPPED", "reason": f"missing selected signal file: {signal_path}"})
        with (vnpy_dir / "vnpy_backtest_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
        return report

    signal_df = pd.read_csv(signal_path, index_col=0, parse_dates=True)
    exported_signal_path, exported_signal = export_dual_signal_file(
        signal_df,
        vnpy_dir / "dual_signal.csv",
        buy_prob_col="buy_prob",
        sell_prob_col="sell_prob",
    )
    engine, daily_result, stats = run_vnpy_dual_signal_t0_backtest(
        vt_symbol=f"{config.get('symbol', '688981')}.SSE",
        signal_path=exported_signal_path,
        signal_df=signal_df,
        buy_threshold=float(best_row["buy_threshold"]),
        sell_threshold=float(best_row["sell_threshold"]),
        vnpy_info=vnpy_info,
        fixed_size=int(config.get("min_lot_size", 100)),
        capital=float(config.get("initial_capital", 1_000_000)),
        commission_rate=float(config.get("commission_rate", 0.001)),
        slippage_abs=float(config.get("slippage_abs", 0.05)),
        stop_loss_pct=float(config.get("sl", 0.002)),
        take_profit_pct=float(config.get("tp", 0.003)),
    )
    if len(daily_result):
        daily_result.to_csv(vnpy_dir / "vnpy_daily_result.csv")
    report.update(
        {
            "status": "PASS" if stats else "SKIPPED",
            "selected_fold": int(best_row["fold"]),
            "selected_model": model_name,
            "signal_file": str(exported_signal_path),
            "signal_count": int(len(exported_signal)),
            "stats": stats,
        }
    )
    with (vnpy_dir / "vnpy_backtest_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    return report


def run_optional_vnpy_stage_with_log(
    walk_forward_result: dict,
    config: dict,
    output_dir: Path,
) -> dict:
    """Run vn.py while capturing its noisy console progress into a log file."""
    vnpy_dir = output_dir / "vnpy_backtest"
    vnpy_dir.mkdir(parents=True, exist_ok=True)
    log_path = vnpy_dir / "vnpy_console.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            report = run_optional_vnpy_stage(walk_forward_result, config, output_dir)
    if "signal_file" in report:
        report["signal_file"] = display_project_path(report["signal_file"])
    report["console_log"] = display_project_path(log_path)
    with (vnpy_dir / "vnpy_backtest_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    return report


def print_compact_run_summary(
    output_dir: Path,
    prepared_rows: int,
    feature_count: int,
    walk_forward_result: dict | None,
    readiness_report: dict | None,
    vnpy_report: dict | None,
) -> None:
    print("Pipeline finished.")
    print(f"Output directory: {display_project_path(output_dir)}")
    print(f"Final summary: {display_project_path(output_dir / 'final_run_summary.json')}")
    print(f"Status file: {display_project_path(output_dir / 'pipeline_status.json')}")
    print(f"Prepared rows: {prepared_rows}, features: {feature_count}")
    if walk_forward_result:
        summary = walk_forward_result["aggregate_summary"]
        print(
            "Walk-forward: "
            f"folds={summary.get('executed_fold_count')}, "
            f"trades={summary.get('total_trades')}, "
            f"median_alpha={summary.get('median_alpha_total_return'):.6f}, "
            f"median_buy_auc={summary.get('median_test_buy_auc'):.4f}, "
            f"median_sell_auc={summary.get('median_test_sell_auc'):.4f}"
        )
        print(f"Walk-forward progress: {display_project_path(output_dir / 'walk_forward' / 'walk_forward_progress.json')}")
        print(f"Walk-forward report: {display_project_path(output_dir / 'walk_forward' / 'walk_forward_report.md')}")
    if readiness_report:
        print(f"Live readiness: {readiness_report.get('status')}")
    if vnpy_report:
        print(f"vn.py backtest: {vnpy_report.get('status')}, log={vnpy_report.get('console_log')}")


def prepare_research_dataset(raw_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str], dict]:
    data = build_feature_set(raw_df, config)
    data = build_path_dependent_opportunity_labels(
        data,
        horizon=int(config["horizon"]),
        tp=float(config["tp"]),
        sl=float(config["sl"]),
        cost=float(config["label_cost"]),
        same_bar_policy=config["same_bar_policy"],
        timeout_policy=config["timeout_policy"],
        buy_label_col=config["buy_label_col"],
        sell_label_col=config["sell_label_col"],
    )
    if config.get("drop_incomplete_labels", True):
        data = data.dropna(subset=["future_entry_open", "future_close"]).copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    feature_cols, feature_report = select_feature_columns(data)
    required = feature_cols + [
        config["buy_label_col"],
        config["sell_label_col"],
        "open",
        "close",
        "trade_return",
        "future_return",
    ]
    data = data.dropna(subset=required).copy()
    label_summary, label_reason_distribution = summarize_opportunity_labels(
        data,
        buy_label_col=config["buy_label_col"],
        sell_label_col=config["sell_label_col"],
    )
    metadata = {
        "feature_engineering_set": config.get("feature_engineering_set", "basic"),
        "feature_report": feature_report,
        "label_summary": label_summary.to_dict(orient="records"),
        "label_reason_distribution": label_reason_distribution.to_dict(orient="records"),
    }
    return data, feature_cols, metadata


def export_pipeline_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    config: dict,
    raw_df: pd.DataFrame,
    market_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    feature_cols: list[str],
    metadata: dict,
    cleaning_report: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(feature_cols, name="feature").to_csv(output_dir / "feature_cols.csv", index=False)
    with open(output_dir / "feature_engineering_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_engineering_set": metadata.get("feature_engineering_set"),
                "feature_report": metadata.get("feature_report"),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    pd.DataFrame(metadata["label_summary"]).to_csv(output_dir / "label_summary.csv", index=False)
    pd.DataFrame(metadata["label_reason_distribution"]).to_csv(
        output_dir / "label_reason_distribution.csv", index=False
    )
    leakage_audit = audit_feature_leakage(feature_cols, output_dir)
    execution_audit = audit_execution_feasibility(market_df, config, output_dir)
    reproducibility_manifest = collect_reproducibility_manifest(
        data_path=args.data_path,
        config=config,
        output_dir=output_dir,
        extra_files=["production_readiness.py", "research_report.py", "execution_audit.py"],
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=_json_default)
    manifest = {
        "data_path": str(args.data_path),
        "raw_shape": list(raw_df.shape),
        "cleaned_market_shape": list(market_df.shape),
        "prepared_shape": list(prepared_df.shape),
        "index_start": str(prepared_df.index.min()),
        "index_end": str(prepared_df.index.max()),
        "feature_count": len(feature_cols),
        "feature_engineering_set": metadata.get("feature_engineering_set"),
        "run_walk_forward": bool(args.run_walk_forward),
        "quick": bool(args.quick),
        "feature_leakage_audit": leakage_audit,
        "market_data_cleaning_report": cleaning_report,
        "execution_feasibility_audit": execution_audit,
        "reproducibility_manifest": {
            "data_sha256": reproducibility_manifest.get("data_sha256"),
            "config_sha256": reproducibility_manifest.get("config_sha256"),
            "source_bundle_sha256": reproducibility_manifest.get("source_bundle_sha256"),
            "git_commit": reproducibility_manifest.get("git_commit"),
            "git_branch": reproducibility_manifest.get("git_branch"),
            "git_dirty": reproducibility_manifest.get("git_dirty"),
        },
        "methodology": {
            "numeric_conversion": "open/high/low/close/volume/amount are coerced to numeric before feature engineering",
            "feature_engineering": "feature_engineering.build_feature_set and select_feature_columns",
            "labeling": "path-dependent buy/sell opportunity labels with tp/sl/cost from MVP_CONFIG",
            "validation": "optional walk-forward subtrain/calibration/out-of-sample test via walk_forward_runner",
        },
    }
    with open(output_dir / "pipeline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=_json_default)


def main() -> None:
    args = parse_args()
    args.data_path = str(resolve_project_path(args.data_path))
    args.output_dir = str(resolve_project_path(args.output_dir))
    output_dir = Path(args.output_dir)
    output_dir = resolve_project_path(output_dir)
    config = build_pipeline_config(args)
    write_pipeline_status(
        output_dir,
        stage="start",
        status="RUNNING",
        details={
            "data_path": display_project_path(args.data_path),
            "output_dir": display_project_path(output_dir),
            "run_walk_forward": bool(args.run_walk_forward),
            "run_explainability": bool(args.run_explainability),
            "run_vnpy_backtest": bool(args.run_vnpy_backtest),
            "run_optuna": bool(args.run_optuna),
            "run_wf_optuna": bool(args.run_wf_optuna),
        },
    )

    write_pipeline_status(output_dir, stage="load_and_prepare_data", status="RUNNING")
    raw_df = load_market_data(args.data_path)
    market_df, cleaning_report = clean_market_data_for_execution(raw_df, config, output_dir)
    prepared_df, feature_cols, metadata = prepare_research_dataset(market_df, config)
    export_pipeline_manifest(output_dir, args, config, raw_df, market_df, prepared_df, feature_cols, metadata, cleaning_report)
    if args.save_prepared_data:
        prepared_df.to_parquet(output_dir / "prepared_research_dataset.parquet")
    write_pipeline_status(
        output_dir,
        stage="load_and_prepare_data",
        status="COMPLETED",
        details={"prepared_rows": int(len(prepared_df)), "feature_count": int(len(feature_cols))},
    )

    optuna_report = None
    if args.run_optuna:
        write_pipeline_status(output_dir, stage="optuna", status="RUNNING")
        optuna_report = run_optuna_from_pipeline(
            data=prepared_df,
            feature_cols=feature_cols,
            config=config,
            model_name=args.optuna_model,
            output_dir=output_dir / "optuna",
            device=args.device,
        )
        write_pipeline_status(output_dir, stage="optuna", status="COMPLETED", details=optuna_report)

    wf_optuna_report = None
    if args.run_wf_optuna:
        write_pipeline_status(output_dir, stage="walk_forward_optuna", status="RUNNING")
        wf_optuna_result = run_walk_forward_optuna_from_pipeline(
            data=prepared_df,
            feature_cols=feature_cols,
            config=config,
            output_dir=output_dir,
            device=args.device,
        )
        wf_optuna_report = wf_optuna_result["report"]
        config.update(wf_optuna_result["best_config"])
        config["walk_forward_optuna_report"] = wf_optuna_report
        with (output_dir / "applied_walk_forward_optuna_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=_json_default)
        write_pipeline_status(output_dir, stage="walk_forward_optuna", status="COMPLETED", details=wf_optuna_report)

    walk_forward_result = None
    readiness_report = None
    vnpy_report = None
    report_path = None
    if args.run_walk_forward:
        write_pipeline_status(
            output_dir,
            stage="walk_forward",
            status="RUNNING",
            details={"progress_file": display_project_path(output_dir / "walk_forward" / "walk_forward_progress.json")},
        )
        walk_forward_result = run_walk_forward_signal_research(
            data=prepared_df,
            feature_cols=feature_cols,
            config=config,
            output_dir=output_dir / "walk_forward",
            model_names=config.get("walk_forward_model_names"),
            device=args.device,
        )
        write_pipeline_status(
            output_dir,
            stage="walk_forward",
            status="COMPLETED",
            details=walk_forward_result["aggregate_summary"],
        )
        with open(output_dir / "reproducibility_manifest.json", "r", encoding="utf-8") as f:
            reproducibility_manifest = json.load(f)
        with open(output_dir / "feature_leakage_audit.json", "r", encoding="utf-8") as f:
            leakage_audit = json.load(f)
        readiness_report = build_live_readiness_report(
            walk_forward_dir=walk_forward_result["output_dir"],
            leakage_audit=leakage_audit,
            reproducibility_manifest=reproducibility_manifest,
            config=config,
            output_dir=output_dir,
        )
        report_path = export_walk_forward_markdown_report(walk_forward_result["output_dir"])
        if args.run_vnpy_backtest:
            write_pipeline_status(output_dir, stage="vnpy_backtest", status="RUNNING")
            vnpy_report = run_optional_vnpy_stage_with_log(walk_forward_result, config, output_dir)
            write_pipeline_status(output_dir, stage="vnpy_backtest", status="COMPLETED", details=vnpy_report)
        if args.verbose:
            if vnpy_report:
                print("vn.py backtest:")
                print(json.dumps(vnpy_report, ensure_ascii=False, indent=2, default=_json_default))
            print("Walk-forward summary:")
            print(json.dumps(walk_forward_result["aggregate_summary"], ensure_ascii=False, indent=2, default=_json_default))
            print(f"Walk-forward report: {display_project_path(report_path)}")
            print("Live readiness:")
            print(json.dumps(readiness_report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print("Prepared research dataset only. Add --run-walk-forward to train and backtest.")

    final_summary = write_final_run_summary(
        output_dir=output_dir,
        args=args,
        config=config,
        prepared_rows=len(prepared_df),
        feature_count=len(feature_cols),
        walk_forward_result=walk_forward_result,
        readiness_report=readiness_report,
        vnpy_report=vnpy_report,
        optuna_report=optuna_report,
        report_path=report_path,
    )
    write_pipeline_status(
        output_dir,
        stage="completed",
        status="COMPLETED",
        details={
            "final_run_summary": display_project_path(output_dir / "final_run_summary.json"),
            "live_readiness_status": readiness_report.get("status") if readiness_report else None,
        },
    )
    if not args.verbose:
        print_compact_run_summary(
            output_dir=output_dir,
            prepared_rows=len(prepared_df),
            feature_count=len(feature_cols),
            walk_forward_result=walk_forward_result,
            readiness_report=readiness_report,
            vnpy_report=vnpy_report,
        )
    else:
        print(f"Output directory: {display_project_path(output_dir)}")
        print(f"Final summary: {display_project_path(output_dir / 'final_run_summary.json')}")
        print(f"Prepared rows: {len(prepared_df)}, features: {len(feature_cols)}")


if __name__ == "__main__":
    main()

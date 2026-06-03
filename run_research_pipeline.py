from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build_basic_features, select_feature_columns
from label_builder import build_path_dependent_opportunity_labels, summarize_opportunity_labels
from mvp_config import build_mvp_config
from research_report import export_walk_forward_markdown_report
from walk_forward_runner import run_walk_forward_signal_research


NUMERIC_OHLCV_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 688981 T+0 research pipeline.")
    parser.add_argument(
        "--data-path",
        default="688981_5min_20200716-20260602.parquet",
        help="Input 5-minute OHLCV parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics/research_pipeline",
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
        help="Model names for walk-forward, e.g. logistic_regression random_forest transformer_lstm.",
    )
    parser.add_argument("--max-folds", type=int, default=None, help="Limit walk-forward folds.")
    parser.add_argument("--device", default=None, help="Torch device for deep sequence models.")
    return parser.parse_args()


def load_market_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
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
                "quality_min_folds": 1,
                "quality_min_total_trades": 1,
            }
        )
    if args.max_folds is not None:
        overrides["walk_forward_max_folds"] = args.max_folds
    if args.model_names:
        overrides["walk_forward_model_names"] = args.model_names
    config = build_mvp_config(overrides)
    config["diagnostics_dir"] = str(Path(args.output_dir))
    return config


def prepare_research_dataset(raw_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str], dict]:
    data = build_basic_features(raw_df)
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
    prepared_df: pd.DataFrame,
    feature_cols: list[str],
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(feature_cols, name="feature").to_csv(output_dir / "feature_cols.csv", index=False)
    pd.DataFrame(metadata["label_summary"]).to_csv(output_dir / "label_summary.csv", index=False)
    pd.DataFrame(metadata["label_reason_distribution"]).to_csv(
        output_dir / "label_reason_distribution.csv", index=False
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=_json_default)
    manifest = {
        "data_path": str(args.data_path),
        "raw_shape": list(raw_df.shape),
        "prepared_shape": list(prepared_df.shape),
        "index_start": str(prepared_df.index.min()),
        "index_end": str(prepared_df.index.max()),
        "feature_count": len(feature_cols),
        "run_walk_forward": bool(args.run_walk_forward),
        "quick": bool(args.quick),
        "methodology": {
            "numeric_conversion": "open/high/low/close/volume/amount are coerced to numeric before feature engineering",
            "feature_engineering": "feature_engineering.build_basic_features and select_feature_columns",
            "labeling": "path-dependent buy/sell opportunity labels with tp/sl/cost from MVP_CONFIG",
            "validation": "optional walk-forward subtrain/calibration/out-of-sample test via walk_forward_runner",
        },
    }
    with open(output_dir / "pipeline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=_json_default)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = build_pipeline_config(args)

    raw_df = load_market_data(args.data_path)
    prepared_df, feature_cols, metadata = prepare_research_dataset(raw_df, config)
    export_pipeline_manifest(output_dir, args, config, raw_df, prepared_df, feature_cols, metadata)
    if args.save_prepared_data:
        prepared_df.to_parquet(output_dir / "prepared_research_dataset.parquet")

    if args.run_walk_forward:
        walk_forward_result = run_walk_forward_signal_research(
            data=prepared_df,
            feature_cols=feature_cols,
            config=config,
            output_dir=output_dir / "walk_forward",
            model_names=config.get("walk_forward_model_names"),
            device=args.device,
        )
        report_path = export_walk_forward_markdown_report(walk_forward_result["output_dir"])
        print("Walk-forward summary:")
        print(json.dumps(walk_forward_result["aggregate_summary"], ensure_ascii=False, indent=2, default=_json_default))
        print(f"Walk-forward report: {report_path}")
    else:
        print("Prepared research dataset only. Add --run-walk-forward to train and backtest.")

    print(f"Output directory: {output_dir}")
    print(f"Prepared rows: {len(prepared_df)}, features: {len(feature_cols)}")


if __name__ == "__main__":
    main()

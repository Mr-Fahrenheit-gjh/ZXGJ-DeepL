from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from project_paths import PROJECT_ROOT, display_project_path, resolve_project_path


DEFAULT_MARKET_GRID = [
    {"lookback": 8, "horizon": 3, "tp": 0.003, "sl": 0.002},
    {"lookback": 16, "horizon": 3, "tp": 0.003, "sl": 0.002},
    {"lookback": 24, "horizon": 3, "tp": 0.003, "sl": 0.002},
    {"lookback": 32, "horizon": 3, "tp": 0.003, "sl": 0.002},
    {"lookback": 16, "horizon": 6, "tp": 0.004, "sl": 0.002},
    {"lookback": 24, "horizon": 6, "tp": 0.004, "sl": 0.003},
]

RISK_PROFILES = {
    "base": {},
    "cool6_loss2": {
        "stop_loss_cooldown_bars": 6,
        "daily_max_losses": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSTM-only parameter experiments for the 688981 T+0 pipeline.")
    parser.add_argument("--output-dir", default="outputs/diagnostics/lstm_param_experiments")
    parser.add_argument("--data-path", default="688981_5min_20200716-20260602.parquet")
    parser.add_argument("--feature-set", default="basic", choices=["basic", "ta", "basic_ta"])
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--train-bars", type=int, default=4000)
    parser.add_argument("--valid-bars", type=int, default=800)
    parser.add_argument("--step-bars", type=int, default=800)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true", help="Use pipeline quick mode, then override lookback/horizon/tp/sl.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def scenario_name(market_params: dict, risk_name: str) -> str:
    tp = str(market_params["tp"]).replace(".", "p")
    sl = str(market_params["sl"]).replace(".", "p")
    return (
        f"lb{market_params['lookback']}_h{market_params['horizon']}"
        f"_tp{tp}_sl{sl}_{risk_name}"
    )


def build_command(args: argparse.Namespace, scenario_dir: Path, market_params: dict, risk_params: dict) -> list[str]:
    cmd = [
        sys.executable,
        "run_research_pipeline.py",
        "--data-path",
        str(args.data_path),
        "--output-dir",
        str(scenario_dir),
        "--run-walk-forward",
        "--model-names",
        "lstm",
        "--feature-set",
        args.feature_set,
        "--max-folds",
        str(args.max_folds),
        "--max-epochs",
        str(args.max_epochs),
        "--train-bars",
        str(args.train_bars),
        "--valid-bars",
        str(args.valid_bars),
        "--step-bars",
        str(args.step_bars),
        "--lookback",
        str(market_params["lookback"]),
        "--horizon",
        str(market_params["horizon"]),
        "--tp",
        str(market_params["tp"]),
        "--sl",
        str(market_params["sl"]),
        "--trade-direction-mode",
        "sell_only",
        "--sell-threshold-quantile",
        "0.95",
    ]
    if args.quick:
        cmd.insert(2, "--quick")
    if args.device:
        cmd.extend(["--device", args.device])
    for key, value in risk_params.items():
        cli_key = "--" + key.replace("_", "-")
        cmd.extend([cli_key, str(value)])
    return cmd


def collect_summary(scenario_dir: Path, scenario: dict) -> dict:
    summary_path = scenario_dir / "final_run_summary.json"
    row = dict(scenario)
    row["output_dir"] = display_project_path(scenario_dir)
    row["status"] = "MISSING"
    if not summary_path.exists():
        return row
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    wf = payload.get("walk_forward_summary", {}) or {}
    live = payload.get("live_readiness", {}) or {}
    row.update(
        {
            "status": payload.get("status"),
            "live_status": live.get("status"),
            "executed_fold_count": wf.get("executed_fold_count"),
            "total_trades": wf.get("total_trades"),
            "mean_alpha_total_return": wf.get("mean_alpha_total_return"),
            "median_alpha_total_return": wf.get("median_alpha_total_return"),
            "mean_test_sell_auc": wf.get("mean_test_sell_auc"),
            "median_test_sell_auc": wf.get("median_test_sell_auc"),
            "mean_test_buy_auc": wf.get("mean_test_buy_auc"),
            "median_test_buy_auc": wf.get("median_test_buy_auc"),
            "stress_min_alpha_total_return": wf.get("stress_min_alpha_total_return"),
            "stress_worst_alpha_drawdown": wf.get("stress_worst_alpha_drawdown"),
            "quality_all_passed": (wf.get("quality_report") or {}).get("all_passed"),
        }
    )
    fold_summary_path = scenario_dir / "walk_forward" / "walk_forward_fold_summary.csv"
    if fold_summary_path.exists():
        fold_df = pd.read_csv(fold_summary_path)
        if "alpha_total_return" in fold_df:
            row["positive_alpha_folds"] = int((fold_df["alpha_total_return"] > 0).sum())
        if "trade_count" in fold_df:
            row["zero_trade_folds"] = int((fold_df["trade_count"] == 0).sum())
    return row


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    scenarios = []
    for market_params in DEFAULT_MARKET_GRID:
        for risk_name, risk_params in RISK_PROFILES.items():
            name = scenario_name(market_params, risk_name)
            scenario_dir = output_dir / name
            scenario = {
                "scenario": name,
                "risk_profile": risk_name,
                **market_params,
                **risk_params,
            }
            cmd = build_command(args, scenario_dir, market_params, risk_params)
            scenarios.append({"scenario": scenario, "scenario_dir": scenario_dir, "cmd": cmd})

    with (output_dir / "experiment_plan.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "scenario": item["scenario"],
                    "output_dir": display_project_path(item["scenario_dir"]),
                    "command": item["cmd"],
                }
                for item in scenarios
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    for idx, item in enumerate(scenarios, start=1):
        scenario = item["scenario"]
        scenario_dir = item["scenario_dir"]
        cmd = item["cmd"]
        print(f"[{idx}/{len(scenarios)}] {scenario['scenario']}")
        print(" ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
            scenario["returncode"] = result.returncode
        rows.append(collect_summary(scenario_dir, scenario))
        pd.DataFrame(rows).to_csv(output_dir / "lstm_param_experiment_summary.csv", index=False)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "lstm_param_experiment_summary.csv"
    summary.to_csv(summary_path, index=False)
    if len(summary):
        sort_cols = [c for c in ["median_alpha_total_return", "mean_alpha_total_return"] if c in summary]
        if sort_cols:
            summary = summary.sort_values(sort_cols, ascending=False)
        print("\nTop scenarios:")
        display_cols = [
            "scenario",
            "risk_profile",
            "lookback",
            "horizon",
            "tp",
            "sl",
            "median_alpha_total_return",
            "mean_alpha_total_return",
            "median_test_sell_auc",
            "total_trades",
            "positive_alpha_folds",
            "live_status",
        ]
        display_cols = [c for c in display_cols if c in summary.columns]
        print(summary[display_cols].head(12).to_string(index=False))
    print(f"\nSummary saved: {display_project_path(summary_path)}")


if __name__ == "__main__":
    main()

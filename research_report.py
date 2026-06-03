from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_walk_forward_summary(output_dir: str | Path) -> tuple[dict, pd.DataFrame]:
    output_dir = Path(output_dir)
    with open(output_dir / "walk_forward_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    fold_summary = pd.read_csv(output_dir / "walk_forward_fold_summary.csv")
    return summary, fold_summary


def build_walk_forward_markdown_report(output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    summary, fold_summary = load_walk_forward_summary(output_dir)
    quality = summary.get("quality_report", {})
    checks = quality.get("checks", {})
    observed = quality.get("observed", {})
    methodology = summary.get("methodology", {})

    lines = [
        "# Walk-forward Research Report",
        "",
        "## Executive Summary",
        "",
        f"- Executed folds: {summary.get('executed_fold_count')}",
        f"- Total trades: {summary.get('total_trades')}",
        f"- Median test buy AUC: {summary.get('median_test_buy_auc')}",
        f"- Median test sell AUC: {summary.get('median_test_sell_auc')}",
        f"- Median total return per fold: {summary.get('median_total_return')}",
        f"- Median alpha total return per fold: {summary.get('median_alpha_total_return')}",
        f"- Quality gate passed: {quality.get('all_passed')}",
        "",
        "## Quality Gate Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Observed Metrics", ""])
    for key, value in observed.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Methodology", ""])
    for key, value in methodology.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Fold Summary", ""])
    if len(fold_summary):
        display_cols = [
            "fold",
            "selected_model",
            "test_buy_auc",
            "test_sell_auc",
            "total_return",
            "alpha_total_return",
            "max_drawdown",
            "alpha_max_drawdown",
            "sharpe",
            "alpha_sharpe",
            "sortino",
            "alpha_sortino",
            "calmar",
            "alpha_calmar",
            "trade_count",
        ]
        available_cols = [col for col in display_cols if col in fold_summary.columns]
        lines.append(fold_summary[available_cols].to_markdown(index=False))
    else:
        lines.append("No folds executed.")
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "This report is a research artifact. A passing gate is not live-trading approval. "
            "Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, "
            "and cross-checking with VeighNa event-driven backtests.",
        ]
    )
    return "\n".join(lines) + "\n"


def export_walk_forward_markdown_report(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    report = build_walk_forward_markdown_report(output_dir)
    path = output_dir / "walk_forward_report.md"
    path.write_text(report, encoding="utf-8")
    return path

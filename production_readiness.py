from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FILES = [
    "project_paths.py",
    "mvp_config.py",
    "feature_engineering.py",
    "label_builder.py",
    "model_signals.py",
    "ensemble.py",
    "risk_management.py",
    "live_monitoring.py",
    "execution_audit.py",
    "validation.py",
    "walk_forward_runner.py",
    "run_research_pipeline.py",
]

NON_SOURCE_DIR_PREFIXES = (
    "outputs/",
    "__pycache__/",
    ".ipynb_checkpoints/",
)


FORBIDDEN_FEATURE_KEYWORDS = [
    "future",
    "label",
    "target",
    "next",
    "hit",
    "exit_reason",
    "exit_price",
    "exit_bar",
]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _parse_status_path(line: str) -> str:
    # git status --short lines look like " M file.py", "?? path", or "R  old -> new".
    path = line[3:] if len(line) > 3 else ""
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip()


def is_source_dirty(git_status_short: str | None) -> bool:
    if not git_status_short:
        return False
    for line in git_status_short.splitlines():
        path = _parse_status_path(line)
        if not path:
            continue
        if path.endswith(".pyc"):
            continue
        if path.startswith(NON_SOURCE_DIR_PREFIXES):
            continue
        return True
    return False


def collect_reproducibility_manifest(
    data_path: str | Path,
    config: dict,
    output_dir: str | Path,
    extra_files: list[str | Path] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)

    source_files = [Path(p) for p in SOURCE_FILES]
    if extra_files:
        source_files.extend(Path(p) for p in extra_files)
    source_hashes = {
        str(path): sha256_file(path)
        for path in source_files
        if path.exists() and path.is_file()
    }

    git_status = _run_git(["status", "--short"])
    source_dirty = is_source_dirty(git_status)
    manifest = {
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path) if data_path.exists() else None,
        "config_sha256": stable_json_hash(config),
        "source_hashes": source_hashes,
        "source_bundle_sha256": stable_json_hash(source_hashes),
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_branch": _run_git(["branch", "--show-current"]),
        "git_status_short": git_status,
        "git_dirty": bool(git_status),
        "source_git_dirty": bool(source_dirty),
        "non_source_dirty_ignored_for_live_gate": list(NON_SOURCE_DIR_PREFIXES),
    }
    with open(output_dir / "reproducibility_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def audit_feature_leakage(feature_cols: list[str], output_dir: str | Path | None = None) -> dict:
    suspicious = [
        col for col in feature_cols
        if any(keyword in col.lower() for keyword in FORBIDDEN_FEATURE_KEYWORDS)
    ]
    report = {
        "feature_count": int(len(feature_cols)),
        "forbidden_keywords": FORBIDDEN_FEATURE_KEYWORDS,
        "suspicious_feature_count": int(len(suspicious)),
        "suspicious_features": suspicious,
        "passed": len(suspicious) == 0,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "feature_leakage_audit.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_walk_forward_quality(output_dir: str | Path) -> tuple[dict, pd.DataFrame]:
    output_dir = Path(output_dir)
    with open(output_dir / "walk_forward_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    fold_summary = pd.read_csv(output_dir / "walk_forward_fold_summary.csv")
    return summary, fold_summary


def build_live_readiness_report(
    walk_forward_dir: str | Path,
    leakage_audit: dict,
    reproducibility_manifest: dict,
    config: dict,
    output_dir: str | Path | None = None,
) -> dict:
    walk_forward_summary, fold_summary = load_walk_forward_quality(walk_forward_dir)
    quality_report = walk_forward_summary.get("quality_report", {})
    observed = quality_report.get("observed", {})

    fold_count = int(walk_forward_summary.get("executed_fold_count", len(fold_summary)))
    total_trades = int(walk_forward_summary.get("total_trades", 0))
    median_alpha_return = _safe_float(walk_forward_summary.get("median_alpha_total_return"))
    median_buy_auc = _safe_float(walk_forward_summary.get("median_test_buy_auc"))
    median_sell_auc = _safe_float(walk_forward_summary.get("median_test_sell_auc"))
    median_drawdown = _safe_float(observed.get("median_max_drawdown"))
    stress_min_alpha = _safe_float(walk_forward_summary.get("stress_min_alpha_total_return"))
    stress_worst_drawdown = _safe_float(walk_forward_summary.get("stress_worst_alpha_drawdown"))
    shadow_report = None
    if output_dir is not None:
        shadow_path = Path(output_dir) / "shadow_monitoring_report.json"
        if shadow_path.exists():
            with shadow_path.open("r", encoding="utf-8") as f:
                shadow_report = json.load(f)
    require_shadow = bool(config.get("require_shadow_monitoring", True))
    shadow_status = shadow_report.get("status") if isinstance(shadow_report, dict) else None
    execution_audit = None
    if output_dir is not None:
        execution_path = Path(output_dir) / "execution_feasibility_audit.json"
        if execution_path.exists():
            with execution_path.open("r", encoding="utf-8") as f:
                execution_audit = json.load(f)
    execution_status = execution_audit.get("status") if isinstance(execution_audit, dict) else None

    live_gates = {
        "reproducibility_clean_source_tree": not reproducibility_manifest.get("source_git_dirty", True),
        "feature_leakage_passed": bool(leakage_audit.get("passed")),
        "execution_feasibility_passed": execution_status == "PASS",
        "walk_forward_quality_passed": bool(quality_report.get("all_passed", False)),
        "min_executed_folds": fold_count >= int(config.get("live_min_folds", 8)),
        "min_total_trades": total_trades >= int(config.get("live_min_total_trades", 200)),
        "positive_median_alpha_return": pd.notna(median_alpha_return) and median_alpha_return > 0,
        "median_buy_auc_above_live_floor": pd.notna(median_buy_auc) and median_buy_auc >= float(config.get("live_min_buy_auc", 0.53)),
        "median_sell_auc_above_live_floor": pd.notna(median_sell_auc) and median_sell_auc >= float(config.get("live_min_sell_auc", 0.53)),
        "drawdown_within_live_limit": pd.notna(median_drawdown) and median_drawdown >= float(config.get("live_max_median_drawdown", -0.05)),
        "stress_alpha_positive": pd.notna(stress_min_alpha) and stress_min_alpha > 0,
        "stress_drawdown_within_live_limit": pd.notna(stress_worst_drawdown) and stress_worst_drawdown >= float(config.get("live_max_median_drawdown", -0.05)),
        "shadow_monitoring_passed": (not require_shadow) or shadow_status == "PASS",
    }
    report = {
        "status": "PASS" if all(live_gates.values()) else "FAIL",
        "live_gates": live_gates,
        "observed": {
            "fold_count": fold_count,
            "total_trades": total_trades,
            "median_alpha_total_return": median_alpha_return,
            "median_test_buy_auc": median_buy_auc,
            "median_test_sell_auc": median_sell_auc,
            "median_max_drawdown": median_drawdown,
            "stress_min_alpha_total_return": stress_min_alpha,
            "stress_worst_alpha_drawdown": stress_worst_drawdown,
            "shadow_monitoring_status": shadow_status,
            "execution_feasibility_status": execution_status,
        },
        "quality_report": quality_report,
        "shadow_monitoring_report": shadow_report,
        "execution_feasibility_audit": execution_audit,
        "hard_rule": "No real capital deployment unless status == PASS and all artifacts are reviewed.",
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "live_readiness_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def summarize_stress_grid(stress_rows: list[dict], output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(output_dir / "stress_test_summary.csv", index=False)
    if len(stress_df):
        aggregate = {
            "scenario_count": int(len(stress_df)),
            "pass_positive_alpha_all": bool((stress_df.get("alpha_total_return", pd.Series(dtype=float)) > 0).all()),
            "min_alpha_total_return": _safe_float(stress_df.get("alpha_total_return", pd.Series(dtype=float)).min()),
            "max_alpha_drawdown": _safe_float(stress_df.get("alpha_max_drawdown", pd.Series(dtype=float)).min()),
            "min_trade_count": int(stress_df.get("trade_count", pd.Series(dtype=float)).min()) if "trade_count" in stress_df else 0,
        }
    else:
        aggregate = {"scenario_count": 0}
    with open(output_dir / "stress_test_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    return stress_df

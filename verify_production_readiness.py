from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_ROOT_FILES = [
    "pipeline_manifest.json",
    "run_config.json",
    "feature_cols.csv",
    "feature_leakage_audit.json",
    "market_data_cleaning_report.json",
    "execution_feasibility_audit.json",
    "reproducibility_manifest.json",
    "live_readiness_report.json",
]

REQUIRED_WALK_FORWARD_FILES = [
    "walk_forward_summary.json",
    "walk_forward_fold_summary.csv",
    "walk_forward_report.md",
    "methodology.json",
    "walk_forward_splits.csv",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_output_dir(output_dir: str | Path) -> tuple[dict, pd.DataFrame]:
    output_dir = Path(output_dir)
    wf_dir = output_dir / "walk_forward"

    checks = []

    for rel in REQUIRED_ROOT_FILES:
        path = output_dir / rel
        checks.append({"check": f"root_file_exists:{rel}", "passed": path.exists(), "detail": str(path)})

    for rel in REQUIRED_WALK_FORWARD_FILES:
        path = wf_dir / rel
        checks.append({"check": f"walk_forward_file_exists:{rel}", "passed": path.exists(), "detail": str(path)})

    leakage = _load_json(output_dir / "feature_leakage_audit.json") if (output_dir / "feature_leakage_audit.json").exists() else {}
    checks.append({
        "check": "feature_leakage_audit_passed",
        "passed": bool(leakage.get("passed", False)),
        "detail": f"suspicious_feature_count={leakage.get('suspicious_feature_count')}",
    })
    execution = _load_json(output_dir / "execution_feasibility_audit.json") if (output_dir / "execution_feasibility_audit.json").exists() else {}
    cleaning = _load_json(output_dir / "market_data_cleaning_report.json") if (output_dir / "market_data_cleaning_report.json").exists() else {}
    checks.append({
        "check": "execution_feasibility_audit_passed",
        "passed": bool(execution.get("status") == "PASS"),
        "detail": str(execution.get("observed", {})),
    })
    checks.append({
        "check": "market_data_cleaning_recorded",
        "passed": bool(cleaning.get("status") in {"PASS", "FAIL"}) and "dropped_rows" in cleaning.get("observed", {}),
        "detail": str(cleaning.get("observed", {})),
    })

    repro = _load_json(output_dir / "reproducibility_manifest.json") if (output_dir / "reproducibility_manifest.json").exists() else {}
    checks.extend(
        [
            {
                "check": "data_hash_recorded",
                "passed": bool(repro.get("data_sha256")),
                "detail": str(repro.get("data_sha256")),
            },
            {
                "check": "source_bundle_hash_recorded",
                "passed": bool(repro.get("source_bundle_sha256")),
                "detail": str(repro.get("source_bundle_sha256")),
            },
            {
                "check": "config_hash_recorded",
                "passed": bool(repro.get("config_sha256")),
                "detail": str(repro.get("config_sha256")),
            },
        ]
    )

    wf_summary = _load_json(wf_dir / "walk_forward_summary.json") if (wf_dir / "walk_forward_summary.json").exists() else {}
    fold_summary_path = wf_dir / "walk_forward_fold_summary.csv"
    fold_summary = pd.read_csv(fold_summary_path) if fold_summary_path.exists() else pd.DataFrame()
    checks.extend(
        [
            {
                "check": "fold_summary_non_empty",
                "passed": len(fold_summary) > 0,
                "detail": f"rows={len(fold_summary)}",
            },
            {
                "check": "stress_metrics_present",
                "passed": all(
                    col in fold_summary.columns
                    for col in ["stress_min_alpha_total_return", "stress_worst_alpha_drawdown"]
                ),
                "detail": ",".join(fold_summary.columns),
            },
            {
                "check": "quality_report_present",
                "passed": isinstance(wf_summary.get("quality_report"), dict),
                "detail": str(wf_summary.get("quality_report", {}).get("all_passed")),
            },
        ]
    )

    readiness = _load_json(output_dir / "live_readiness_report.json") if (output_dir / "live_readiness_report.json").exists() else {}
    live_gates = readiness.get("live_gates", {})
    shadow_path = output_dir / "shadow_monitoring_report.json"
    shadow = _load_json(shadow_path) if shadow_path.exists() else {}
    checks.extend(
        [
            {
                "check": "live_readiness_status_present",
                "passed": readiness.get("status") in {"PASS", "FAIL"},
                "detail": str(readiness.get("status")),
            },
            {
                "check": "live_readiness_all_gates_boolean",
                "passed": bool(live_gates) and all(isinstance(v, bool) for v in live_gates.values()),
                "detail": str(live_gates),
            },
            {
                "check": "execution_feasibility_gate_present",
                "passed": "execution_feasibility_passed" in live_gates,
                "detail": str(live_gates.get("execution_feasibility_passed")),
            },
            {
                "check": "hard_no_capital_rule_present",
                "passed": "No real capital" in readiness.get("hard_rule", ""),
                "detail": readiness.get("hard_rule", ""),
            },
            {
                "check": "shadow_monitoring_gate_present",
                "passed": "shadow_monitoring_passed" in live_gates,
                "detail": str(live_gates.get("shadow_monitoring_passed")),
            },
            {
                "check": "shadow_monitoring_report_status_valid_if_present",
                "passed": (not shadow_path.exists()) or shadow.get("status") in {"PASS", "FAIL"},
                "detail": str(shadow.get("status", "missing")),
            },
        ]
    )

    checks_df = pd.DataFrame(checks)
    report = {
        "output_dir": str(output_dir),
        "check_count": int(len(checks_df)),
        "passed_count": int(checks_df["passed"].sum()) if len(checks_df) else 0,
        "all_checks_passed": bool(checks_df["passed"].all()) if len(checks_df) else False,
        "live_readiness_status": readiness.get("status"),
        "note": "Verification checks artifact completeness and gate presence. It does not override live readiness.",
    }
    return report, checks_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify research pipeline production-readiness artifacts.")
    parser.add_argument("output_dir", help="Pipeline output directory containing walk_forward/ and readiness files.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report, checks_df = verify_output_dir(output_dir)
    checks_df.to_csv(output_dir / "production_readiness_checks.csv", index=False)
    with (output_dir / "production_readiness_verification.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
